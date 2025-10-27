import torch
import torch.nn as nn
import torch.nn.functional as F

from models import DoubleConv, default_bn, default_act, TimesC, OutMatrixC, UpMB, DownMB, get_padding_mode

class DepthPred(nn.Module):
    """(convolution => [LN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, num_hidden_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert num_hidden_layers >= 1, "num_hidden_layers should be at least 1"
        assert 2 ** (num_hidden_layers - 1) < in_channels, "in_channels should be greater than 2 ** (num_hidden_layers - 1)"
        self.num_hidden_layers = num_hidden_layers
        
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding="same", padding_mode=get_padding_mode(), bias=True),
            default_bn(in_channels),
            default_act(),
        ]
        
        if num_hidden_layers > 1:
            for i in range(num_hidden_layers - 1):
                current_middle_channels = in_channels // (2 ** i)
                next_middle_channels = in_channels // (2 ** (i + 1))
                
                layers.append(
                    nn.Conv2d(current_middle_channels, next_middle_channels, kernel_size=3, padding="same", padding_mode=get_padding_mode(), bias=True)
                )
                layers.append(default_bn(next_middle_channels))
                layers.append(default_act())

            current_middle_channels = in_channels // (2 ** (num_hidden_layers - 1))
            layers.append(
                nn.Conv2d(current_middle_channels, out_channels, kernel_size=1, padding="same", padding_mode=get_padding_mode(), bias=True),
                default_act(act_type='relu'),
            )
        else:
            layers.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding="same", padding_mode=get_padding_mode(), bias=False),
                default_bn(in_channels),
                default_act(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding="same", padding_mode=get_padding_mode(), bias=False),
                default_act(act_type='relu'),
            )
        
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)# + x_ie
    
class DepthEffWNet(nn.Module):
    def __init__(self, n_channels, out_depth=1, inc_f0=1, bilinear=False, n_lyr=4, ch1=24, c_is_const=False, c_is_scalar=False, outer_conv=False, depthnet_hidden_layers=3):
        super().__init__()
        self.n_channels = n_channels
        self.out_depth = out_depth
        self.inc_f0 = inc_f0
        self.bilinear = bilinear
        self.outer_conv = outer_conv
        self.depthnet_hidden_layers = depthnet_hidden_layers
        
        n_chs = [ch1 * (2 ** power) for power in range(n_lyr+1)]
        n_rep_dn = [2, 2, 4, 4, 6]
        lyr_ts = ["fused", "fused", "depthwise", "depthwise", "depthwise"]
        n_rep_up = [6, 4, 4, 2, 2]
        expans = [1, 2, 4, 4, 6]
        pool_szs = [3, 3, 2, 2, 5]
        factor = 2 if bilinear else 1

        self.mparams = {"n_lyr": n_lyr, "bilinear": bilinear, "n_chs": n_chs, "n_rep_dn": n_rep_dn, "lyr_ts": lyr_ts, "n_rep_up": n_rep_up, "expans": expans, "pool_szs": pool_szs, "factor": factor}

        self.inc = DoubleConv(n_channels, n_chs[0])
        self.downs = nn.ModuleList()

        for i in range(n_lyr):
            out_chnl = n_chs[i+1] // factor if i == n_lyr-1 else n_chs[i+1]
            lyr = DownMB(n_chs[i], out_chnl, lyr_ts[i], expansion=expans[i], n_repeats=n_rep_dn[i], pool_size=pool_szs[i])
            self.downs.append(lyr)
        
        self.ups = self.ups_builder()
        
        self.depth_pred = DepthPred(n_chs[0], out_depth, num_hidden_layers=depthnet_hidden_layers)

    def freeze_encoder(self):
        for param in self.inc.parameters():
            param.requires_grad = False
            
        for param in self.downs.parameters():
            param.requires_grad = False
            
        for param in self.ups.parameters():
            param.requires_grad = False

    def ups_builder(self):
        ups = nn.ModuleList()
        for i in range(self.mparams["n_lyr"]):
            rev_i = self.mparams["n_lyr"]-i-1
            out_chnl = self.mparams["n_chs"][rev_i] if i == self.mparams["n_lyr"]-1 else self.mparams["n_chs"][rev_i] // self.mparams["factor"]

            lyr = UpMB(self.mparams["n_chs"][rev_i+1], out_chnl, self.mparams["lyr_ts"][rev_i], expansion=self.mparams["expans"][rev_i], n_repeats=self.mparams["n_rep_up"][i], bilinear=self.mparams["bilinear"], scale_factor=self.mparams["pool_szs"][rev_i])
            ups.append(lyr)
        
        return ups


    def forward(self, x):
        x0 = x
        
        x1 = self.inc(x0)
        xs = [x1]

        for dn in self.downs:
            tmp_x = dn(xs[-1])
            # print("dn")
            # print(tmp_x.shape)
            xs.append(tmp_x)

        # print()

        x_ie = xs[-1]
        rev_xs = xs[::-1]
        for up, xr in zip(self.ups, rev_xs[1:]):
            # print("x_ie", x_ie.shape)
            # print("xr", xr.shape)
            x_ie = up(x_ie, xr)
            # print(x_ie.shape)
        
        depth = self.depth_pred(x_ie)
        return depth

        
class DepthFromEventsModel(nn.Module):
    def __init__(self, n_channels, out_depth, inc_f0=1, bilinear=False, n_lyr=4, ch1=24, c_is_const=False, c_is_scalar=False, outer_conv=False, depthnet_hidden_layers=2):
        super().__init__()
        self.effwnet = DepthEffWNet(n_channels, out_depth, inc_f0, bilinear, n_lyr, ch1, c_is_const, c_is_scalar, outer_conv, depthnet_hidden_layers=depthnet_hidden_layers)
    
    def freeze_encoder(self):
        self.effwnet.freeze_encoder()
        
    def forward(self, event_data):
        pred = self.effwnet(event_data)
        return pred
    
    
class SSLDepthModel(nn.Module):
    def __init__(self, n_channels, out_depth, inc_f0=1, bilinear=False, n_lyr=4, ch1=24, c_is_const=False, c_is_scalar=False, outer_conv=False, depthnet_hidden_layers=2):
        '''
        NOTE: Training would be like this
        
        # idx: Eg. some number in (160, 1580] for split 2 Indoor Flying.
        
        ssl_event_data_for_some_idx: 5 Channeled tensor (B, 5, H, W)
        corresponding_image_ground_truth_for_the_same_idx: N Channeled tensor (B, N, H, W)
        corresponding_depth_ground_truth_for_the_same_idx: 1 Channeled tensor (B, 1, H, W)

        # For example:
        model_input = torch.cat([
            ssl_event_data_for_some_idx, 
            corresponding_image_ground_truth_for_the_same_idx, 
            corresponding_depth_ground_truth_for_the_same_idx], 
        dim=1)  
        # Then, "model_input" will be of shape (B, 5+N+1, H, W)
        
        pred = model(model_input) # pred will be of shape (B, 1, H, W)
        loss = loss_fn(pred, corresponding_depth_ground_truth)
        '''
        super().__init__()
        self.effwnet = DepthEffWNet(n_channels, out_depth, inc_f0, bilinear, n_lyr, ch1, c_is_const, c_is_scalar, outer_conv, depthnet_hidden_layers=depthnet_hidden_layers)
    
    def freeze_encoder(self):
        self.effwnet.freeze_encoder()
        
    def forward(self, ssl_event_data):
        pred = self.effwnet(ssl_event_data)
        return pred
    