import torch
import torch.nn as nn
import torch.nn.functional as F

from models import DoubleConv, default_bn, default_act, TimesC, OutMatrixC, UpMB, DownMB, get_padding_mode
from smnet_model.stereo_matching import StereoMatchingNet

class DisparityPred(nn.Module):
    """(convolution => [LN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, num_hidden_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert num_hidden_layers >= 1, "num_hidden_layers should be at least 1"
        assert 2 ** (num_hidden_layers - 1) < in_channels, "in_channels should be greater than 2 ** (num_hidden_layers - 1)"
        self.num_hidden_layers = num_hidden_layers
        
        layers = [
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding="same", padding_mode=get_padding_mode(), bias=False),
            default_bn(in_channels),
            default_act(),
        ]
        
        if num_hidden_layers > 1:
            for i in range(num_hidden_layers - 1):
                current_middle_channels = in_channels // (2 ** i)
                next_middle_channels = in_channels // (2 ** (i + 1))
                
                layers.append(
                    nn.Conv2d(current_middle_channels, next_middle_channels, kernel_size=3, padding="same", padding_mode=get_padding_mode(), bias=False)
                )
                layers.append(default_bn(next_middle_channels))
                layers.append(default_act())

            current_middle_channels = in_channels // (2 ** (num_hidden_layers - 1))
            layers.extend([
                nn.Conv2d(current_middle_channels, out_channels, kernel_size=1, padding="same", padding_mode=get_padding_mode(), bias=False),
                default_act(act_type='relu'),
            ])
        else:
            layers.extend([
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding="same", padding_mode=get_padding_mode(), bias=False),
                default_bn(in_channels),
                default_act(),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding="same", padding_mode=get_padding_mode(), bias=False),
                default_act(act_type='relu'),
            ])
        
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)# + x_ie
    
class EventEffWNetEncoder(nn.Module):
    def __init__(self, n_channels, out_depth=1, inc_f0=1, bilinear=False, n_lyr=4, ch1=24, c_is_const=False, c_is_scalar=False, outer_conv=False, dispnet_hidden_layers=3):
        super().__init__()
        self.n_channels = n_channels
        self.out_depth = out_depth
        self.inc_f0 = inc_f0
        self.bilinear = bilinear
        self.outer_conv = outer_conv
        self.depthnet_hidden_layers = dispnet_hidden_layers
        
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
        
        # print('x0', x0.shape)
        x1 = self.inc(x0)
        # print('x1', x1.shape)
        xs = [x1]

        for dn in self.downs:
            tmp_x = dn(xs[-1])
            # print("dn", tmp_x.shape)
            xs.append(tmp_x)

        # print()

        x_ie = xs[-1]
        rev_xs = xs[::-1]
        for up, xr in zip(self.ups, rev_xs[1:]):
            # print("x_ie", x_ie.shape)
            # print("xr", xr.shape)
            x_ie = up(x_ie, xr)
            # print(x_ie.shape)
        # exit()
        return x_ie

        
class DispFromEventsModel(nn.Module):
    def __init__(self, use_images:bool, n_channels, out_depth, inc_f0=1, bilinear=False, n_lyr=4, ch1=24, c_is_const=False, c_is_scalar=False, outer_conv=False, depthnet_hidden_layers=2):
        super().__init__()
        self.use_images = use_images
        
        self.effwnet = EventEffWNetEncoder(n_channels, out_depth, inc_f0, bilinear, n_lyr, ch1, c_is_const, c_is_scalar, outer_conv, dispnet_hidden_layers=depthnet_hidden_layers)
        
        n_chs = [ch1 * (2 ** power) for power in range(n_lyr+1)]
        depth_effwnet_n_channels = 2 * n_chs[0]
        
        if use_images:
            self.img_effwnet = EventEffWNetEncoder(1, out_depth, inc_f0, bilinear, n_lyr, ch1, 
                                                    c_is_const, c_is_scalar, outer_conv, 
                                                    dispnet_hidden_layers=depthnet_hidden_layers+1)
            depth_effwnet_n_channels += 2 * n_chs[0]
            
            
        self.disp_effwnet = EventEffWNetEncoder(depth_effwnet_n_channels, out_depth, inc_f0, bilinear, n_lyr, ch1,
                                                 c_is_const, c_is_scalar, outer_conv,
                                                 dispnet_hidden_layers=depthnet_hidden_layers+1)
        
        self.disp_pred = DisparityPred(n_chs[0], out_depth, num_hidden_layers=depthnet_hidden_layers)

        # self.disp_pred = DisparityPred(2 * n_chs[0], out_depth, num_hidden_layers=depthnet_hidden_layers)

        # self.stereo_matching = StereoMatchingNet(max_disp=192, in_channels=2 * n_chs[0], num_downsample=2)

    def freeze_encoder(self):
        with torch.no_grad():
            for param in self.effwnet.parameters():
                param.requires_grad = False
    
    def forward(self, event_data, image_input=None):
        if self.use_images:
            assert image_input is not None, "image_input should be provided when use_images is True."

        event_latents = self.effwnet(event_data)
        assert event_latents.shape[0] % 2 == 0, "Batch size of event_latents should be even."
        # print('event_latents', event_latents.shape)
        event_latents = event_latents.reshape(-1, 2 * event_latents.shape[1], event_latents.shape[2], event_latents.shape[3])
        # print('after reshape event_latents', event_latents.shape)
        if self.use_images:
            image_latents = self.img_effwnet(image_input)
            assert image_latents.shape[0] % 2 == 0, "Batch size of image_latents should be even."
            
            image_latents = image_latents.reshape(-1, 2 * image_latents.shape[1], image_latents.shape[2], image_latents.shape[3])
            
            event_latents = torch.cat([event_latents, image_latents], dim=1)
            # event_latents = event_latents + image_latents
            
        disp_latents = self.disp_effwnet(event_latents)
        
        disp = self.disp_pred(disp_latents)

        return event_latents, disp
        
        
class SSLDisparityModel(nn.Module):
    def __init__(self, n_channels, out_depth, inc_f0=1, bilinear=False, n_lyr=4, ch1=24, c_is_const=False, c_is_scalar=False, outer_conv=False, dispnet_hidden_layers=2):
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
        self.effwnet = EventEffWNetEncoder(n_channels, out_depth, inc_f0, bilinear, n_lyr, ch1, c_is_const, c_is_scalar, outer_conv, dispnet_hidden_layers=dispnet_hidden_layers)
        
        n_chs = [ch1 * (2 ** power) for power in range(n_lyr+1)]
        self.disp_pred = DisparityPred(n_chs[0], out_depth, num_hidden_layers=dispnet_hidden_layers)
        
    def freeze_encoder(self):
        with torch.no_grad():
            for param in self.effwnet.parameters():
                param.requires_grad = False
        
    def forward(self, ssl_event_data):
        event_latents = self.effwnet(ssl_event_data)
        assert event_latents.shape[0] % 2 == 0, "Batch size of event_latents should be even."
        event_latents = event_latents.reshape(-1, 2 * event_latents.shape[1], event_latents.shape[2], event_latents.shape[3])
        disp = self.disp_pred(event_latents)
        return event_latents, disp
    