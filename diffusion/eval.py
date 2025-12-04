# from ..utils.metric import AverageMeter, MeanDepthError, MeanDisparityError, NPixelAccuracy
# from collections import OrderedDict
import numpy as np
import torch
# from models import SSLEventModel
# from mvsec_dataset import MVSECDataset, MVSECSampler, SingleMVSECSampler

gt_path = "/root/code/EventDiffusion/workdir/ema_0.9993_035000/sample_35000/split=test/ground_truth/steps=1/samples_4244x256x256x3_nfe0.npz"
gen_path = "/root/code/EventDiffusion/workdir/ema_0.9993_035000/sample_35000/split=test/euler/steps=12/samples_4244x256x256x3_nfe13.npz"
evaluating_set = "test"

# log_dict = OrderedDict([('Loss', AverageMeter(string_format='%6.3lf')), 
#                         ('MDeE', MeanDepthError(average_by='image', string_format='%6.3lf')), 
#                         ('MDisE', MeanDisparityError(average_by='image', string_format='%6.3lf')), 
#                         ('1PA', NPixelAccuracy(n=1, average_by='image', string_format='%6.3lf'))
#                         ])


gt = torch.from_numpy(np.load(gt_path)['arr_0']).float()
gen = torch.from_numpy(np.load(gen_path)['arr_0']).float()

mse_loss = torch.square(gen - gt).mean()
print(mse_loss.item())

abs_error = torch.abs(gen - gt).mean()
print(abs_error.item())