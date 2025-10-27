import numpy as np
from models import EffWNet
import torch
import argparse
from mvsec_helper import *
from torchsummary import summary
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from mvsec_dataset import MVSECDataset, MVSECSampler, SingleMVSECSampler
import math
from tqdm import tqdm
import torchvision.utils as vutils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--data_dir', default='/root/data/MVSEC', type=str)
    parser.add_argument('--split', default=1, type=int)
    parser.add_argument('--scenario', default='indoor_flying', type=str)
    parser.add_argument('--loss_type', default='l1', type=str)
    args = parser.parse_args()
    
    device = 'cuda'
    data_dir = args.data_dir
    split = args.split
    scenario = args.scenario
    model_path = args.model_path
    model_dir = model_path.rsplit('/', 1)[0]
    
    loss_type = args.loss_type
    assert os.path.exists(model_path), f'Model path does not exist: {model_path}' 
    
    # Create Dataset and DataLoader
    event_transforms = transforms.Compose([
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToDtype(torch.float32, scale=False),
    ])
    
    image_transforms = transforms.Compose([
        transforms.ToImage(),
        # transforms.ToDtype(torch.uint8, scale=True),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToDtype(torch.float32, scale=True),
    ])
    dataset = MVSECDataset(data_dir=data_dir, scenario=scenario, split=split, is_training=False,
                           event_transforms=event_transforms, image_transforms=image_transforms)
    
    _sampler = SingleMVSECSampler(scenario=scenario, split=split, is_training=False)
    sampler = MVSECSampler(sampler=_sampler, batch_size=1, drop_last=False)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True, batch_sampler=sampler)
    dataloader = DataLoader(dataset, num_workers=0, pin_memory=True, batch_sampler=sampler)
    
    # Create online model
    model = EffWNet(n_channels=3, out_depth=1, inc_f0=1, bilinear=True, n_lyr=4, ch1=24, c_is_const=False, c_is_scalar=False)
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)
    model.to(device)
    
    model.eval()
    count = 0
    for (event_data, image_data) in tqdm(dataloader, total=len(dataloader)):
        
        event_data_left = event_data['left'].to(device)
        event_data_right = event_data['right'].to(device)
        
        image_at_t0_left, image_at_t1_left = image_data['left']
        image_at_t0_right, image_at_t1_right = image_data['right']
        image_at_t0_left, image_at_t1_left = image_at_t0_left.to(device), image_at_t1_left.to(device)
        image_at_t0_right, image_at_t1_right = image_at_t0_right.to(device), image_at_t1_right.to(device)
        
        event_data_left_ch0 = event_data_left[:, 0:1, :, :]
        event_data_left_ch1 = event_data_left[:, 1:2, :, :]
        event_data_left_ch2 = event_data_left[:, 2:3, :, :]
        
        events = torch.cat([event_data_left, event_data_right], dim=0)
        t0_images = torch.cat([image_at_t0_left, image_at_t0_right], dim=0)
        t1_images = torch.cat([image_at_t1_left, image_at_t1_right], dim=0)

        output = model(events, t0_images)
        
        if loss_type == 'mse':
            loss = torch.square(output - t1_images)
        elif loss_type == 'l1':
            loss = torch.abs(output - t1_images)
        elif loss_type == 'ph':
            c = 0.00054 * math.sqrt(math.prod(t0_images.shape[1:]))
            loss: torch.Tensor = ((output - t1_images).square() + (c ** 2)).sqrt() - c
        
        loss = loss.mean()

        if count == 0:
            _image_at_t0_left = (image_at_t0_left * 255).clamp(0, 255).to(torch.uint8).contiguous()
            _image_at_t1_left = (image_at_t1_left * 255).clamp(0, 255).to(torch.uint8).contiguous()
            vutils.save_image(_image_at_t0_left.float(), f'{model_dir}/t0_left.png', normalize=True)
            vutils.save_image(_image_at_t1_left.float(), f'{model_dir}/t1_left.png', normalize=True)
            vutils.save_image(event_data_left_ch0.float(), f'{model_dir}/event_ch0_left.png', normalize=True)
            vutils.save_image(event_data_left_ch1.float(), f'{model_dir}/event_ch1_left.png', normalize=True)
            vutils.save_image(event_data_left_ch2.float(), f'{model_dir}/event_ch2_left.png', normalize=True)
            pred_image_at_t1_left = (output[0:1, :, :, :] * 255).clamp(0, 255).to(torch.uint8).contiguous()
            vutils.save_image(pred_image_at_t1_left.float(), f'{model_dir}/t1_left_pred.png', normalize=True)
            print(f'loss at sample {count}: {loss.item()}')
        
        count += 1

