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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--workdir', default='/root/code/EventDiffusion/', type=str)
    parser.add_argument('--date', type=str)
    parser.add_argument('--data_dir', default='/root/data/MVSEC', type=str)
    parser.add_argument('--split', default=1, type=int)
    parser.add_argument('--scenario', default='indoor_flying', type=str)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--bs', default=4, type=int)
    parser.add_argument('--loss_type', default='l1', type=str)
    parser.add_argument('--resume_epoch', default=-1, type=int)
    parser.add_argument('--save_freq', default=1, type=int)
    args = parser.parse_args()
    
    device = 'cuda'
    loss_type = args.loss_type
    split = args.split
    scenario = args.scenario
    data_dir = args.data_dir
    epochs = args.epochs
    learning_rate = args.lr
    batch_size = args.bs
    workdir = args.workdir + f'experiments/{args.date}/mvsec/{loss_type}_loss/scenario_{scenario}/split_{split}'
    resume_epoch = args.resume_epoch
    save_freq = args.save_freq
    if not os.path.exists(workdir):
        os.makedirs(workdir)
    
    assert loss_type in ['mse', 'l1', 'ph'], f'Unknown loss type: {loss_type}'

    with open(f'{workdir}/log.txt', mode='w') as f:
        if resume_epoch > 0:
            f.write(f'Resuming from Epoch: {resume_epoch}\n')
        f.write(f'Experiment Workdir: {workdir}\n')
        f.write(f'Scenario: {scenario}\n')
        f.write(f'Training For Split: {split}\n')
        f.write(f'Loss Type: {loss_type}\n')
        f.write(f'Batch Size: {batch_size}\n')
        f.write(f'Epochs: {epochs}\n')
        f.write(f'Learning Rate: {learning_rate}\n')
        f.write(f'Save Frequency: Every {save_freq} Epoch(s)\n')
        f.write('--'*20 + '\n\n')
    
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
    dataset = MVSECDataset(data_dir=data_dir, scenario=scenario, split=split, 
                           event_transforms=event_transforms, image_transforms=image_transforms)
    
    _sampler = SingleMVSECSampler(scenario=scenario, split=split)
    sampler = MVSECSampler(sampler=_sampler, batch_size=batch_size, drop_last=False)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True, batch_sampler=sampler)
    dataloader = DataLoader(dataset, num_workers=1, pin_memory=True, batch_sampler=sampler)
    
    # Create online model
    model = EffWNet(n_channels=3, out_depth=1, inc_f0=1, bilinear=True, n_lyr=4, ch1=24, c_is_const=False, c_is_scalar=False)

    model_stats = summary(model, (1, 3, 260, 346), f0=torch.ones((1, 3, 260, 346)), batch_dim=None, verbose=False, device=device)
    with open(f'{workdir}/model_info.txt', mode='w') as f:
        f.write(str(model_stats))
    
    model.to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    
    if resume_epoch > 0:
        print(f'Loading model, opt, and scheduler from epoch {resume_epoch}...')
        model.load_state_dict(torch.load(f'{workdir}/model_epoch_{resume_epoch}.pt'))
        optimizer.load_state_dict(torch.load(f'{workdir}/optimizer_epoch_{resume_epoch}.pt'))
        scheduler.load_state_dict(torch.load(f'{workdir}/scheduler_epoch_{resume_epoch}.pt'))
    
    for epoch in range(1, epochs + 1):
        
        for (event_data, image_data) in tqdm(dataloader, total=len(dataloader), desc=f'Epoch {epoch}/{epochs}'):
            optimizer.zero_grad()
            
            event_data_left = event_data['left'].to(device)
            event_data_right = event_data['right'].to(device)
            
            image_at_t0_left, image_at_t1_left = image_data['left']
            image_at_t0_right, image_at_t1_right = image_data['right']
            image_at_t0_left, image_at_t1_left = image_at_t0_left.to(device), image_at_t1_left.to(device)
            image_at_t0_right, image_at_t1_right = image_at_t0_right.to(device), image_at_t1_right.to(device)
            
            events = torch.cat([event_data_left, event_data_right], dim=0)
            t0_images = torch.cat([image_at_t0_left, image_at_t0_right], dim=0)
            t1_images = torch.cat([image_at_t1_left, image_at_t1_right], dim=0)
            # print('events:', events.min(), events.max(), events.shape)
            # print('t0_images:', t0_images.min(), t0_images.max(), t0_images.shape)
            # print('t1_images:', t1_images.min(), t1_images.max(), t1_images.shape)
            # exit()

            output = model(events, t0_images)
            
            if loss_type == 'mse':
                loss = torch.square(output - t1_images)
            elif loss_type == 'l1':
                loss = torch.abs(output - t1_images)
            elif loss_type == 'ph':
                c = 0.00054 * math.sqrt(math.prod(t0_images.shape[1:]))
                loss: torch.Tensor = ((output - t1_images).square() + (c ** 2)).sqrt() - c
            
            loss = loss.mean()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # print(f'Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}')
        msg = f'Epoch {epoch}/{epochs}: Loss = {loss.item():.6f}'
        print(msg)
        with open(f'{workdir}/log.txt', mode='a') as f:
            f.write(msg + '\n')
        if epoch % save_freq == 0:
            print(f'Epoch {epoch}: Saving model, opt, and scheduler...')
            torch.save(model.state_dict(), f'{workdir}/model_epoch_{epoch}.pt')
            torch.save(optimizer.state_dict(), f'{workdir}/optimizer_epoch_{epoch}.pt')
            torch.save(scheduler.state_dict(), f'{workdir}/scheduler_epoch_{epoch}.pt')        
