import numpy as np
from models import EffWNet
import torch
import argparse
from mvsec_helper import *
from torchsummary import summary
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from mvsec_dataset import MVSECDataset, MVSECSampler, SingleMVSECSampler
from copy import deepcopy
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
    
    # print(len(dataloader))
    # exit()
    for epoch in range(1, epochs + 1):
        # print(f'Starting epoch {epoch}/{epochs}...')
        # for (event_data, image_data) in dataloader:
        for (event_data, image_data) in tqdm(dataloader, total=len(dataloader), desc=f'Epoch {epoch}/{epochs}'):
            optimizer.zero_grad()
            
            event_data_left = event_data['left'].to(device)
            event_data_right = event_data['right'].to(device)
            
            image_at_t0_left, image_at_t1_left = image_data['left']
            image_at_t0_right, image_at_t1_right = image_data['right']
            image_at_t0_left, image_at_t1_left = image_at_t0_left.to(device), image_at_t1_left.to(device)
            image_at_t0_right, image_at_t1_right = image_at_t0_right.to(device), image_at_t1_right.to(device)

            # print('event_data_left:', event_data_left.shape)
            # print('event_data_right:', event_data_right.shape)
            # print('image_at_t0_left:', image_at_t0_left.shape)
            # print('image_at_t1_left:', image_at_t1_left.shape)
            # print('image_at_t0_right:', image_at_t0_right.shape)
            # print('image_at_t1_right:', image_at_t1_right.shape)
            # exit()
            
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
                loss: torch.Tensor = ((output - t1_images).square() + (c**2)).sqrt() - c
            
            loss = loss.mean()
            loss.backward()
            optimizer.step()
        scheduler.step()
        
    # for epoch in range(1, epochs + 1):
    #     # print(f'Starting epoch {epoch}/{epochs}...')
    #     # for (event_data, image_data) in dataloader:
    #     for (event_data, image_data) in tqdm(dataloader, total=len(dataloader), desc=f'Epoch {epoch}/{epochs + 1}'):
    #         optimizer.zero_grad()
            
    #         events_left = event_data['left']
    #         events_right = event_data['right']

    #         image_t0_left_split2, image_t1_left_split2 = image_data['left']['2']
    #         image_t0_right_split2, image_t1_right_split2 = image_data['right']['2']
            
    #         image_t0_left_split3, image_t1_left_split3 = image_data['left']['3']
    #         image_t0_right_split3, image_t1_right_split3 = image_data['right']['3']

    #         # Concatenate left and right events for the splits:
    #         split_2_events = torch.cat([ events_left[0][0], events_right[0][0] ], dim=0).to(device)
    #         split_3_events = torch.cat([ events_left[1][0], events_right[1][0] ], dim=0).to(device)

    #         # Concatenate left and right images for the splits:
    #         split_2_images_t0 = torch.cat([ image_t0_left_split2[0], image_t0_right_split2[0] ], dim=0)
    #         split_2_images_t1 = torch.cat([ image_t1_left_split2[0], image_t1_right_split2[0] ], dim=0)
    #         split_3_images_t0 = torch.cat([ image_t0_left_split3[0], image_t0_right_split3[0] ], dim=0)
    #         split_3_images_t1 = torch.cat([ image_t1_left_split3[0], image_t1_right_split3[0] ], dim=0)

    #         # Concatenate t0 and t1 images for the splits:
    #         split_2_images = torch.cat([ split_2_images_t0, split_2_images_t1 ], dim=1).to(device)
    #         split_3_images = torch.cat([ split_3_images_t0, split_3_images_t1 ], dim=1).to(device)

    #         # print('split_2_events:', split_2_events.shape)
    #         # print('split_3_events:', split_3_events.shape)
    #         # print('split_2_images:', split_2_images.shape)
    #         # print('split_3_images:', split_3_images.shape)
    #         # exit()
            
    #         # Concatenate all event data:
    #         events = torch.cat([split_2_events, split_3_events], dim=0).to(device)
            
    #         t0_images = torch.cat([split_2_images_t0, split_3_images_t0], dim=0).to(device)
    #         t1_images = torch.cat([split_2_images_t1, split_3_images_t1], dim=0).to(device)
    #         # print('events:', events.shape)
    #         # print('t0_images:', t0_images.shape)
    #         # print('split_2_images_t0:', split_2_images_t0.shape)
    #         # print('split_3_images_t0:', split_3_images_t0.shape)
    #         # exit()
            
    #         output = model(events, t0_images)
            
    #         # loss = torch.nn.functional.mse_loss(output, t1_images)
    #         # c = 0.00054 * math.sqrt(math.prod(t0_images.shape[1:]))
    #         # loss: torch.Tensor = ((output - t1_images).square() + (c**2)).sqrt() - c
    #         loss = torch.abs(output - t1_images)
    #         loss = loss.mean()
            
    #         loss.backward()
            
    #         optimizer.step()
    #     scheduler.step()

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
    
    # ev_rep_data = []
    
    # for loc in ['left', 'right']:
    #     for exp in EXPERIMENTS[scenario]:
    #         if exp == split:
    #             continue

    #         ev_rep_dir = Path(data_dir) / f'{scenario}/{scenario}{exp}/evrep_train/evrep_{loc}.npy'
    #         assert ev_rep_dir.exists(), ev_rep_dir
            
    #         ev_rep = torch.from_numpy(np.load(ev_rep_dir)).float()
    #         # print(ev_rep.min(), ev_rep.max())
    #         ev_rep_data.append(ev_rep)
    
    # for _data in ev_rep_data:
    #     print(_data.shape)
    
    
    
