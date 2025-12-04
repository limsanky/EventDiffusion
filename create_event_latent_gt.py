from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from models import SSLEventModel
from mvsec_dataset import MVSECDataset, MVSECSampler, SingleMVSECSampler
import torch

data_dir = '/root/data/MVSEC'
split = 1
scenario = 'indoor_flying'
batch_size = 64
device = 'cuda'

event_model_path = '/root/code/EventDiffusion/pretrained_models/l1_loss/model_epoch_300.pt'
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

disp_transforms = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=False),
])

dataset = MVSECDataset(data_dir=data_dir, scenario=scenario, split=split, is_training=True,
                        event_transforms=event_transforms, image_transforms=image_transforms, 
                        get_disparity=True, disparity_transforms=disp_transforms)
test_dataset = MVSECDataset(data_dir=data_dir, scenario=scenario, split=split, is_training=False,
                        event_transforms=event_transforms, image_transforms=image_transforms, 
                        get_disparity=True, disparity_transforms=disp_transforms)

sampler = MVSECSampler(sampler=SingleMVSECSampler(scenario=scenario, split=split, is_training=True), 
                        batch_size=batch_size, drop_last=False)
test_sampler = MVSECSampler(sampler=SingleMVSECSampler(scenario=scenario, split=split, is_training=False), 
                            batch_size=batch_size, drop_last=False)

data = DataLoader(dataset, num_workers=1, pin_memory=True, batch_sampler=sampler)
test_data = DataLoader(test_dataset, num_workers=1, pin_memory=True, batch_sampler=test_sampler)


event_model = SSLEventModel(n_channels=3, out_depth=1, bilinear=True, n_lyr=4, ch1=24, c_is_const=False, c_is_scalar=False)
event_model.load_state_dict(torch.load(event_model_path))
event_model.to(device)

event_model.eval()

latent_1, latent_2, latent_3 = [], [], []
all_disp_data = []

for (event_data, image_data, depth_data, disp_data, index_data) in data:
    assert depth_data is not None
    
    event_data_left = event_data['left'].to(device)
    event_data_right = event_data['right'].to(device)
    
    image_at_t0_left, image_at_t1_left = image_data['left']
    image_at_t0_right, image_at_t1_right = image_data['right']
    
    events = torch.cat([event_data_left, event_data_right], dim=0).to(device)
    t0_images = torch.cat([image_at_t0_left, image_at_t0_right], dim=0).to(device)
    t1_images = torch.cat([image_at_t1_left, image_at_t1_right], dim=0).to(device)
    
    with torch.no_grad():
        event_latents = event_model(events, t0_images)[0]
    
    # print('event_latents:', event_latents[0].shape, event_latents[1].shape, event_latents[2].shape)
    # print(image_at_t0_left.shape, image_at_t1_left.shape)
    # exit()
    latent_1.append(event_latents[0].cpu())
    latent_2.append(event_latents[1].cpu())
    latent_3.append(event_latents[2].cpu())
    
    # Convert back to original disparity values since we saved as (disp*256).to(uint16) in the create_mvsec_dataset.py script
    disp_data = disp_data / 256.0
    # Normalize to [0, 1] range for training
    disp_data = disp_data / 255.0
    all_disp_data.append(disp_data.cpu())
    

latent_1 = torch.cat(latent_1, dim=0).to(device)
latent_2 = torch.cat(latent_2, dim=0).to(device)
latent_3 = torch.cat(latent_3, dim=0).to(device)

all_disp_data = torch.cat(all_disp_data, dim=0).to(device)

print('latent_1:', latent_1.shape)
print('latent_2:', latent_2.shape)
print('latent_3:', latent_3.shape)
print('all_disp_data:', all_disp_data.shape)

print('\nStatistics:')
print('latent_1: min', latent_1.min().item(), 'max', latent_1.max().item())
print('latent_2: min', latent_2.min().item(), 'max', latent_2.max().item())
print('latent_3: min', latent_3.min().item(), 'max', latent_3.max().item())
print('all_disp_data: min', all_disp_data.min().item(), 'max', all_disp_data.max().item())

latent_1 = (latent_1 - latent_1.min()) / (latent_1.max() - latent_1.min())
latent_2 = (latent_2 - latent_2.min()) / (latent_2.max() - latent_2.min())
latent_3 = (latent_3 - latent_3.min()) / (latent_3.max() - latent_3.min())

print('\nAfter normalization to [0, 1]:')
print('latent_1: mean', latent_1.mean().item(), 'std', latent_1.std().item())
print('latent_2: mean', latent_2.mean().item(), 'std', latent_2.std().item())
print('latent_3: mean', latent_3.mean().item(), 'std', latent_3.std().item())
print('all_disp_data: mean', all_disp_data.mean().item(), 'std', all_disp_data.std().item())


