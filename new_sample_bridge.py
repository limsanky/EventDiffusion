"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import time 
import numpy as np
import torch
import torchvision.utils as vutils
import torch.distributed as dist
import datetime

from torchinfo import summary
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from models import SSLEventModel
from mvsec_dataset import MVSECDataset, MVSECSampler, SingleMVSECSampler
from torch.nn.functional import interpolate
from diffusion import dist_util, logger
from diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
# from cbm.new_cbm_karras_diffusion import karras_sample
from diffusion.karras_diffusion import karras_sample

from pathlib import Path
from collections import OrderedDict
from utils.metric import AverageMeter, MeanDepthError, MeanDisparityError, NPixelAccuracy
from utils.visualizer import magma_save

def main():
    args = create_argparser().parse_args()
    max_disp = float(args.max_disp)
    # args.use_fp16 

    workdir = os.path.join("/root/code/EventDiffusion/workdir", os.path.basename(args.model_path)[:-3])
    # print('noise_schedule:', args.noise_schedule)
    # exit()
    # print('args.steps:', args.steps)
    # exit()

    ## assume ema ckpt format: ema_{rate}_{steps}.pt
    split = args.model_path.replace("_adapted", "").split("_")
    step = int(split[-1].split(".")[0])
    if args.sampler == "dbim":
        sample_dir = Path(workdir) / f"sample_{step}/split={args.eval_split}/dbim_eta={args.eta}/steps={args.steps}"
    elif args.sampler == "dbim_high_order":
        sample_dir = Path(workdir) / f"sample_{step}/split={args.eval_split}/dbim_order={args.order}/steps={args.steps}"
    elif args.sampler == "dbim_karras":
        sample_dir = Path(workdir) / f"sample_{step}/split={args.eval_split}/dbim_karras_eta={args.eta}/steps={args.steps}"
    elif args.sampler == "dbim_high_order_karras":
        sample_dir = Path(workdir) / f"sample_{step}/split={args.eval_split}/dbim_karras_order={args.order}/steps={args.steps}"
    elif args.sampler == "exp_euler":
        assert int(args.order) in [1, 2]
        sample_dir = Path(workdir) / f"sample_{step}/split={args.eval_split}/exp_euler_order={args.order}/steps={args.steps}"
    else:
        sample_dir = Path(workdir) / f"sample_{step}/split={args.eval_split}/{args.sampler}/steps={args.steps}"
    dist_util.setup_dist()
    
    images_folder = sample_dir / "images"
    magma_folder = sample_dir / "magma"
    
    if dist.get_rank() == 0:
        sample_dir.mkdir(parents=True, exist_ok=True)
        images_folder.mkdir(parents=True, exist_ok=True) 
        magma_folder.mkdir(parents=True, exist_ok=True)
    logger.configure(dir=str(sample_dir))

    logger.log("Current Time: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.log("creating model and diffusion...")
    
    batch_size = args.batch_size
    original_img_size = [ int(item) for item in args.original_img_size.split(',') ]
    assert len(original_img_size) == 2, "Please provide original image size as H,W"
    original_img_size = (original_img_size[0], original_img_size[1])
    resize_size = int(args.resize_size)
    
    event_latent_stds = [ float(item) for item in args.event_latent_stds.split(',') ]
    event_latent_means = [ float(item) for item in args.event_latent_means.split(',') ]
    min_event_latent_vals = [ float(item) for item in args.min_event_latent_vals.split(',') ]
    max_event_latent_vals = [ float(item) for item in args.max_event_latent_vals.split(',') ]
    
    model, diffusion = create_model_and_diffusion(dataset=args.dataset, sigma_data_end=event_latent_stds, **args_to_dict(args, model_and_diffusion_defaults().keys()))
    event_encoder = SSLEventModel(**args_to_dict(args, ['n_channels', 'out_depth', 'bilinear', 'n_lyr', 'ch1', 'c_is_const', 'c_is_scalar']))
    
    logger.log(f"loading model from checkpoint at {args.model_path}...")
    model_dict = torch.load(args.model_path)    
    model.load_state_dict(model_dict, strict=True)
    
    assert args.event_encoder_path is not None, "Please provide a pretrained event encoder path."
    event_encoder_dict = torch.load(args.event_encoder_path)
    with torch.no_grad():
        missed, unexpected = event_encoder.load_state_dict(event_encoder_dict, strict=True)
        for param in event_encoder.parameters():
            param.requires_grad = False
    
    model.to(dist_util.dev())
    event_encoder.to(dist_util.dev())
    dtype = model.dtype
    
    model_stats = summary(model.unet, input_data=torch.ones((1, 3, resize_size, resize_size)).to(dtype), batch_dim=None, device=dist_util.dev(), verbose=0, 
                              xT=torch.ones((1, 3, resize_size, resize_size)).to(dtype), timestep=torch.Tensor([0.52]).to(dtype), dtypes=[dtype, dtype, dtype])
    event_encoder_stats = summary(event_encoder, (1, 3, 260, 346), f0=torch.ones((1, 3, 260, 346)), batch_dim=None, verbose=False, device=dist_util.dev())
    
    with open(f'{sample_dir}/model_info.txt', mode='w') as f:
        f.write(str(model_stats))
        f.write('\n\n\n\n' + str(event_encoder_stats))
        
    model.eval()
    event_encoder.eval()
    
    with open(f'{sample_dir}/sampling_info.txt', mode='w') as f:
        f.write('-'*20 + '\n')
        for _arg_name, _arg_value in vars(args).items():
            f.write(f'{_arg_name}: {_arg_value}\n')
        f.write('\n' + ('-'*20))
        
    logger.log("sampling...")

    eval_split = args.eval_split
    training_data_split = args.training_data_split
    scenario = args.scenario
    data_dir = args.data_dir
    
    all_images = []
    all_labels = []

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
    
    dataset = MVSECDataset(data_dir=data_dir, scenario=scenario, split=training_data_split, is_training=True,
                           event_transforms=event_transforms, image_transforms=image_transforms, 
                           get_disparity=True, disparity_transforms=disp_transforms)
    test_dataset = MVSECDataset(data_dir=data_dir, scenario=scenario, split=training_data_split, is_training=False,
                           event_transforms=event_transforms, image_transforms=image_transforms, 
                           get_disparity=True, disparity_transforms=disp_transforms)
    
    sampler = MVSECSampler(sampler=SingleMVSECSampler(scenario=scenario, split=training_data_split, is_training=True, shuffle=False), 
                           batch_size=batch_size, drop_last=False)
    test_sampler = MVSECSampler(sampler=SingleMVSECSampler(scenario=scenario, split=training_data_split, is_training=False, shuffle=False), 
                                batch_size=batch_size, drop_last=False)
    
    data = DataLoader(dataset, num_workers=1, pin_memory=True, batch_sampler=sampler)
    test_data = DataLoader(test_dataset, num_workers=1, pin_memory=True, batch_sampler=test_sampler)
    
    if args.eval_split == "train":
        dataloader = data
    elif args.eval_split == "test":
        dataloader = test_data
    else:
        raise NotImplementedError
    
    log_dict = OrderedDict([('MSE', AverageMeter(string_format='%6.3lf')), 
                        ('MDeE', MeanDepthError(average_by='image', string_format='%6.3lf')), 
                        ('MDisE', MeanDisparityError(average_by='image', string_format='%6.3lf')), 
                        ('1PA', NPixelAccuracy(n=1, average_by='image', string_format='%6.3lf'))
                        ])
    
    args.num_samples = len(dataloader.dataset)
    num = 0
    start_time = time.time()
    prev_prediction = None
    showing_images_for = list(range(0, 30))
    for i, (event_data, image_data, depth_data, prev_depth_data, disp_data, prev_disp_data, index_data) in enumerate(dataloader):
        
        # Convert back to original disparity values since we saved as (disp*256).to(uint16) in the create_mvsec_dataset.py script
        disp_data = disp_data.to(dist_util.dev()) / 256.0
        prev_disp_data = prev_disp_data.to(dist_util.dev()) / 256.0
        # save disp data for metrics + get mask
        org_disp_data = disp_data.clone()
        org_prev_disp_data = prev_disp_data.clone()
        disp_mask = org_disp_data > 0
        prev_disp_mask = org_prev_disp_data > 0
        # disp_mask = (org_disp_data > 0) & (org_disp_data < max_disp)
        # Normalize to [0, 1] range for training
        disp_data = disp_data / 255.0
        prev_disp_data = prev_disp_data / 255.0
        
        event_data_left = event_data['left']
        event_data_right = event_data['right']
        
        image_at_t0_left, image_at_t1_left = image_data['left']
        image_at_t0_right, image_at_t1_right = image_data['right']
        
        events = torch.cat([event_data_left, event_data_right], dim=0).to(dist_util.dev())
        t0_images = torch.cat([image_at_t0_left, image_at_t0_right], dim=0).to(dist_util.dev())
        # t1_images = torch.cat([image_at_t1_left, image_at_t1_right], dim=0).to(dist_util.dev())

        with torch.no_grad():
            event_latents, t1_pred = event_encoder(events, t0_images)
        
        event_latents = [ (item - min_event_latent_vals[i]) / (max_event_latent_vals[i] - min_event_latent_vals[i]) for i, item in enumerate(event_latents) ]
        event_latents = [ ((item * 2) - 1) for item in event_latents ]
        cond = torch.cat(event_latents, dim=1).to(dist_util.dev())

        x0 = (disp_data * 2) - 1
        y0 = cond
        prev_x0 = (prev_disp_data * 2) - 1
        
        x0 = interpolate(x0, mode='nearest', size=(resize_size, resize_size))
        y0 = interpolate(y0, mode='nearest', size=(resize_size, resize_size))
        prev_x0 = interpolate(prev_x0, mode='nearest', size=(resize_size, resize_size))
        
        x0 = x0.clamp(-1, 1)
        y0 = y0.clamp(-1, 1)
        prev_x0 = prev_x0.clamp(-1, 1)
        
        x0 = x0.repeat(2, 3, 1, 1)
        prev_x0 = prev_x0.repeat(2, 3, 1, 1)
        
        model_kwargs = {"xT": y0}

        if "inpaint" in args.dataset:
            _, mask, label = data[2]
            mask = mask.to(dist_util.dev())
            label = label.to(dist_util.dev())
            model_kwargs["y"] = label
        else:
            mask = None
        
        if args.seed == -1: # added on 11th aug
            seed = None 
        else: 
            seed = data[2][0].numpy() + args.seed
        
        seed = None
        # indexes = torch.tensor(index_data.numpy()).to(dist_util.dev())
        indexes = None
        assert x0 is not None
        with torch.autocast(device_type='cuda', dtype=dtype):
            sample, path, nfe, pred_x0, sigmas, _ = karras_sample(
                diffusion,
                model,
                y0,
                x0,
                prev_x_0=prev_x0,
                prev_x_0_prediction=prev_prediction,
                steps=args.steps,
                mask=mask,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                clip_denoised=args.clip_denoised,
                sampler=args.sampler,
                churn_step_ratio=args.churn_step_ratio,
                eta=args.eta,
                order=args.order,
                seed=seed,
                dbm3_ratio_first=None,
                dbm3_ratio_second=None,
            )
            prev_prediction = sample.clone()

        # print('x0', x0.min().item(), x0.max().item())
        # print('sample', sample.min().item(), sample.max().item())
        # exit()
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        
        if indexes is not None:
            gathered_index = [torch.zeros_like(indexes) for _ in range(dist.get_world_size())]
            # print("huh")
            dist.all_gather(gathered_index, indexes)
            
            gathered_samples = torch.cat(gathered_samples)
            # print("huh2")
            gathered_index = torch.cat(gathered_index)
            # print('huh3')
            try:
                gathered_samples = gathered_samples[torch.argsort(gathered_index)]
            except Exception as e:
                print('nooooooo')
                print(e)
                print(gathered_samples.shape)
                print(gathered_index.shape)
                print(gathered_samples)
                print(gathered_index)
                raise RuntimeError('error...')
            # print('huh4')
        else:
            # print('done')
            gathered_samples = torch.cat(gathered_samples)
                
        if "inpaint" in args.dataset:
            gathered_labels = [torch.zeros_like(label) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_labels, label)
            gathered_labels = torch.cat(gathered_labels)
            if indexes is not None:
                gathered_labels = gathered_labels[torch.argsort(gathered_index)]
        num += gathered_samples.shape[0]

        # num_display = min(4, sample.shape[0])
        num_display = min(1, sample.shape[0])
        # num_display = min(64, sample.shape[0])
        # print('sample shape', sample.shape, 'x0 shape', x0.shape)
        # exit()
        
        # num_rep_images = 3
        num_rep_images = 1
        # if i in showing_images_for and dist.get_rank() == 0:
        #     for image_idx in range(num_display):
                
        #         if x0 is not None:
        #             vutils.save_image(
        #                 x0[image_idx:image_idx+1, 0].unsqueeze(1) / 2 + 0.5,
        #                 f"{images_folder}/x_{image_idx}_batch{i}.png", nrow=1,
        #             )
        #             vutils.save_image(
        #                 prev_x0[image_idx:image_idx+1, 0].unsqueeze(1) / 2 + 0.5,
        #                 f"{images_folder}/prev_x_{image_idx}_batch{i}.png", nrow=1,
        #             )
        #             # magma_save(magma_folder, f"x_{image_idx}.png", 
        #             #            (max_disp * (x0[image_idx:image_idx+1, 0].unsqueeze(1) / 2 + 0.5)), 
        #             #            original_img_size[0], original_img_size[1])
                
        #         for j in range(num_rep_images):
        #             # print('rep idx', j)
        #             # print(sample.permute(0, 3, 1, 2)[image_idx:image_idx+1, j].unsqueeze(1).shape, 
        #             #       sample.permute(0, 3, 1, 2)[image_idx:image_idx+1, j].unsqueeze(1).min().item(), 
        #             #       sample.permute(0, 3, 1, 2)[image_idx:image_idx+1, j].unsqueeze(1).max().item())
        #             # print(x0[image_idx:image_idx+1, 0].unsqueeze(1).shape)
        #             # exit()
        #             vutils.save_image(
        #                 sample.permute(0, 3, 1, 2)[image_idx:image_idx + 1, j].unsqueeze(1).float() / 255,
        #                 f"{images_folder}/sample_{image_idx}_batch{i}.png", nrow=1)
                    
        #             # magma_save(magma_folder, f"sample_{image_idx}.png", 
        #             #            (max_disp * sample.permute(0, 3, 1, 2)[image_idx:image_idx + 1, j].unsqueeze(1).float() / 255), 
        #             #            original_img_size[0], original_img_size[1])
                    
        #             # continue
        #             vutils.save_image(
        #                 y0[image_idx:image_idx+1, j].unsqueeze(1) / 2 + 0.5,
        #                 f"{images_folder}/y_{image_idx}_batch{i}.png", nrow=1)
                    
        #             for intermediate_x0_idx in range(len(pred_x0)):
        #                 intermediate_x0 = ((pred_x0[intermediate_x0_idx][image_idx: image_idx + 1, j].unsqueeze(1) + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        #                 intermediate_x0 = intermediate_x0.permute(0, 2, 3, 1).contiguous()
        #                 vutils.save_image(
        #                     intermediate_x0.permute(0, 3, 1, 2).float() / 255,
        #                     f"{images_folder}/intermediate_x0_at_time_idx{intermediate_x0_idx}_batch{i}.png", nrow=1)
        #                 # magma_save(magma_folder, f"intermediate_x0_at_time_idx{intermediate_x0_idx}.png", 
        #                 #        (max_disp * intermediate_x0.permute(0, 3, 1, 2).float() / 255).clamp(0, 255), 
        #                 #        original_img_size[0], original_img_size[1])
                        
        if i in showing_images_for and dist.get_rank() == 0:
            if x0 is not None:
                vutils.save_image(
                    x0[:num_display, 0].unsqueeze(1) / 2 + 0.5,
                    f"{images_folder}/full_x_batch{i}.png",
                    nrow=int(np.sqrt(num_display)),
                )
                vutils.save_image(
                    prev_x0[:num_display, 0].unsqueeze(1) / 2 + 0.5,
                    f"{images_folder}/full_prev_x_batch{i}.png",
                    nrow=int(np.sqrt(num_display)),
                )
                # magma_save(magma_folder, f"full_x_{i}.png", 
                #     (max_disp * (x0[:num_display, 0].unsqueeze(1) / 2 + 0.5)).clamp(0, 255), 
                #     original_img_size[0], original_img_size[1])
            for j in range(num_rep_images):
                vutils.save_image(
                    sample.permute(0, 3, 1, 2)[:num_display, j].unsqueeze(1).float() / 255,
                    f"{images_folder}/full_sample_batch{i}_idx{j}.png",
                    nrow=int(np.sqrt(num_display)),
                )
                # magma_save(magma_folder, f"full_sample_{i}_idx{j}.png", 
                #     (max_disp * sample.permute(0, 3, 1, 2)[:num_display, j].unsqueeze(1).float() / 255).clamp(0, 255), 
                #     original_img_size[0], original_img_size[1])
                
                vutils.save_image(
                    y0[:num_display, j].unsqueeze(1) / 2 + 0.5,
                    f"{images_folder}/full_y_batch{i}_idx{j}.png",
                    nrow=int(np.sqrt(num_display)),
                )
                
                for intermediate_x0_idx in range(len(pred_x0)):
                    intermediate_x0 = ((pred_x0[intermediate_x0_idx][:num_display, j].unsqueeze(1) + 1) * 127.5).clamp(0, 255).to(torch.uint8)
                    intermediate_x0 = intermediate_x0.permute(0, 2, 3, 1).contiguous()
                    vutils.save_image(
                        intermediate_x0.permute(0, 3, 1, 2).float() / 255,
                        f"{images_folder}/full_intermediate_x0_at_time_idx{intermediate_x0_idx}_batch{i}_RepIdx{j}.png", 
                        nrow=int(np.sqrt(num_display)),
                    )
                    # magma_save(magma_folder, f"full_intermediate_x0_at_time_idx{intermediate_x0_idx}_RepIdx{j}.png", 
                    #     (max_disp * intermediate_x0.permute(0, 3, 1, 2).float() / 255).clamp(0, 255), 
                    #     original_img_size[0], original_img_size[1])
        
        # if i in showing_images_for and dist.get_rank() == 0:
        #     vutils.save_image(
        #         sample.permute(0, 3, 1, 2)[:num_display].float() / 255,
        #         f"{sample_dir}/sample_{i}.png",
        #         nrow=int(np.sqrt(num_display)),
        #     )
        #     if x0 is not None:
        #         vutils.save_image(
        #             x0_image[:num_display] / 2 + 0.5,
        #             f"{sample_dir}/x_{i}.png",
        #             nrow=int(np.sqrt(num_display)),
        #         )
        #     vutils.save_image(
        #         cond_image[:num_display] / 2 + 0.5,
        #         f"{sample_dir}/y_{i}.png",
        #         nrow=int(np.sqrt(num_display)),
        #     )
        all_images.append(gathered_samples.to(torch.uint8).detach().cpu().numpy())
        if "inpaint" in args.dataset:
            all_labels.append(gathered_labels.detach().cpu().numpy())

        if dist.get_rank() == 0:
            logger.log(f"sampled {num} images")
        
        # Update metrics
        half_batch = sample.shape[0] // 2
        
        img_idx_for_metric_updates = 0
        pred = interpolate((sample / 255.).permute(0, 3, 1, 2).float()[:half_batch, img_idx_for_metric_updates].unsqueeze(1), size=original_img_size, mode='nearest')
        pred = (pred * 255).clamp(0, 255)
        
        for img_idx in range(pred.shape[0]):
            magma_save(magma_folder, f"sample_{img_idx}_idx{img_idx_for_metric_updates}.png", 
                100 * pred[img_idx].view(original_img_size).float() / 255, 
                original_img_size[0], original_img_size[1])
        
        pred = pred / 255.
        org_disp_data = org_disp_data / 255.
        org_prev_disp_data = org_prev_disp_data / 255.
        pred = pred * max_disp
        org_disp_data = org_disp_data * max_disp
        org_prev_disp_data = org_prev_disp_data * max_disp
        
        # disp_mask = torch.logical_and(disp_mask, org_disp_data < max_disp)
        # print(disp_mask.shape, pred.shape, org_disp_data.shape)
        # exit()
        
        mse_loss = torch.square(pred[disp_mask] - org_disp_data[disp_mask]).mean()
        # print('mse_loss:', mse_loss.item())
        # print('pred min/max:', pred.min().item(), pred.max().item())
        # print('pred[disp_mask]:', pred[disp_mask].min().item(), pred[disp_mask].max().item())
        # print('org_disp_data min/max:', org_disp_data.min().item(), org_disp_data.max().item())
        # print('org_disp_data[disp_mask]:', org_disp_data[disp_mask].min().item(), org_disp_data[disp_mask].max().item())
        # exit()
        log_dict['MSE'].update(mse_loss.item(), pred.size(0))
        log_dict['MDeE'].update(pred, org_disp_data, disp_mask)
        log_dict['MDisE'].update(pred, org_disp_data, disp_mask)
        log_dict['1PA'].update(pred, org_disp_data, disp_mask)
    
    end_time = time.time()
    
    gen_rate = args.num_samples / (end_time - start_time)
    
    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if "inpaint" in args.dataset:
        labels = np.concatenate(all_labels, axis=0)
        labels = labels[: args.num_samples]

    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(sample_dir, f"samples_{shape_str}_nfe{nfe}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)
        if "inpaint" in args.dataset:
            shape_str = "x".join([str(x) for x in labels.shape])
            out_path = os.path.join(sample_dir, f"labels_{shape_str}_nfe{nfe}.npz")
            logger.log(f"saving to {out_path}")
            np.savez(out_path, labels)

    dist.barrier()
    logger.log("sampling complete")
    total_time = (args.num_samples / gen_rate) / 60
    
    logger.log("Sampler:", args.sampler)
    logger.log("Rate:", gen_rate, "img/s.")
    logger.log(f"Created {args.num_samples} samples in {total_time:.4f} mins.") 
        
    with open(f'{sample_dir}/test_log.txt', mode='w') as f:
        # Print test log
        test_summary = (f"Test | time for test: {total_time} min\n"
                        f"Test | Loss: {log_dict['MSE']} | MDeE: {log_dict['MDeE']} | MDisE: {log_dict['MDisE']} | 1PA: {log_dict['1PA']}")
        logger.log(test_summary + '\n')


def create_argparser():
    defaults = dict(
        event_encoder_path=None,
        data_dir="",  ## only used in bridge
        dataset="e2d",
        eval_split='train',
        training_data_split=1,
        scenario="indoor_flying",
        max_disp=37,
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        churn_step_ratio=0.0,
        rho=7.0,
        steps=40,
        model_path="",
        exp="",
        seed=42,
        num_workers=8,
        eta=1.0,
        order=1,
        dbm3_ratio_first=None,
        dbm3_ratio_second=None,
        n_channels=3,
        out_depth=1,
        bilinear=True,
        n_lyr=4,
        ch1=24,
        c_is_const=False,
        c_is_scalar=False,
        event_latent_stds=None,
        event_latent_means=None,
        max_event_latent_vals=None,
        min_event_latent_vals=None,
        mu_data=None,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
