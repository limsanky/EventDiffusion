"""
Train a diffusion model on images.
"""

import argparse

from diffusion import dist_util, logger
from diffusion.resample import create_named_schedule_sampler
from diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    sample_defaults,
    args_to_dict,
    add_dict_to_argparser,
    get_workdir,
)
from diffusion.train_util import DBMTrainLoop
from torchinfo import summary
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from models import SSLEventModel
from mvsec_dataset import MVSECDataset, MVSECSampler, SingleMVSECSampler

import torch.distributed as dist

from pathlib import Path

import wandb

from glob import glob
import os
import torch

def main(args):

    workdir = get_workdir(args.exp, args.date)
    Path(workdir).mkdir(parents=True, exist_ok=True)
    (Path(workdir) / 'images').mkdir(parents=True, exist_ok=True)

    dist_util.setup_dist()
    logger.configure(dir=workdir)
    if dist.get_rank() == 0:
        name = args.exp if args.resume_checkpoint == "" else args.exp + "_resume"
        wandb.init(
            project="bridge",
            group=args.exp,
            name=name,
            config=vars(args),
            mode="offline" if not args.debug else "disabled",
        )
        logger.log("creating model and diffusion...")

    original_img_size = [ int(item) for item in args.original_img_size.split(',') ]
    assert len(original_img_size) == 2, "Please provide original image size as H,W"
    original_img_size = (original_img_size[0], original_img_size[1])
    resize_size = int(args.resize_size)
    padding_to_size = None if args.padding_to_size is None else int(args.padding_to_size)
    img_size = padding_to_size if resize_size == 0 else resize_size
    
    event_latent_stds = [ float(item) for item in args.event_latent_stds.split(',') ]
    event_latent_means = [ float(item) for item in args.event_latent_means.split(',') ]
    min_event_latent_vals = [ float(item) for item in args.min_event_latent_vals.split(',') ]
    max_event_latent_vals = [ float(item) for item in args.max_event_latent_vals.split(',') ]

    # Load target model
    resume_train_flag = False
    if args.resume_checkpoint == "":
        model_ckpts = list(glob(f"{workdir}/*model*[0-9].*"))
        if len(model_ckpts) > 0:
            max_ckpt = max(model_ckpts, key=lambda x: int(x.split("model_")[-1].split(".")[0]))
            if os.path.exists(max_ckpt):
                args.resume_checkpoint = max_ckpt
                resume_train_flag = True
        elif args.pretrained_ckpt is not None:
            max_ckpt = args.pretrained_ckpt
            args.resume_checkpoint = max_ckpt
        if dist.get_rank() == 0 and args.resume_checkpoint != "":
            logger.log("Resuming from checkpoint: ", max_ckpt)

    model, diffusion = create_model_and_diffusion(dataset=args.dataset, sigma_data_end=event_latent_stds, **args_to_dict(args, model_and_diffusion_defaults().keys()))
    event_encoder = SSLEventModel(**args_to_dict(args, ['n_channels', 'out_depth', 'bilinear', 'n_lyr', 'ch1', 'c_is_const', 'c_is_scalar']))
    
    assert args.event_encoder_path is not None, "Please provide a pretrained event encoder path."
    event_encoder_dict = torch.load(args.event_encoder_path)
    with torch.no_grad():
        missed, unexpected = event_encoder.load_state_dict(event_encoder_dict, strict=True)
        for param in event_encoder.parameters():
            param.requires_grad = False
    
    model.to(dist_util.dev())
    event_encoder.to(dist_util.dev())
    dtype = model.dtype
    
    model_stats = summary(model.unet, input_data=torch.ones((1, 3, img_size, img_size)).to(dtype), batch_dim=None, device=dist_util.dev(), verbose=0, 
                              xT=torch.ones((1, 3, img_size, img_size)).to(dtype), timestep=torch.Tensor([0.52]).to(dtype), dtypes=[dtype, dtype, dtype])
    event_encoder_stats = summary(event_encoder, (1, 3, 260, 346), f0=torch.ones((1, 3, 260, 346)), batch_dim=None, verbose=False, device=dist_util.dev())
    
    with open(f'{workdir}/model_info.txt', mode='w') as f:
        f.write(str(model_stats))
        f.write('\n\n\n\n' + str(event_encoder_stats))
        
    model.train()
    event_encoder.eval()
    
    with open(f'{workdir}/training_info.txt', mode='w') as f:
        f.write('-'*20 + '\n')
        for _arg_name, _arg_value in vars(args).items():
            f.write(f'{_arg_name}: {_arg_value}\n')
        f.write('\n' + ('-'*20))

    # if dist.get_rank() == 0:
    #     wandb.watch(model, log="all")
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
    # print("args.batch_size:", args.batch_size, args.global_batch_size)
    # exit()
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}")
    else:
        batch_size = args.batch_size

    if dist.get_rank() == 0:
        logger.log("creating data loader...")
    
    split = args.split
    scenario = args.scenario
    data_dir = args.data_dir
    
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

    if dist.get_rank() == 0:
        logger.log("training...")
    
    DBMTrainLoop(
        model=model,
        event_encoder=event_encoder,
        diffusion=diffusion,
        train_data=data,
        test_data=test_data,
        resize_size=resize_size,
        original_size=original_img_size,
        # data_image_size=data_image_size,
        batch_size=batch_size,
        microbatch=-1 if args.microbatch >= batch_size else args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        test_interval=args.test_interval,
        save_interval=args.save_interval,
        save_interval_for_preemption=args.save_interval_for_preemption,
        resume_checkpoint=args.resume_checkpoint,
        workdir=workdir,
        use_fp16=args.use_fp16,
        use_bf16=args.use_bf16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        train_mode=args.train_mode,
        resume_train_flag=resume_train_flag,
        total_training_steps=args.total_training_steps,
        event_latent_means=event_latent_means,
        event_latent_stds=event_latent_stds,
        max_event_latent_vals=max_event_latent_vals,
        min_event_latent_vals=min_event_latent_vals,
        mu_data=args.mu_data,
        use_disp_mask=args.use_disp_mask,
        padding_to_size=padding_to_size,
        **sample_defaults(),
    ).run_loop()


def create_argparser():
    defaults = dict(
        event_encoder_path=None,
        data_dir="",
        dataset="e2d",
        split=1,
        scenario="indoor_flying",
        schedule_sampler="real-uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=256,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=50,
        test_interval=500,
        save_interval=10000,
        save_interval_for_preemption=50000,
        resume_checkpoint="",
        exp="",
        use_fp16=True,
        use_bf16=False,
        fp16_scale_growth=1e-3,
        debug=False,
        num_workers=8,
        use_augment=False,
        pretrained_ckpt=None,
        train_mode="ddbm",
        total_training_steps=200000,
        date='0',
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
        use_disp_mask=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
