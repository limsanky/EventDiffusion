import copy
import functools
import os
import numpy as np

import blobfile as bf
import torch
import torch.distributed as dist
from torch.nn.functional import interpolate
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam
import datetime
import torchvision.utils as vutils
from pathlib import Path

from diffusion import dist_util, logger
from diffusion.nn import update_ema

from diffusion.random_util import get_generator
from torchvision.transforms.v2.functional import crop

import glob

import wandb


class DBMTrainLoop:
    def __init__(
        self,
        *,
        model,
        event_encoder,
        diffusion,
        train_data,
        test_data,
        # data_image_size: int,
        resize_size: int,
        original_size,
        batch_size,
        microbatch,
        lr,
        ema_rate, 
        log_interval,
        test_interval,
        save_interval,
        save_interval_for_preemption,
        resume_checkpoint,
        workdir,
        use_fp16=False,
        use_bf16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        total_training_steps=10000000,
        augment_pipe=None,
        train_mode="ddbm",
        resume_train_flag=False,
        event_latent_means=None,
        event_latent_stds=None,
        max_event_latent_vals=None,
        min_event_latent_vals=None,
        # mu_data=None,
        use_disp_mask=False,
        padding_to_size=None,
        **sample_kwargs,
    ):
        self.resize_size = resize_size
        self.original_size = original_size
        self.event_latent_means = event_latent_means
        self.max_event_latent_vals = max_event_latent_vals
        self.min_event_latent_vals = min_event_latent_vals
        # self.mu_data = mu_data
        self.sigma_data_end = event_latent_stds
        self.use_disp_mask = use_disp_mask
        self.padding_to_size = padding_to_size
        if padding_to_size:
            assert resize_size == 0, "Cannot use both resize and padding!"
        self.model = model
        self.diffusion = diffusion
        self.data = train_data
        self.test_data = test_data
        self.image_size = model.image_size
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = [ema_rate] if isinstance(ema_rate, float) else [float(x) for x in ema_rate.split(",")]
        self.log_interval = log_interval
        self.workdir = workdir
        self.img_dir = Path(workdir) / "images"
        self.test_interval = test_interval
        self.save_interval = save_interval
        self.save_interval_for_preemption = save_interval_for_preemption
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        assert not (self.use_fp16 and self.use_bf16), "Cannot use both fp16 and bf16 for training; pick one!"
        if self.use_bf16:
            assert not self.use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.total_training_steps = total_training_steps
        self.current_num_steps = 0

        # self.data_image_size = data_image_size
        self.train_mode = train_mode

        self.step = 0
        self.resume_train_flag = resume_train_flag
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        # self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_fp16)
        self.scaler = torch.amp.GradScaler(enabled=self.use_fp16)

        self._load_and_sync_parameters()
        if not self.resume_train_flag:
            self.resume_step = 0

        opt_betas = (0.9, 0.999) # original
        self.opt = RAdam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                         betas=opt_betas)
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [self._load_ema_parameters(rate) for rate in self.ema_rate]
            self._load_scaler_state()
        else:
            self.ema_params = [copy.deepcopy(list(self.model.parameters())) for _ in range(len(self.ema_rate))]

        if torch.cuda.is_available():
            self.use_ddp = True
            local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn("Distributed training requires CUDA. " "Gradients will not be synchronized properly!")
            self.use_ddp = False
            self.ddp_model = self.model
        
        self.event_encoder = event_encoder
        self.step = self.resume_step

        self.generator = get_generator(sample_kwargs["generator"], self.batch_size, 42)
        self.sample_kwargs = sample_kwargs

        self.augment = augment_pipe
        
        if dist.get_rank() == 0:
            logger.log(datetime.datetime.now().strftime("Experiment Start Time: %Y-%m-%d %H-%M-%S"))

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            if self.resume_train_flag:
                self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                logger.log("Resume step: ", self.resume_step)

            self.model.load_state_dict(torch.load(resume_checkpoint, map_location="cpu"))
            self.model.to(dist_util.dev())

            dist.barrier()
    
    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(list(self.model.parameters()))

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = torch.load(ema_checkpoint, map_location=dist_util.dev())
            ema_params = [state_dict[name] for name, _ in self.model.named_parameters()]

            dist.barrier()
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if main_checkpoint.split("/")[-1].startswith("freq"):
            prefix = "freq_"
        else:
            prefix = ""
        opt_checkpoint = bf.join(bf.dirname(main_checkpoint), f"{prefix}opt_{self.resume_step:06}.pt")
        if bf.exists(opt_checkpoint):
            if dist.get_rank() == 0:
                logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = torch.load(opt_checkpoint, map_location=dist_util.dev())
            self.opt.load_state_dict(state_dict)
            
            dist.barrier()

    def _load_scaler_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if main_checkpoint.split("/")[-1].startswith("freq"):
            prefix = "freq_"
        else:
            prefix = ""
        scaler_checkpoint = bf.join(bf.dirname(main_checkpoint), f"{prefix}scaler_{self.resume_step:06}.pt")
        if bf.exists(scaler_checkpoint):
            if dist.get_rank() == 0:
                logger.log(f"loading scaler state from checkpoint: {scaler_checkpoint}")
            state_dict = torch.load(scaler_checkpoint, map_location=dist_util.dev())
            self.scaler.load_state_dict(state_dict)
            
            dist.barrier()
            
    def run_loop(self):
        while True:
            for (event_data, image_data, depth_data, prev_depth_data, disp_data, prev_disp_data, index_data) in self.data:

                # left_index, right_index = index_data['left'], index_data['right']
                # # assert (left_index[0] == right_index[0]).all() and (left_index[1] == right_index[1]).all(), f'Left and Right indices do not match: {left_index}, {right_index}'
                
                # disp_data = disp_data
                # Convert back to original disparity values since we saved as (disp*256).to(uint16) in the create_mvsec_dataset.py script
                disp_data = disp_data / 256.0
                if self.use_disp_mask:
                    # get mask
                    disp_mask = (disp_data > 0).to(dist_util.dev()).to(torch.bool)
                # Normalize to [0, 1] range for training
                disp_data = disp_data / 255.0
                
                event_data_left = event_data['left']
                event_data_right = event_data['right']
                
                image_at_t0_left, image_at_t1_left = image_data['left']
                image_at_t0_right, image_at_t1_right = image_data['right']
                
                events = torch.cat([event_data_left, event_data_right], dim=0).to(dist_util.dev())
                t0_images = torch.cat([image_at_t0_left, image_at_t0_right], dim=0).to(dist_util.dev())
                # t1_images = torch.cat([image_at_t1_left, image_at_t1_right], dim=0).to(dist_util.dev())

                with torch.no_grad():
                    event_latents, t1_pred = self.event_encoder(events, t0_images)
                
                event_latents = [ (item - self.min_event_latent_vals[i]) / (self.max_event_latent_vals[i] - self.min_event_latent_vals[i]) for i, item in enumerate(event_latents) ]
                
                event_latents = [ ((item * 2) - 1) for item in event_latents ]
                
                cond = torch.cat(event_latents, dim=1)

                # Normalize to [-1, 1] range
                batch = (disp_data * 2) - 1
                
                # 27th Nov: used to be done before interpolation, but this causes a slight difference in values after resizing
                # cond = cond.clamp(-1, 1)
                # batch: torch.Tensor = batch.clamp(-1, 1)
                
                padding_mask = None
                if self.padding_to_size is None:
                    batch = interpolate(batch, mode='nearest',
                                        size=(self.resize_size, self.resize_size))
                    cond = interpolate(cond, mode='nearest',
                                        size=(self.resize_size, self.resize_size))
                    if self.use_disp_mask:
                        disp_mask = interpolate(disp_mask.float(), mode='nearest',
                                            size=(self.resize_size, self.resize_size))
                        disp_mask[disp_mask >= 0.5] = 1.0
                        disp_mask[disp_mask < 0.5] = 0.0
                    else:
                        disp_mask = None
                else:
                    padding_mask = torch.ones_like(batch)
                    pad_l, pad_r = (self.padding_to_size - self.original_size[1]) // 2, (self.padding_to_size - self.original_size[1]) - ((self.padding_to_size - self.original_size[1]) // 2)
                    # pad_l, pad_r = 1, 1
                    pad_t, pad_b = (self.padding_to_size - self.original_size[0]) // 2, (self.padding_to_size - self.original_size[0]) - ((self.padding_to_size - self.original_size[0]) // 2)
                    # print('Original size:', batch.shape, self.original_size)
                    batch = torch.nn.functional.pad(batch, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
                    # print('Padded size:', batch.shape)
                    # exit()
                    cond = torch.nn.functional.pad(cond, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
                    padding_mask = torch.nn.functional.pad(padding_mask, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
                    if self.use_disp_mask:
                        disp_mask = torch.nn.functional.pad(disp_mask.float(), (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
                # 27th Nov: moved to after interpolation to avoid slight value changes due to resizing
                cond = cond.clamp(-1, 1)
                batch: torch.Tensor = batch.clamp(-1, 1)
                
                batch = batch.repeat(2, 3, 1, 1)
                if padding_mask is not None:
                    padding_mask = padding_mask.repeat(2, 3, 1, 1)
                
                if "inpaint" in self.workdir:
                    _, mask, label = _
                else:
                    mask = None
                
                if not (not self.lr_anneal_steps or self.step < self.total_training_steps):
                    # Save the last checkpoint if it wasn't already saved.
                    if (self.step - 1) % self.save_interval != 0:
                        self.save()
                    return

                # if self.augment is not None:
                #     batch, _ = self.augment(batch)
                if isinstance(cond, torch.Tensor) and batch.ndim == cond.ndim:
                    cond = {"xT": cond}
                else:
                    cond["xT"] = cond["xT"]
                if mask is not None:
                    cond["mask"] = mask
                    cond["y"] = label
                
                cond["disp_mask"] = disp_mask
                cond["padding_mask"] = padding_mask

                took_step = self.run_step(batch, cond)
                if took_step and self.step % self.log_interval == 0:
                    logs = logger.dumpkvs()
                    if dist.get_rank() == 0:
                        logger.log(datetime.datetime.now().strftime("Time: %Y-%m-%d %H-%M-%S"))
                        wandb.log(logs, step=self.step)

                if took_step and self.step % self.save_interval == 0:
                    self.save()
                    # Run for a finite amount of time in integration tests.
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                        return

                    # test_batch, test_cond, _ = next(iter(self.test_data))
                    (test_event_data, test_image_data, test_depth_data, test_prev_depth_data, test_disp_data, test_prev_disp_data, test_index_data) = next(iter(self.test_data))
                        
                    test_disp_data = test_disp_data / 256.0
                    if self.use_disp_mask:
                        test_disp_mask = (test_disp_data > 0).to(dist_util.dev()).to(torch.bool)
                    test_disp_data = test_disp_data / 255.0
                    
                    test_event_data_left = test_event_data['left']
                    test_event_data_right = test_event_data['right']
                    test_image_at_t0_left, test_image_at_t1_left = test_image_data['left']
                    test_image_at_t0_right, test_image_at_t1_right = test_image_data['right']
                    
                    test_events = torch.cat([test_event_data_left, test_event_data_right], dim=0).to(dist_util.dev())
                    test_t0_images = torch.cat([test_image_at_t0_left, test_image_at_t0_right], dim=0).to(dist_util.dev())
                    
                    with torch.no_grad():
                        test_event_latents, t1_pred = self.event_encoder(test_events, test_t0_images)
                    
                    test_event_latents = [ (item - self.min_event_latent_vals[i]) / (self.max_event_latent_vals[i] - self.min_event_latent_vals[i]) for i, item in enumerate(test_event_latents) ]
                    
                    test_event_latents = [ ((item * 2) - 1) for item in test_event_latents ]
                    
                    test_cond = torch.cat(test_event_latents, dim=1)

                    # Normalize to [-1, 1] range
                    test_batch = (test_disp_data * 2) - 1
                    
                    test_padding_mask = None
                    if self.padding_to_size is None:
                        test_batch = interpolate(test_batch, mode='nearest',
                                            size=(self.resize_size, self.resize_size))
                        test_cond = interpolate(test_cond, mode='nearest',
                                            size=(self.resize_size, self.resize_size))
                        if self.use_disp_mask:
                            test_disp_mask = interpolate(test_disp_mask.float(), mode='nearest',
                                            size=(self.resize_size, self.resize_size))
                            test_disp_mask[test_disp_mask >= 0.5] = 1.0
                            test_disp_mask[test_disp_mask < 0.5] = 0.0
                        else:
                            test_disp_mask = None
                    else:
                        test_padding_mask = torch.ones_like(test_batch)
                        pad_l, pad_r = (self.padding_to_size - self.original_size[1]) // 2, (self.padding_to_size - self.original_size[1]) - ((self.padding_to_size - self.original_size[1]) // 2)
                        pad_t, pad_b = (self.padding_to_size - self.original_size[0]) // 2, (self.padding_to_size - self.original_size[0]) - ((self.padding_to_size - self.original_size[0]) // 2)
                        test_batch = torch.nn.functional.pad(test_batch, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
                        test_cond = torch.nn.functional.pad(test_cond, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
                        test_padding_mask = torch.nn.functional.pad(test_padding_mask, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
                        if self.use_disp_mask:
                            test_disp_mask = torch.nn.functional.pad(test_disp_mask.float(), (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
                    
                    test_cond = test_cond.clamp(-1, 1)
                    test_batch: torch.Tensor = test_batch.clamp(-1, 1)
                    
                    test_batch = test_batch.repeat(2, 3, 1, 1)
                    if test_padding_mask is not None:
                        test_padding_mask = test_padding_mask.repeat(2, 3, 1, 1)
                    
                    if "inpaint" in self.workdir:
                        _, mask, label = _
                    else:
                        mask = None
                        
                    if isinstance(test_cond, torch.Tensor) and test_batch.ndim == test_cond.ndim:
                        test_cond = {"xT": test_cond}
                    else:
                        test_cond["xT"] = test_cond["xT"]
                    
                    test_cond["disp_mask"] = test_disp_mask
                    test_cond["padding_mask"] = test_padding_mask
                    
                    if mask is not None:
                        test_cond["mask"] = mask
                        test_cond["y"] = label
                    self.run_test_step(test_batch, test_cond)
                    logs = logger.dumpkvs()

                    if dist.get_rank() == 0:
                        wandb.log(logs, step=self.step)

                if took_step and self.step % self.save_interval_for_preemption == 0:
                    self.save(for_preemption=True)

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        logger.logkv_mean("lg_loss_scale", np.log2(self.scaler.get_scale()))
        self.scaler.unscale_(self.opt)

        def _compute_norms():
            grad_norm = 0.0
            param_norm = 0.0
            for p in self.model.parameters():
                with torch.no_grad():
                    param_norm += torch.norm(p, p=2, dtype=torch.float32).item() ** 2
                    if p.grad is not None:
                        grad_norm += torch.norm(p.grad, p=2, dtype=torch.float32).item() ** 2
            return np.sqrt(grad_norm), np.sqrt(param_norm)

        grad_norm, param_norm = _compute_norms()

        logger.logkv_mean("grad_norm", grad_norm)
        logger.logkv_mean("param_norm", param_norm)

        self.scaler.step(self.opt)
        self.scaler.update()
        self.step += 1
        self._update_ema()
        
        self._anneal_lr()
        self.log_step()
        return True

    def run_test_step(self, batch, cond):
        with torch.no_grad():
            self.forward_backward(batch, cond, train=False)

    def forward_backward(self, batch, cond, train=True):
        if train:
            self.opt.zero_grad()
        
        # Commented out on 4th Sep:
        assert batch.shape[0] % self.microbatch == 0
        num_microbatches = batch.shape[0] // self.microbatch
        # # Added on 4th Sep:
        # if batch.shape[0] % self.microbatch == 0:
        #     num_microbatches = batch.shape[0] // self.microbatch
        # else:
        #     num_microbatches = 1
        
        _dtype = torch.float16 if self.use_fp16 else (torch.bfloat16 if self.use_bf16 else torch.float32)
        assert _dtype == torch.bfloat16
        for i in range(0, batch.shape[0], self.microbatch):
            
            # with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
            with torch.autocast(device_type="cuda", dtype=_dtype, enabled=self.use_fp16 or self.use_bf16):
                micro = batch[i : i + self.microbatch].to(dist_util.dev())
                micro_cond = {k: v[i : i + self.microbatch].to(dist_util.dev()) for k, v in cond.items()}
                last_batch = (i + self.microbatch) >= batch.shape[0]

                t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
                t, weights = t.to(micro.dtype), weights.to(micro.dtype)
                
                return_image_samples = True
                eps = 1e-4
                if self.train_mode == "ddbm":
                    compute_losses = functools.partial(
                        self.diffusion.training_bridge_losses,
                        self.ddp_model,
                        micro,
                        t,
                        model_kwargs=micro_cond,
                        return_image_samples=return_image_samples,
                        ema_params=self.ema_params,
                        is_training=train,
                        eps=eps,
                    )
                else:
                    raise NotImplementedError()

                if last_batch or not self.use_ddp:
                    if return_image_samples:
                        losses, test_xts, denoised, test_t, ema_denoised_t, ema_denoised_T_min_eps = compute_losses()
                    else:
                        losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        if return_image_samples:
                            losses, test_xts, denoised, test_t, ema_denoised_t, ema_denoised_T_min_eps = compute_losses()
                        else:
                            losses = compute_losses()

                loss = (losses["loss"] * weights).mean() / num_microbatches
                
            log_loss_dict(self.diffusion, t, {k if train else "test_" + k: v * weights for k, v in losses.items()},
                          curr_step=self.step, log_interval=self.log_interval)
            if train:
                self.scaler.scale(loss).backward()
                world_size = float(dist.get_world_size())
                for name, param in self.ddp_model.named_parameters():
                    if param.grad is None:
                        print(f"WARNING: {name} has no grad!")
                        continue
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size
                dist.barrier()
            else:
                if dist.get_rank() == 0 and return_image_samples and test_t is not None:
                    
                    num_display = min(4, int(denoised.shape[0]))
                    sigma_t_val = f"{test_t.unique().item():.3f}"
                    sigma_T_val = f"{self.diffusion.t_max - eps:.3f}"
                    
                    pad_l, pad_t = (self.padding_to_size - self.original_size[1]) // 2, (self.padding_to_size - self.original_size[0]) // 2
                    
                    micro = crop(micro, top=pad_t, left=pad_l, height=self.original_size[0], width=self.original_size[1])
                    denoised = crop(denoised, top=pad_t, left=pad_l, height=self.original_size[0], width=self.original_size[1])
                    micro_cond['xT'] = crop(micro_cond['xT'], top=pad_t, left=pad_l, height=self.original_size[0], width=self.original_size[1])
                    ema_denoised_t = [ crop(item, top=pad_t, left=pad_l, height=self.original_size[0], width=self.original_size[1]) for item in ema_denoised_t ]
                    ema_denoised_T_min_eps = [ crop(item, top=pad_t, left=pad_l, height=self.original_size[0], width=self.original_size[1]) for item in ema_denoised_T_min_eps ]
                    test_xts = [ crop(item, top=pad_t, left=pad_l, height=self.original_size[0], width=self.original_size[1]) for item in test_xts ]
                    assert micro.shape[2] == self.original_size[0] and micro.shape[3] == self.original_size[1], f'Crop size incorrect: {micro.shape}, expected size: {self.original_size}'
                    
                    vutils.save_image(interpolate(micro[:num_display, 0].unsqueeze(1), size=self.original_size, mode='nearest')/2+0.5, 
                                      f'{self.img_dir}/x_{self.step}_{sigma_t_val}.png',nrow=int(np.sqrt(num_display)))
                    for i in range(3):
                        vutils.save_image(interpolate(denoised[:num_display, i].unsqueeze(1), size=self.original_size, mode='nearest')/2+0.5, 
                                          f'{self.img_dir}/pred_{self.step}_{sigma_t_val}_idx{i}.png', nrow=int(np.sqrt(num_display)))
                        
                        vutils.save_image(interpolate(micro_cond['xT'][:num_display, i].unsqueeze(1), size=self.original_size, mode='nearest')/2+0.5, 
                                          f'{self.img_dir}/y_{self.step}_{sigma_t_val}_idx{i}.png',nrow=int(np.sqrt(num_display)))
                    
                        for rate, ema_x0_t, ema_x0_T, ema_xt in zip(self.ema_rate, ema_denoised_t, ema_denoised_T_min_eps, test_xts):
                            vutils.save_image(interpolate(ema_x0_t[:num_display, i].unsqueeze(1), size=self.original_size, mode='nearest').float()/2+0.5, 
                                              f'{self.img_dir}/ema_{rate}_pred_{self.step}_{sigma_t_val}_idx{i}.png', nrow=int(np.sqrt(num_display)))
                            
                            vutils.save_image(interpolate(ema_x0_T[:num_display, i].unsqueeze(1), size=self.original_size, mode='nearest').float()/2+0.5, 
                                              f'{self.img_dir}/ema_{rate}_{self.step}_{sigma_T_val}_idx{i}.png', nrow=int(np.sqrt(num_display)))
                            
                            save_denoised_images(interpolate(ema_xt[:num_display, i].unsqueeze(1), size=self.original_size, mode='nearest'), 
                                                 f'{self.img_dir}/ema_{rate}_xt_{self.step}_{sigma_t_val}_idx{i}.png', num_display)

                    print('Images saved!!\n')
                    # exit()

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.model.parameters(), rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        if dist.get_rank() == 0:
            wandb.log({'step': self.step, 'samples': (self.step + 1) * self.global_batch}, step=self.step)
        logger.logkv("step", self.step)
        logger.logkv("samples", (self.step + 1) * self.global_batch)

    def save(self, for_preemption=False):
        def maybe_delete_earliest(filename):
            wc = filename.split(f"{(self.step):06d}")[0] + "*"
            freq_states = list(glob.glob(os.path.join(get_blob_logdir(), wc)))
            if len(freq_states) > 3000:
                earliest = min(freq_states, key=lambda x: x.split("_")[-1].split(".")[0])
                os.remove(earliest)

        # if dist.get_rank() == 0 and for_preemption:
        #     maybe_delete_earliest(get_blob_logdir())
        def save_checkpoint(rate, params, is_target_model=False):
            state_dict = self.model.state_dict()
            for i, (name, _) in enumerate(self.model.named_parameters()):
                assert name in state_dict
                state_dict[name] = params[i]
            if dist.get_rank() == 0:
                if is_target_model:
                    logger.log(f"saving target model...")
                else:
                    logger.log(f"saving model {rate}...")
                if not rate:
                    if is_target_model:
                        filename = f"target_{(self.step):06d}.pt"
                    else:
                        filename = f"model_{(self.step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step):06d}.pt"
                if for_preemption:
                    filename = f"freq_{filename}"
                    # maybe_delete_earliest(filename)

                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    torch.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            filename = f"opt_{(self.step):06d}.pt"
            if for_preemption:
                filename = f"freq_{filename}"
                # maybe_delete_earliest(filename)

            with bf.BlobFile(
                bf.join(get_blob_logdir(), filename),
                "wb",
            ) as f:
                torch.save(self.opt.state_dict(), f)
                
            filename = f"scaler_{(self.step):06d}.pt"
            if for_preemption:
                filename = f"freq_{filename}"
                # maybe_delete_earliest(filename)

            with bf.BlobFile(
                bf.join(get_blob_logdir(), filename),
                "wb",
            ) as f:
                torch.save(self.scaler.state_dict(), f)

        # Save model parameters last to prevent race conditions where a restart
        # loads model at step N, but opt/ema state isn't saved for step N.
        save_checkpoint(0, list(self.model.parameters()))
        # save_checkpoint(0, list(self.target_model.parameters()), is_target_model=True)
        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/model_NNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0
    
def parse_target_model_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/target_NNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("target_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    if main_checkpoint.split("/")[-1].startswith("freq"):
        prefix = "freq_"
    else:
        prefix = ""
    filename = f"{prefix}ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses, curr_step, log_interval):
    dictionary = {}
    for key, values in losses.items():
        dictionary[key] = values.mean().item()
        logger.logkv_mean(key, dictionary[key])
        # Log the quantiles (four quartiles, in particular).
        # for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #     quartile = int(4 * sub_t)
        #     logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

    if dist.get_rank() == 0 and curr_step % log_interval == 0:
        wandb.log(dictionary, step=curr_step)
        
def save_denoised_images(denoised_tensor, filename, num_display):
    """
    Note: 'denoised_tensor' input needs to be in the range of (-1, 1).
    """
    # assert denoised_tensor.min() >= -1, denoised_tensor.min()
    # assert denoised_tensor.max() <= 1, denoised_tensor.max()
    denoised_tensor = ((denoised_tensor + 1) * 127.5).clamp(0, 255).to(torch.uint8).contiguous()
    vutils.save_image(denoised_tensor[:num_display].float(), filename, normalize=True,  nrow=int(np.sqrt(num_display)))