import argparse

from .karras_diffusion import (
    KarrasDenoiser,
    VPNoiseSchedule,
    VENoiseSchedule,
    TrigFlowVENoiseSchedule,
    DBMPreCond,
)

NUM_CLASSES = 1000

def get_workdir(exp, date):
    workdir = f"./experiments/{date}/{exp}"
    return workdir


def sample_defaults():
    return dict(
        generator="determ",
        clip_denoised=True,
        sampler="cbmsolver",
        s_churn=0.0,
        s_tmin=0.002,
        s_tmax=80,
        s_noise=1.0,
        steps=40,
        model_path="",
        seed=42,
        ts="",
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        sigma_data=0.5,
        sigma_min=0.002,
        sigma_max=80.0,
        beta_d=2,
        beta_min=0.1,
        beta_max=1.0,
        cov_xy=0.0,
        original_img_size='260,346',
        resize_size=256,
        padding_to_size=384,
        in_channels=3,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=64,
        unet_type="adm",
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.1,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_fp16=True,
        use_bf16=False,
        use_new_attention_order=False,
        condition_mode=None,
        noise_schedule="ve",
        xT_norm=True,
        normalize_qk=False,
        c_noise_type="1000t",
        end_warmup_step=10000,
        return_logvar=False,
    )
    return res


def create_model_and_diffusion(
    dataset,
    resize_size,
    padding_to_size,
    original_img_size,
    in_channels,
    class_cond,
    num_channels,
    num_res_blocks,
    return_logvar,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_bf16,
    use_new_attention_order,
    condition_mode,
    noise_schedule,
    c_noise_type:str,
    xT_norm:bool,
    normalize_qk:bool,
    end_warmup_step:int,
    sigma_data:float,
    sigma_data_end,
    sigma_min=0.002,
    sigma_max=80.0,
    beta_d=2,
    beta_min=0.1,
    beta_max=1.0,
    cov_xy=0.0,
    unet_type="adm",
):
    if resize_size == 0:
        assert padding_to_size is not None, "Either resize_size or padding_to_size must be specified."
        resize_size = padding_to_size
        
    model = create_model(
        dataset,
        resize_size,
        in_channels,
        num_channels,
        num_res_blocks,
        normalize_qk=normalize_qk,
        return_logvar=return_logvar,
        unet_type=unet_type,
        channel_mult=channel_mult,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_bf16=use_bf16,
        use_new_attention_order=use_new_attention_order,
        condition_mode=condition_mode,
    )
    
    if noise_schedule.startswith("vp"):
        ns = VPNoiseSchedule(beta_d=beta_d, beta_min=beta_min)
        precond = DBMPreCond(ns, sigma_data=sigma_data, cov_xy=cov_xy, xT_normalization=xT_norm, c_noise_type=c_noise_type, sigma_data_end=sigma_data_end)
    elif noise_schedule == "ve":
        ns = VENoiseSchedule(sigma_max=sigma_max)
        precond = DBMPreCond(ns, sigma_data=sigma_data, cov_xy=cov_xy, xT_normalization=xT_norm, c_noise_type=c_noise_type, sigma_data_end=sigma_data_end)
    elif noise_schedule == "tf_ve":
        ns = TrigFlowVENoiseSchedule(sigma_max=sigma_max)
        precond = DBMPreCond(ns, sigma_data=sigma_data, cov_xy=cov_xy, xT_normalization=xT_norm, c_noise_type=c_noise_type, sigma_data_end=sigma_data_end)
    else:
        raise ValueError(f"Unknown noise schedule: {noise_schedule}")
    
    diffusion = KarrasDenoiser(
        noise_schedule=ns,
        precond=precond,
        t_max=sigma_max,
        t_min=sigma_min,
    )
    
    return model, diffusion


def create_model(
    dataset,
    resize_size,
    in_channels,
    num_channels,
    num_res_blocks,
    return_logvar:bool,
    normalize_qk:bool,
    unet_type="adm",
    channel_mult="",
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_bf16=False,
    use_new_attention_order=False,
    condition_mode=None,
):
    if dataset.__contains__("inpaint"):
        raise NotImplementedError
        from ddbm.unet_imagenet import UNetModel
        use_condition_labels = True
    else:
        from .unet import UNetModel
        # from .jvp_unet import UNetModel
        use_condition_labels = False
        
    assert unet_type in ["cbm_unet", "adm", "cbm_unet_sizeS", 
                         "adm_cbm", "adm_cbm2", "edm_scbm", "adm_test", "adm_cbm3"], unet_type
    
    # if channel_mult == "":
    #     if image_size == 512:
    #         channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
    #     elif image_size == 256:
    #         channel_mult = (1, 1, 2, 2, 4, 4)
    #     elif image_size == 128:
    #         channel_mult = (1, 1, 2, 3, 4)
    #     elif image_size == 64:
    #         channel_mult = (1, 2, 3, 4)
    #     elif image_size == 32:
    #         channel_mult = (1, 2, 3, 4)
    #     else:
    if channel_mult == "":
        if resize_size == 384:
            channel_mult = (1, 2)
        elif resize_size == 256:
            channel_mult = (1, 2)
        else:
            raise ValueError(f"unsupported image size: {resize_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(resize_size // int(res))

    if unet_type in ["adm", 'adm_cbm']:
        # if not use_new_attention_order:
        #     print("\nUsing legacy attention order. Set use_new_attention_order=True to use the new attention order.\n")
        #     attention_type = 'legacy'
        #     # use_checkpoint = not use_checkpoint
        # else:
        attention_type = 'flash'
        
        from .unet import LogVarUNet as UNetModel
        return UNetModel(
            logvar_channels=128,
            return_logvar=return_logvar,
            image_size=resize_size,
            in_channels=in_channels,
            model_channels=num_channels,
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=tuple(attention_ds),
            dropout=dropout,
            channel_mult=channel_mult,
            num_classes=(NUM_CLASSES if class_cond else None),
            use_checkpoint=use_checkpoint,
            # use_fp16=use_fp16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            condition_mode=condition_mode,
            attention_type=attention_type,
            normalize_qk=normalize_qk,
        )
    else:
        raise ValueError(f"Unsupported unet type: {unet_type}")


# def create_ema_and_scales_fn(
#     target_ema_mode,
#     start_ema,
#     scale_mode,
#     start_scales,
#     end_scales,
#     total_steps,
# ):
#     def ema_and_scales_fn(step=None):
#         if target_ema_mode == "fixed" and scale_mode == "fixed":
#             target_ema = start_ema
#             scales = start_scales
#         elif target_ema_mode == "fixed" and scale_mode == "progressive":
#             assert step is not None
#             target_ema = start_ema
#             scales = np.ceil(
#                 np.sqrt(
#                     (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
#                     + start_scales**2
#                 )
#                 - 1
#             ).astype(np.int32)
#             scales = np.maximum(scales, 1)
#             scales = scales + 1

#         elif target_ema_mode == "adaptive" and scale_mode == "progressive":
#             assert step is not None
#             scales = np.ceil(
#                 np.sqrt(
#                     (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
#                     + start_scales**2
#                 )
#                 - 1
#             ).astype(np.int32)
#             scales = np.maximum(scales, 1)
#             c = -np.log(start_ema) * start_scales
#             target_ema = np.exp(-c / scales)
#             scales = scales + 1
#         elif target_ema_mode == "fixed" and scale_mode in ["ict", "ict_m1"]:
#             assert step is not None
#             total_training_steps_prime = np.floor(
#                     total_steps
#                     / (np.log2(np.floor(end_scales / start_scales)) + 1)
#                 )
#             num_timesteps = start_scales * np.power(
#                 2, np.floor(step / total_training_steps_prime)
#             )
            
#             scales = min(num_timesteps, end_scales)
            
#             if scale_mode == "ict":
#                 scales = scales + 1
                
#             target_ema = start_ema
#         else:
#             raise NotImplementedError

#         return float(target_ema), int(scales)

#     return ema_and_scales_fn


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
