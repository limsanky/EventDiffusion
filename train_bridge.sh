export PYTHONPATH=$PYTHONPATH:./

# DATE=1128
# DATE=1202
DATE=1204
DATASET_NAME=e2d
PRED=vp

source diffusion_args.sh $DATASET_NAME $PRED

C_NOISE_TYPE=1000t
XT_NORM=True
EMA_RATE="0.9993,0.9999"

USE_DISP_MASK=True

SAVE_ITER=1000
# SAVE_ITER=10
FREQ_SAVE_ITER=5000
EXP=${DATASET_NAME}-${PRED}

# CKPT=assets/ckpts/256x256_diffusion_fixedsigma.pt

# For cluster
# export ADDR=$1
# run_args="--nproc_per_node 8 \
#           --master_addr $ADDR \
#           --node_rank $RANK \
#           --master_port $MASTER_PORT \
#           --nnodes $WORLD_SIZE"
# For local
# export CUDA_VISIBLE_DEVICES=0
# run_args="--nproc_per_node 1 --master_port 29511"
export CUDA_VISIBLE_DEVICES=0,1
run_args="--nproc_per_node 2 --master_port 29511"

BS=16
MICRO_BS=2
# BS=2
# MICRO_BS=2
LR=0.0001

USE_FP16=False
USE_BF16=True

SCHEDULER="real-uniform"
EVENT_ENCODER_PATH=/root/code/EventDiffusion/pretrained_models/l1_loss/model_epoch_300.pt

ORG_IMG_SIZE=260,346
# RESIZE_SIZE=256
PADDING_TO_SIZE=384
RESIZE_SIZE=0

MINS="-3.944004774093628,0.0,-2.6182522773742676"
MAXES="4.1059675216674805,1.675461769104004,2.372314453125"
MEANS="0.4872281551361084,0.08612743020057678,0.5240721702575684"
STDS="0.07482649385929108,0.07482782006263733,0.04035165160894394"

# DISP_MEAN=0.26924431324005127
DISP_STD=0.1981430947780609

torchrun $run_args train_bridge.py --exp=$EXP --date $DATE \
 --class_cond $CLASS_COND --c_noise_type $C_NOISE_TYPE \
 --dropout $DROPOUT  --microbatch $MICRO_BS \
 --original_img_size $ORG_IMG_SIZE --resize_size $RESIZE_SIZE  --num_channels $NUM_CH  \
 --num_res_blocks $NUM_RES_BLOCKS  --condition_mode=$COND  \
 --noise_schedule=$PRED  --batch_size=$BS --lr=$LR \
 --use_new_attention_order $ATTN_TYPE  \
 ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"} ${BETA_MAX:+ --beta_max="${BETA_MAX}"}  \
 --data_dir=$DATA_DIR --dataset=$DATASET --use_fp16=$USE_FP16 --use_bf16=$USE_BF16 \
 --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN --xT_norm=$XT_NORM \
 --save_interval_for_preemption=$FREQ_SAVE_ITER --save_interval=$SAVE_ITER --debug=False \
 ${CKPT:+ --resume_checkpoint="${CKPT}"} --schedule_sampler=$SCHEDULER \
 --event_encoder_path=$EVENT_ENCODER_PATH \
 --n_channels=3 --out_depth=1 --bilinear=True --n_lyr=4 --ch1=24 --c_is_const=False --c_is_scalar=False \
 --event_latent_stds=$STDS --event_latent_means=$MEANS --max_event_latent_vals=$MAXES --min_event_latent_vals=$MINS \
 --sigma_data=$DISP_STD --ema_rate=$EMA_RATE --use_disp_mask=$USE_DISP_MASK --padding_to_size=$PADDING_TO_SIZE