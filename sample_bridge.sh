
export PYTHONPATH=$PYTHONPATH:./

# For cluster
# export ADDR=$1
# run_args="--nproc_per_node 8 \
#           --master_addr $ADDR \
#           --node_rank $RANK \
#           --master_port $MASTER_PORT \
#           --nnodes $WORLD_SIZE"
# For local
# export CUDA_VISIBLE_DEVICES=0,1
# run_args="--nproc_per_node 2 --master_port 29511"
export CUDA_VISIBLE_DEVICES=0
run_args="--nproc_per_node 1 --master_port 29511"

DATASET_NAME=e2d
TRAINING_DATA_SPLIT=1

# Batch size per GPU
BS=2
# BS=6
SEED=-1
# SEED=42

# Dataset and checkpoint
PRED=vp

source diffusion_args.sh $DATASET_NAME $PRED

MAX_DISP=37
ORG_IMG_SIZE=260,346
RESIZE_SIZE=256

UNET=adm
RET_LOGVAR=False
XT_NORM=True
C_NOISE_TYPE=1000t
NORMALIZE_QK=False
USE_FP16=False
USE_BF16=True

# EVAL_SPLIT=train
EVAL_SPLIT=test

EVENT_ENCODER_PATH=/root/code/EventDiffusion/pretrained_models/l1_loss/model_epoch_300.pt
# MODEL_PATH=/root/code/EventDiffusion/workdir/ema_0.9993_035000.pt
MODEL_PATH=/root/code/EventDiffusion/workdir/ema_0.9993_074000.pt

SAVE_DIR=/root/code/EventDiffusion/workdir/${PRED}/

MINS="-3.944004774093628,0.0,-2.6182522773742676"
MAXES="4.1059675216674805,1.675461769104004,2.372314453125"
MEANS="0.4872281551361084,0.08612743020057678,0.5240721702575684"
STDS="0.07482649385929108,0.07482782006263733,0.04035165160894394"

DISP_STD=0.1981430947780609

DBM3_RATIO_FIRST=None
DBM3_RATIO_SECOND=None

# Sampler
GEN_SAMPLER=$1

# Number of function evaluations (NFE)
NFE=$2

ORDER=1

if [[ $GEN_SAMPLER == "ground_truth" ]]; then
    N=$((NFE))
elif [[ $GEN_SAMPLER == "dbim" ]]; then
    N=$((NFE))
elif [[ $GEN_SAMPLER == "euler" ]]; then
    N=$((NFE))
elif [[ $GEN_SAMPLER == "exp_euler" ]]; then
    N=$((NFE))
    ORDER=$3
    if [[ $ORDER == 2 ]]; then
        N=$(echo "$NFE" | awk '{print ($1 + 2) / 2}')
    fi
fi


export OMP_NUM_THREADS=8 
export WANDB_MODE=offline 
export OMPI_MCA_opal_cuda_support=true

# torchrun $run_args sample_bridge.py \

torchrun $run_args new_sample_bridge.py \
 --steps $N --sampler $GEN_SAMPLER --batch_size $BS --eval_split $EVAL_SPLIT \
 --model_path $MODEL_PATH --class_cond $CLASS_COND --noise_schedule $PRED \
 ${BETA_D:+ --beta_d="${BETA_D}"} ${BETA_MIN:+ --beta_min="${BETA_MIN}"} ${BETA_MAX:+ --beta_max="${BETA_MAX}"} \
 --condition_mode=$COND  --sigma_max=$SIGMA_MAX --sigma_min=$SIGMA_MIN \
 --dropout $DROPOUT --original_img_size $ORG_IMG_SIZE --num_channels $NUM_CH  --num_res_blocks $NUM_RES_BLOCKS \
 --use_new_attention_order $ATTN_TYPE --data_dir=$DATA_DIR --dataset=$DATASET --training_data_split $TRAINING_DATA_SPLIT \
 ${CHURN_STEP_RATIO:+ --churn_step_ratio="${CHURN_STEP_RATIO}"} ${ETA:+ --eta="${ETA}"} \
 ${ORDER:+ --order="${ORDER}"} --use_fp16=$USE_FP16 --use_bf16=$USE_BF16 --attention_resolutions $ATTN --unet_type $UNET \
 --dbm3_ratio_first ${DBM3_RATIO_FIRST} --dbm3_ratio_second ${DBM3_RATIO_SECOND} --seed $SEED --c_noise_type $C_NOISE_TYPE \
 --normalize_qk=$NORMALIZE_QK \
 --xT_norm=$XT_NORM --c_noise_type=$C_NOISE_TYPE --sigma_data=$DISP_STD \
 --event_latent_stds=$STDS --event_latent_means=$MEANS --max_event_latent_vals=$MAXES --min_event_latent_vals=$MINS \
 --n_channels=3 --out_depth=1 --bilinear=True --n_lyr=4 --ch1=24 --c_is_const=False --c_is_scalar=False \
 --event_encoder_path=$EVENT_ENCODER_PATH --max_disp=$MAX_DISP 