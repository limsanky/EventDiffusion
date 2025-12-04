DATASET_NAME=$1
PRED=$2

UNET=adm
ATTN=8,16,32

if [[ $DATASET_NAME == "e2d" ]]; then
    DATA_DIR=/root/data/MVSEC/ # added on 12th Sep
    DATASET=e2d
    IMG_SIZE=64

    NUM_CH=192
    NUM_RES_BLOCKS=3
    ATTN_TYPE=True

    EXP="e2d${IMG_SIZE}_${NUM_CH}d"
    SAVE_ITER=100000
    MICRO_BS=64
    DROPOUT=0.1
    CLASS_COND=False
fi
    
if  [[ $PRED == "ve" ]]; then
    EXP+="_ve"
    COND=concat
    SIGMA_MAX=80.0
    SIGMA_MIN=0.002
elif  [[ $PRED == "vp" ]]; then
    EXP+="_vp"
    COND=concat
    BETA_D=2
    BETA_MIN=0.1
    SIGMA_MAX=1
    SIGMA_MIN=0.0001
elif  [[ $PRED == "tf" ]]; then
    EXP+="_tf"
    COND=concat
    # SIGMA_MAX=1.5707
    SIGMA_MAX=1.5707963268
    SIGMA_MIN=0.0001
elif  [[ $PRED == "tf_ve" ]]; then
    EXP+="_tf_ve"
    COND=concat
    SIGMA_MAX=80.0
    SIGMA_MIN=0.002
else
    echo "Not supported"
    exit 1
fi