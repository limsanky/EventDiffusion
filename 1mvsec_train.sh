
DEVICES=1

DATE=1028_2
BATCH_SIZE=12
SAVE_FREQ=4
SAVE_IMAGES=True
LR=1e-3

CUDA_VISIBLE_DEVICES=$DEVICES python mvsec_ssl_event_representations.py --date $DATE --loss_type ph \
 --bs $BATCH_SIZE --save_freq $SAVE_FREQ --save_images $SAVE_IMAGES --lr $LR --epochs 400