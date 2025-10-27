
DEVICES=0

DATE=1027
BATCH_SIZE=12
SAVE_FREQ=4
SAVE_IMAGES=True

CUDA_VISIBLE_DEVICES=$DEVICES python mvsec_ssl_event_representations.py --date $DATE --loss_type l1 --bs $BATCH_SIZE --save_freq $SAVE_FREQ --save_images $SAVE_IMAGES