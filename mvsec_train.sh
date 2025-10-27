
DATE=1027_2
BATCH_SIZE=12
SAVE_FREQ=4
SAVE_IMAGES=True

python mvsec_ssl_event_representations.py --date $DATE --loss_type l1 --bs $BATCH_SIZE --save_freq $SAVE_FREQ --save_images $SAVE_IMAGES
python mvsec_ssl_event_representations.py --date $DATE --loss_type ph --bs $BATCH_SIZE --save_freq $SAVE_FREQ --save_images $SAVE_IMAGES