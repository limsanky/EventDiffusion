
DATE=1027
BATCH_SIZE=12
SAVE_FREQ=4

python mvsec_ssl_event_representations.py --date $DATE --loss_type l1 --bs $BATCH_SIZE --save_freq $SAVE_FREQ
python mvsec_ssl_event_representations.py --date $DATE --loss_type ph --bs $BATCH_SIZE --save_freq $SAVE_FREQ