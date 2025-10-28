
DEVICES=0

BATCH_SIZE=16
SAVE_FREQ=4
SAVE_IMAGES=True
EPOCHS=300

ENCODER_PATH='/root/code/EventDiffusion/pretrained_models/l1_loss/model_epoch_300.pt'

DATE=1029_depth_encoder_l1
LR=5e-4

CUDA_VISIBLE_DEVICES=$DEVICES python depth_from_event_latents.py --date $DATE --loss_type ph \
 --bs $BATCH_SIZE --save_freq $SAVE_FREQ --save_images $SAVE_IMAGES --lr $LR --epochs $EPOCHS \
 --event_encoder_path $ENCODER_PATH


DATE=1029_depth_encoder_l1_2
LR=1e-3

CUDA_VISIBLE_DEVICES=$DEVICES python depth_from_event_latents.py --date $DATE --loss_type ph \
 --bs $BATCH_SIZE --save_freq $SAVE_FREQ --save_images $SAVE_IMAGES --lr $LR --epochs $EPOCHS \
 --event_encoder_path $ENCODER_PATH


DATE=1029_depth_encoder_l1_3
LR=1e-4

CUDA_VISIBLE_DEVICES=$DEVICES python depth_from_event_latents.py --date $DATE --loss_type ph \
 --bs $BATCH_SIZE --save_freq $SAVE_FREQ --save_images $SAVE_IMAGES --lr $LR --epochs $EPOCHS \
 --event_encoder_path $ENCODER_PATH


DATE=1029_depth_encoder_l1_4
LR=2e-5

CUDA_VISIBLE_DEVICES=$DEVICES python depth_from_event_latents.py --date $DATE --loss_type ph \
 --bs $BATCH_SIZE --save_freq $SAVE_FREQ --save_images $SAVE_IMAGES --lr $LR --epochs $EPOCHS \
 --event_encoder_path $ENCODER_PATH