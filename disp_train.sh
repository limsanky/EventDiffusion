
DEVICES=0

BATCH_SIZE=1
SAVE_FREQ=4
SAVE_IMAGES=True
EPOCHS=300
USE_IMAGES=False

ENCODER_PATH='/root/code/EventDiffusion/pretrained_models/l1_loss/model_epoch_300.pt'

DATE=1030_new_disp_encoder_l1_mse
# LR=2e-5
LR=1e-3


CUDA_VISIBLE_DEVICES=$DEVICES python disp_from_event_latents.py --date $DATE --loss_type mse \
 --bs $BATCH_SIZE --save_freq $SAVE_FREQ --save_images ${SAVE_IMAGES} --lr $LR --epochs $EPOCHS \
 --event_encoder_path $ENCODER_PATH --use_images ${USE_IMAGES} 
 

# DATE=1029_new_disp_encoder_l1_2_mse
# LR=1e-4

# CUDA_VISIBLE_DEVICES=$DEVICES python disp_from_event_latents.py --date $DATE --loss_type mse \
#  --bs $BATCH_SIZE --save_freq $SAVE_FREQ --save_images $SAVE_IMAGES --lr $LR --epochs $EPOCHS \
#  --event_encoder_path $ENCODER_PATH --use_images $USE_IMAGES

DATE=1030_new_disp_encoder_l1_3_mse
LR=5e-4
# LR=5e-5

CUDA_VISIBLE_DEVICES=$DEVICES python disp_from_event_latents.py --date $DATE --loss_type mse \
 --bs $BATCH_SIZE --save_freq $SAVE_FREQ --save_images $SAVE_IMAGES --lr $LR --epochs $EPOCHS \
 --event_encoder_path $ENCODER_PATH --use_images $USE_IMAGES


DATE=1030_new_disp_encoder_l1_4_mse
LR=2e-4

CUDA_VISIBLE_DEVICES=$DEVICES python disp_from_event_latents.py --date $DATE --loss_type mse \
 --bs $BATCH_SIZE --save_freq $SAVE_FREQ --save_images $SAVE_IMAGES --lr $LR --epochs $EPOCHS \
 --event_encoder_path $ENCODER_PATH --use_images $USE_IMAGES