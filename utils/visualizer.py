import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def tensor_to_disparity_image(tensor_data):
    assert len(tensor_data.size()) == 2
    assert (tensor_data >= 0.0).all().item()

    disparity_image = Image.fromarray(np.asarray(tensor_data * 256.0).astype(np.uint16))

    return disparity_image


def tensor_to_disparity_magma_image(tensor_data, vmax=None, mask=None):
    assert len(tensor_data.size()) == 2, tensor_data.shape
    assert (tensor_data >= 0.0).all().item()

    numpy_data = np.asarray(tensor_data)

    if vmax is not None:
        numpy_data = numpy_data * 255 / vmax
        numpy_data = np.clip(numpy_data, 0, 255)

    numpy_data = numpy_data.astype(np.uint8)
    numpy_data_magma = cv2.applyColorMap(numpy_data, cv2.COLORMAP_MAGMA)
    numpy_data_magma = cv2.cvtColor(numpy_data_magma, cv2.COLOR_BGR2RGB)

    if mask is not None:
        assert tensor_data.size() == mask.size()
        numpy_data_magma[~mask] = [255, 255, 255]

    disparity_image = Image.fromarray(numpy_data_magma)

    return disparity_image

def magma_save(vis_save_dir, file_name, pred, h, w):
    # print(pred.shape)
    # exit()
        
    # for idx in range(len(file_index)):
        # file_name = str(file_index[idx].item()).zfill(6)    # e.g., 000002, 000004, ...
    # save_path = vis_save_dir  + '/' + file_name + '.png'
    magma_save_path = vis_save_dir / file_name
    
    tensor_img = pred[:h, :w].cpu()
    # pred_img = tensor_to_disparity_image(tensor_img)
    pred_magma = tensor_to_disparity_magma_image(tensor_img, vmax=100)
    
    # pred_img.save(save_path)
    pred_magma.save(magma_save_path)