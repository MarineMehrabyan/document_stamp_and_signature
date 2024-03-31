import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import measure


def separate(original_image):

    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    hue_mean = np.mean(hsv_image[:,:,0])
    #print(f"Mean hue: {hue_mean}")

    channel = 0
    if hue_mean >= 48 and hue_mean < 78:
        channel = 115
    elif hue_mean >= 75 and hue_mean < 80:
        channel = 100
    elif hue_mean >= 80 and hue_mean < 100:
        channel = 105
    else:
        channel = 120

    #print(channel)
    mask = hsv_image[:,:,0] > channel
    
    blobs_labels = measure.label(mask, background=0)
    biggest_component = None
    biggest_area = 0
    

    for region in measure.regionprops(blobs_labels):
        if region.area > biggest_area:
            biggest_area = region.area
            biggest_component = region.coords
            
    component_mask = np.zeros_like(mask, dtype=np.uint8)
    component_mask[biggest_component[:, 0], biggest_component[:, 1]] = 255
    sign_final_image = np.where(component_mask[:, :, None] == 255, 
                                original_image, np.ones_like(original_image) * 255)
    
    inverted_mask = 255 - component_mask
    stamp_final_image = np.where(inverted_mask[:, :, None] == 255, original_image, 
                                 np.ones_like(original_image) * 255)
    

    return sign_final_image, stamp_final_image


