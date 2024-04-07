import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import measure

def separate(original_image, channel_value=0):
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    hue_channel = hsv_image[:, :, 0]
    if channel_value == 0:
        hist = np.histogram(hue_channel, bins=180, range=[0, 180])[0][1:]
        hist = hist[hist != 0]
        hue_mean = np.mean(hist)
        hue_median = np.median(hist)
        print("mean", hue_mean)
        print("median", hue_median)
        channel_values = {
            (0, 48): 115,
            (48, 73): 108,
            (73, 80): 100,
            (80, 140): 105,
            (140, 150): 115,
            (150, 180): 117,
            (180, 400): 120,
            (400, 10000): 125,
        }

        for (min_value, max_value), channel in channel_values.items():
            if hue_mean>100 and hue_median<30:
                print("ERROR: Cannot be separated")
            if min_value <= hue_mean < max_value:
                channel_value = channel
                break
        else:
            channel_value = 120      

    
    mask = hsv_image[:, :, 0] > channel_value
    blobs_labels = measure.label(mask, background=0)
    if blobs_labels.max() == 0:
        return None, None
    regions = measure.regionprops(blobs_labels)
    largest_component = max(regions, key=lambda prop: prop.area)
    biggest_component_coords = largest_component.coords
    component_mask = np.zeros_like(mask, dtype=np.uint8)
    component_mask[biggest_component_coords[:, 0], biggest_component_coords[:, 1]] = 255
    sign_final_image = np.where(component_mask[:, :, None] == 255, original_image, np.ones_like(original_image) * 255)
    stamp_final_image = np.where(255 - component_mask[:, :, None] == 255, original_image, np.ones_like(original_image) * 255)
    
    return sign_final_image, stamp_final_image


''' 
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

'''
