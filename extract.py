import os
import cv2
import matplotlib.pyplot as plt
from skimage import measure, morphology
from skimage.measure import regionprops
import numpy as np


def process_image(source_image):
    scale = 800 / max(source_image.shape[:2]) if max(source_image.shape[:2]) > 800 else 1.0
    image = cv2.resize(source_image, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def extract_not_blue(source_image, constant_parameter_1 = 84,
                      constant_parameter_2 = 250,
                      constant_parameter_3 = 100,
                      constant_parameter_4 = 18):
    img = source_image

    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary

    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)
    the_biggest_component = 0
    total_area = 0
    counter = 0
    average = 0.0
    for region in regionprops(blobs_labels):

        if (region.area > 10):
            total_area = total_area + region.area
            counter = counter + 1
        if (region.area >= 250):
            if (region.area > the_biggest_component):
                the_biggest_component = region.area

    average = (total_area/counter)
    a4_small_size_outliar_constant = ((average/constant_parameter_1)*
                                      constant_parameter_2)+constant_parameter_3
    #print("a4_small_size_outliar_constant: " + str(a4_small_size_outliar_constant))

    a4_big_size_outliar_constant = a4_small_size_outliar_constant*constant_parameter_4
    #print("a4_big_size_outliar_constant: " + str(a4_big_size_outliar_constant))

    pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outliar_constant)
    component_sizes = np.bincount(pre_version.ravel())
    too_small = component_sizes > (a4_big_size_outliar_constant)
    #too_small-ը օգտագործվում է որպես mask
    too_small_mask = too_small[pre_version]

    pre_version[too_small_mask] = 0

    plt.imsave('pre_version.png', pre_version)
    img = cv2.imread('pre_version.png', 0)
    img = cv2.threshold(img, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return img

def extract(image):
    image_copy = image
    factor=0.5
    factor = max(0, min(1, factor))
    b, g, r = cv2.split(image)                    
    b = np.clip(b * factor, 0, 255).astype(np.uint8)
    g = np.clip(g * factor, 0, 255).astype(np.uint8)
    r = np.clip(r * factor, 0, 255).astype(np.uint8)
    image = cv2.merge([b, g, r])                      
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue_hue = 100
    upper_blue_hue = 140
    blue_mask = cv2.inRange(hsv_image, (lower_blue_hue, 50, 50), (upper_blue_hue, 255, 255))
    blue_mask = cv2.merge([blue_mask, blue_mask, blue_mask])
    # ete kapuyt masky datark 0 e nshanakum e petq e kanchem en severi functony
    if np.all(blue_mask == 0):
        #print("img is not blue")
        img = process_image(image_copy)
        img = extract_not_blue(img)
        return img
    else: 
        white_image = np.ones_like(image) * 255
        result_image = np.where(blue_mask > 0, image, white_image)
        gray_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
        _, thresholded_image = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY_INV)
        inverted_image = cv2.bitwise_not(thresholded_image)

        return inverted_image
    
