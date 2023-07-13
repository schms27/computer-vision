import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_image(image = None, title = "Image", size = 5, color_conversion = cv2.COLOR_BGR2RGB):
    width, height = image.shape[0], image.shape[1]
    aspect_ratio = width/height
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, color_conversion))
    plt.title(title)
    plt.show()

def median_canny(image, lower_thresh_ratio=0.6, upper_thres_ratio=1.4):
    blurred = cv2.blur(image, ksize=(5,5))
    median_value = np.median(blurred) 
    lower = int(max(0, lower_thresh_ratio * median_value))
    upper = int(min(255, upper_thres_ratio * median_value))
    return cv2.Canny(blurred, threshold1=lower, threshold2=upper)