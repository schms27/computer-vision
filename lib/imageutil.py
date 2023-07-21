import cv2
import PIL
import numpy as np
from io import BytesIO
from IPython import display
from matplotlib import pyplot as plt


def show_image(image = None, title = "Image", size = 5, color_conversion = cv2.COLOR_BGR2RGB):
    width, height = image.shape[0], image.shape[1]
    aspect_ratio = width/height
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, color_conversion))
    plt.title(title)
    plt.show()

def visualize_image_grid(images, title = "Imagegrid", size = (10,10), subplot_grid = (8, 8), save_to_file=None, cmap='gray'):
    plt.figure(figsize=size)

    for i, img in enumerate(images):
        plt.subplot(subplot_grid[0], subplot_grid[1], i+1)
        plt.imshow(img, cmap=cmap)
        plt.axis('off')
        
    if save_to_file != None:
        plt.savefig(save_to_file)

    if title != None:
        plt.suptitle(title)
        plt.show()
    
    plt.close()

def median_canny(image, lower_thresh_ratio=0.6, upper_thres_ratio=1.4):
    blurred = cv2.blur(image, ksize=(5,5))
    median_value = np.median(blurred) 
    lower = int(max(0, lower_thresh_ratio * median_value))
    upper = int(min(255, upper_thres_ratio * median_value))
    return cv2.Canny(blurred, threshold1=lower, threshold2=upper)

def array_to_image(array, format='jpeg'):
    # create binary stream object
    frame = BytesIO()

    # convert array to binary stream object
    PIL.Image.fromarray(array).save(frame, format)

    return display.Image(data=frame.getvalue())

def mse(image_1, image_2):
    if image_1.shape != image_2.shape:
        raise Exception("Images need to be of the same dimensions")
    error = np.sum((image_1.astype("float") - image_2.astype("float")) ** 2)
    error /= float(image_1.shape[0] * image_1.shape[1]) 
    return error