import cv2
from matplotlib import pyplot as plt


def show_image(image = None, title = "Image", size = 5, color_conversion = cv2.COLOR_BGR2RGB):
    width, height = image.shape[0], image.shape[1]
    aspect_ratio = width/height
    plt.figure(figsize=(size * aspect_ratio,size))
    plt.imshow(cv2.cvtColor(image, color_conversion))
    plt.title(title)
    plt.show()