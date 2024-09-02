import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

def display_img_opencv_resize_first(img, img_name, new_max_dim=900):
    """
    We can resize the image before displaying it.  This is good
    for thumbnails.
    """
    max_dim = max(img.shape)
    scale = new_max_dim / max_dim
    new_height = int(img.shape[0] * scale)
    new_width = int(img.shape[1] * scale)
    img_1 = cv2.resize(img, (new_width, new_height))
    name_to_display = 'Resized ' + img_name
    cv2.imshow(name_to_display, img_1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = np.zeros((700,1600,3))

cv2.putText(img, "What language do geese speak?", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
cv2.putText(img, "Portugeese", (50, 220), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255,255,255), 2)
cv2.putText(img, "How many cans does it take to make a bird?", (50, 460), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
cv2.putText(img, "Two cans", (50, 580), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 2, (255,255,255), 2)

# dilate the text on the image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,13))
img_temp = cv2.dilate(img, kernel, iterations=2)

# convert to CV_8U
img_temp = np.uint8(img_temp)

# find edges
edge = cv2.Canny(img_temp, 127, 255) 

# overlay edges onto original image
img[edge == 255] = (255,255,255)

display_img_opencv_resize_first(img, "img")
