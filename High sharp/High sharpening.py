"""
Title: Google Maps Image Sharpening
Author: Jitender Singh Virk (Virksaab)
Date created: 3 Oct, 2018
Last Modified: 3 Oct, 2018
"""

import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt

parent_data_dir = "F:\\SAVERA\\from 2-SEP-2018\\savera-raw-data"
# parent_data_dir = '/media/virk/SBPD/SAVERA/from 2-SEP-2018/savera-raw-data'
# Aug 27 data
data_dir1 = os.path.join(parent_data_dir, "Aug 27")

# GET ALL IMAGES PATH FROM FOLDERS
imagepaths = []
for dirpath, dirnames, filenames in os.walk(data_dir1):
    for filename in filenames:
        if 'Roofs' in dirpath:
            fullpath = os.path.join(dirpath, filename)
            imagepaths.append(fullpath)

# ITERATE OVER ALL IMAGES
for i, imgpath in enumerate(imagepaths[:20]):
    # GET IMAGE AND RESIZE
    bgrimg = cv2.imread(imgpath)
    bgrimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGRA2RGB)
    bgrimg = cv2.resize(bgrimg, (300, 300), interpolation=cv2.INTER_CUBIC)
    
    # HIGH SHARPEN
    kernel_sharp = np.array(([-2, -2, -2], [-2, 17, -2], [-2, -2, -2])) # High sharpening
    sbgrimg = cv2.filter2D(bgrimg, -1, kernel_sharp) # image sharpening
    sbgrimg = cv2.bilateralFilter(sbgrimg, 9, 100, 100) # Noise reduction

    # Display
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.title('Original')
    plt.imshow(bgrimg)
    plt.subplot(1,2,2)
    plt.axis('off')
    plt.title('Sharp')
    plt.imshow(sbgrimg)
    plt.show()
