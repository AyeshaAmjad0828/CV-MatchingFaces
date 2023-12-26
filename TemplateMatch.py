import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline


# img = plt.imread('Pictures/elon_face.jpg')
# plt.imshow(img)

elon_face = plt.imread('Pictures/elon_face.jpg')
elon = plt.imread('Pictures/Elon2.jpg')
plt.imshow(elon_face)

elon_face.shape
elon.shape

height,width,channels = elon_face.shape

resize_factor = 0.5  
resized_template = cv2.resize(elon_face, (int(width * resize_factor), int(height * resize_factor)))
height, width, channels = resized_template.shape

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']


for m in methods:
    original = elon.copy()
    
    method = eval(m)
    res = cv2.matchTemplate(original, resized_template, method)
    

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc    
    else:
        top_left = max_loc
        
    bottom_right = (top_left[0] + width, top_left[1] + height)


    cv2.rectangle(original,top_left, bottom_right, 255, 10)

    #Plotting the results of template matching
    plt.subplot(121) 
    plt.imshow(res)
    plt.title('Result of Template Matching')
    
    plt.subplot(122)
    plt.imshow(original)
    plt.title('Detected Point for face')
    plt.suptitle(m)
    
    plt.show()
    print('\n')
    print('\n')


###Trying iterative resizing approach for sensitive methods
    
elon_face.shape
elon.shape
height,width,channels = elon_face.shape
    
resize_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]  

#methods2 = ['cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
methods2 = ['cv2.TM_CCORR']

for factor in resize_factors:
    resized_template = cv2.resize(elon_face, (int(width * factor), int(height * factor)))
    height, width, channels = resized_template.shape

    for m in methods2:
        original = elon.copy()
        res = cv2.matchTemplate(original, resized_template, eval(m))

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + width, top_left[1] + height)

        
        cv2.rectangle(original,top_left, bottom_right, 255, 10)


        plt.subplot(121) 
        plt.imshow(res)
        plt.title(f'Resize Factor: {factor} - Method: {m}')
    
        plt.subplot(122)
        plt.imshow(original)
        plt.title('Detected Point for face')
        plt.suptitle(m)
    
    
        plt.show()
        print('\n')
        print('\n')
