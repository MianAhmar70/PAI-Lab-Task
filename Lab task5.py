import cv2
import numpy as np
import matplotlib.pyplot as plt
# importing os module  
import os
# Python code to read image


# To read image from disk, we use
# cv2.imread function, in below method,
img = cv2.imread("E:/VENV/Ahmar.jpg", cv2.IMREAD_COLOR)

# Creating GUI window to display an image on screen
# first Parameter is windows title (should be in string format)
# Second Parameter is image array
cv2.imshow("image", img)


cv2.waitKey(0)


cv2.destroyAllWindows()
img.shape


#Displaying image using plt.imshow() method
plt.imshow(img)

#hold the window
plt.waitforbuttonpress()
plt.close('all')
# Converting BGR color to RGB color format
RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#Displaying image using plt.imshow() method
plt.imshow(RGB_img)

# hold the window
plt.waitforbuttonpress()
plt.close('all')
path = r"E:/VENV/Ahmar.jpg"

# Using cv2.imread() method 
# Using 0 to read image in grayscale mode 
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

# Displaying the image 
cv2.imshow('image', img) 
cv2.waitKey(0)
cv2.destroyAllWindows()
#writing the image
image_path = r"E:/VENV/Ahmar.jpg"

# Image directory
directory = r"E:/VENV/Ahmar.jpg"

img = cv2.imread(image_path)

os.chdir(directory)
print("Before saving image:")  
print(os.listdir(directory))  

# Filename
filename = 'savedImage.jpg'

# Using cv2.imwrite() method
# Saving the image
cv2.imwrite(filename, img)
#Arithmatic operations
image1 = cv2.imread("E:/VENV/Ahmar.jpg")  
image2 = cv2.imread("E:/VENV/Ahmar.jpg") 
weightedSum = cv2.addWeighted(image1, 0.5, image2, 0.4, 0) 
cv2.imshow('Weighted Image', weightedSum)
if cv2.waitKey(0) & 0xff == 27:  
    cv2.destroyAllWindows()  
    
#blur the image

# Gaussian Blur 
Gaussian = cv2.GaussianBlur(image1, (7, 7), 0) 
cv2.imshow('Gaussian Blurring', Gaussian) 
cv2.waitKey(0) 
  
# Median Blur 
median = cv2.medianBlur(image1, 5) 
cv2.imshow('Median Blurring', median) 
cv2.waitKey(0) 
  
  
# Bilateral Blur 
bilateral = cv2.bilateralFilter(image1, 9, 75, 75) 
cv2.imshow('Bilateral Blurring', bilateral) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 

#Create borders 

image = cv2.imread(path) 
   
# Window name in which image is displayed 
window_name = 'Image'
  
# Using cv2.copyMakeBorder() method 
image = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, None, value = 0) 
  
# Displaying the image  
cv2.imshow(window_name, image)

#Gray Scalling 

# Use the cvtColor() function to grayscale the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale', gray_image)
cv2.waitKey(0)  

# Window shown waits for any key pressing event
cv2.destroyAllWindows()

img = cv2.imread('E:/VENV/Ahmar.jpg', 0)

cv2.imshow('Grayscale Image', img)
cv2.waitKey(0)

# Window shown waits for any key pressing event
cv2.destroyAllWindows

