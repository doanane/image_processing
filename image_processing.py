# import libraries
from skimage import io
from skimage import color
from pylab import *

 # Read image
img = io.imread('images/image2.jpg')

# # Convert to HSV
# img_hsv = color.rgb2hsv(img)
#  # Convert back to RGB
# img_rgb = color.hsv2rgb(img_hsv)


 
# figure(0)
# imshow(img_hsv)
# title('HSV Image')

# figure(1)
# imshow(img_rgb)
# title('RGB Image')

# show()

# # converting to gray scale
# img_gray = color.rgb2gray(img)
#  # Convert back to RGB
# img_rgb = color.gray2rgb(img_gray)
#  # Show both figures

# figure(0)
# imshow(img_gray)
# title('RGB Image')

# figure(1)
# imshow(img_rgb)
# title('GRAY Image')

# show()

# # converting to gray scale
# img_gray = color.rgb2gray(img)
#  # Convert back to RGB
# img_rgb = color.gray2rgb(img_gray)
#  # Show both figures

# figure(0)
# imshow(img_gray)
# title('RGB Image')

# figure(1)
# imshow(img_rgb)
# title('GRAY Image')

# show()


# Convert to XYZ Color Space
img_xyz = color.rgb2xyz(img)
 # Convert back to RGB
img_rgb = color.xyz2rgb(img_xyz)


 
figure(0)
imshow(img_xyz)
title('XYZ Image')

figure(1)
imshow(img_rgb)
title('RGB Image')

show()
