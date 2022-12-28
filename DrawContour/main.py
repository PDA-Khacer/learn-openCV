import cv2
import numpy as np

 
image = cv2.imread('in/chess_tot2.png')

# convert the image to grayscale format
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply binary thresholding
ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
binary=thresh
# visualize the binary image

contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                      
# draw contours on the original image
image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
                
# see the results
cv2.imwrite('contours_none_image1.jpg', image_copy)

# create a mask for floodfill function, see documentation
h,w,_ = image.shape
mask = np.zeros((h+2,w+2), np.uint8)

# determine which contour belongs to a square or rectangle
for cnt in contours:
    poly = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt,True),True)
    if len(poly) == 4:
        # if the contour has 4 vertices then floodfill that contour with black color
        cnt = np.vstack(cnt).squeeze()
        _,binary,_,_ = cv2.floodFill(binary, mask, tuple(cnt[0]), 0)
# convert image back to original color
binary = cv2.bitwise_not(binary)        

cv2.imshow('Image', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ################################################################################################
# image1 = cv2.imread('in/chess_tot2.png')
# img_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
 
# ret, thresh1 = cv2.threshold(img_gray1, 150, 255, cv2.THRESH_BINARY)
# contours2, hierarchy2 = cv2.findContours(thresh1, cv2.RETR_TREE,
#                                                cv2.CHAIN_APPROX_SIMPLE)
# image_copy2 = image1.copy()
# cv2.drawContours(image_copy2, contours2, -1, (0, 255, 0), 1, cv2.LINE_AA)
# image_copy3 = image1.copy()
# for i, contour in enumerate(contours2): # loop over one contour area
#    for j, contour_point in enumerate(contour): # loop over the points
#        # draw a circle on the current contour coordinate
#        cv2.circle(image_copy3, ((contour_point[0][0], contour_point[0][1])), 2, (0, 255, 0), 1, cv2.LINE_AA)
# # see the results
# cv2.imwrite('contour_point_simple.jpg', image_copy3)