import cv2
import numpy as np

# Read image
filename='chess_tot3'
tail='.png'
img = cv2.imread('in/'+filename+tail)
hh, ww = img.shape[:2]

# threshold on white
# Define lower and uppper limits
lower = np.array([200, 200, 200])
upper = np.array([255, 255, 255])

# Create mask to only select black
thresh = cv2.inRange(img, lower, upper)

# apply morphology
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

# invert morp image
mask = 255 - morph

# apply mask to image
result = cv2.bitwise_and(img, img, mask=mask)


# save results
# cv2.imwrite('out/'+filename+'_thresh'+tail, thresh)
# cv2.imwrite('out/'+filename+'_morph'+tail, morph)
# cv2.imwrite('out/'+filename+'_mask'+tail, mask)
cv2.imwrite('out/'+filename+'_result'+tail, result)

# src = cv2.imread('out/'+filename+'_result'+tail, 1)
# without rm
src = cv2.imread('in/'+filename+tail)
tmp = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
_,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY) # THRESH_BINARY | THRESH_BINARY_INV | THRESH_OTSU
b, g, r = cv2.split(src)
rgba = [b,g,r, alpha]
dst = cv2.merge(rgba,4)
cv2.imwrite('out/'+filename+'_result_trans.png', dst)

# cv2.imshow('thresh', thresh)
# cv2.imshow('morph', morph)
# cv2.imshow('mask', mask)
# cv2.imshow('result', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()