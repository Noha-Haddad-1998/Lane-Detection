import cv2.cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt

def canny(image):
    # creating grayscale from the coloured image as it has only one channel
    grayImage = cv2.cvtColor(lane_image, cv2.cv2.COLOR_RGB2GRAY)
    # cv2.imshow('gray Lane', grayImage)
    # cv2.waitKey(0)

    # noise reduction
    blurredImage = cv2.medianBlur(grayImage, 5)
    # cv2.imshow('blurred Lane', blurredImage)
    # cv2.waitKey(0)

    # compute the gradient in all directions of our blurred image
    cannyImage = cv2.Canny(blurredImage, 20, 130)
    # cv2.imshow('canny Lane',cannyImage)  # traces an outline of the edges that correspond to the most sharp changes in intensity
    # cv2.waitKey(0)
    return cannyImage

def region_of_interest(image):
     hight = image.shape[0]
     #since fillpoly fills an area bounded by several polygons not just one
     polygons1 = np.array([[(200,hight),(1100, hight), (550,250)]]) #array of polygons
     polygons2 = np.array([[(0,400),(0,700),(570,290)]])
     mask1 = np.zeros_like(image) #creates an array with the same shape as the imaged corresponding array
     mask2 = np.zeros_like(image) #creates an array with the same shape as the imaged corresponding array
     #fill this mask with polygons
     cv2.fillPoly(mask1, polygons1, 250)
     cv2.fillPoly(mask2,polygons2,250)
     maskedImage1 = cv2.bitwise_and(image,mask1)
     maskedImage2 = cv2.bitwise_and(image,mask2)
     maskedImage= cv2.cv2.bitwise_or(maskedImage1,maskedImage2)
     # cv2.imshow('ROI', maskedImage)
     # cv2.waitKey(0)
     return maskedImage

# This is the function that will build the Hough Accumulator for the given image
def hough_lines_acc(img):

    height= img.shape[0]
    width = img.shape[1] # we need heigth and width to calculate the diag
    maxRadius = np.sqrt(height*height + width*width) # a**2 + b**2 = c**2
    rhos = np.arange(-maxRadius, maxRadius,step=(2*maxRadius/maxRadius))
    thetas = np.deg2rad(np.arange(0,181))
    cos=np.cos(thetas)
    sin= np.sin(thetas)

    H = np.zeros((len(rhos),181))
    print(H.shape)
    for y in range(height): # cycle through edge points
     for x in range(width):
        if img[y][x] !=0:
            for theta in range(181): # cycle through thetas and calc rho
                rho = (x-width/2) * cos[theta] + (y-height/2)*sin[theta]
                rhoIndex=np.argmin(np.abs(rhos-rho))
                H[rhoIndex, theta] += 1
    return H, rhos, thetas,width,height


#######################################################
#drawing lines on the original image
def hough_lines_draw(img,width,height,H,rhos, thetas):
    lineImage=np.zeros_like(image)
    for y in range(H.shape[0]):
      for x in range(H.shape[1]):
         if H[y][x] > 280:
            rho = rhos[y]
            theta = thetas[x]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = (a * rho) + (width/2)
            y0 = (b * rho) + (height/2)
            # these are then scaled so that the lines go off the edges of the image
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(lineImage, (x1, y1), (x2, y2), (255, 0, 0), 10)
    polygons = np.array([[(0,330 ),(0,700), (1100, img.shape[0]), (710,400)]])  # array of polygons
    mask = np.zeros_like(img)
    mask=cv2.fillPoly(mask, polygons, (255, 0, 0))
    maskedImage2 = cv2.bitwise_and(lineImage, mask)
    cv2.imshow('Lane', mask)
    cv2.waitKey(0)
    return maskedImage2

#Project begins
image = cv2.imread('lane2.jpg')
lane_image = np.copy(image)#copying image as not to change the original array
cannyImage=canny(lane_image)
ROI=region_of_interest(cannyImage)
H, rhos,thetas,width,height= hough_lines_acc(ROI)
img=hough_lines_draw(image,width,height,H,rhos,thetas)
comboImage = cv2.cv2.addWeighted(image,0.8,img,3,3)
cv2.imshow('Lane', comboImage)
cv2.waitKey(0)
