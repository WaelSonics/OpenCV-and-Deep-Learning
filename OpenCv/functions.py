import cv2 as cv
import numpy  as np

def main():
    img = cv.imread('OpenCv/Cats/1.jpg')

    #resize
    img = cv.resize(img, (400, 600))
    cv.imshow('IMG', img)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #cv.imshow('Gray', gray)
    blur = cv.GaussianBlur(img, (5,5), cv.BORDER_DEFAULT)       #kernel size inc --> more blur
    #cv.imshow('Blur', blur)
    
    # Edge Cascade
    canny = cv.Canny(blur, 150, 150)                            # more blur --> less edges 
    #cv.imshow('Canny Edges', canny)

    # Cropping (slicing)
    cropped = img[50:200, 200:400]
    # cv.imshow('Cropped', cropped)
    
    #translating 
    transImg = translate(img, -50, 100)
    # cv.imshow('Translated', transImg)

    #rotation
    rotImg = rotate(img, 90)    
    # cv.imshow('Rotated', rotImg)

    #flipping image
    flippedImg = cv.flip(img, -1)
    cv.imshow('flip', flippedImg)

    cv.waitKey(0)

def translate(img, x, y):
    '''Function that translates images
        x --> Right
       -x --> Left
        y --> Down
       -y --> Up
    '''
    trans = np.float32([[1,0,x], [0,1,y]])
    dimentions = (img.shape[1], img.shape[0])
    return cv.warpAffine(img, trans, dimentions)

def rotate(img, angle, center = None):
    (height, width) = img.shape[:2]
    dimentions = (height, width)  #(width, height) 
    if center is None:
        center = (width//2, height//2)
    rot = cv.getRotationMatrix2D(center, angle, 1.0)
    return cv.warpAffine(img, rot, dimentions, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(255, 255, 255))

if __name__ == "__main__":
    main()