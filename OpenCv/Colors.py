import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main():
    img = cv.imread('OpenCv/Cats/3.jpg')
    img = cv.resize(img, (400, 600))
    cv.imshow('IMG', img)
    # plt.imshow(img)               #img shown in reverse
    # plt.show()

    print(img.shape[:2])

    #bgr to rgb
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # plt.imshow(rgb)           
    # plt.show()
    
    #bgr to gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # cv.imshow('Gray', gray)

    #split into b, g, r
    b, g, r= cv.split(img)  
    cv.imshow('B', b)               #they will be shown in grey scale, because each is 2D, representing the 
    cv.imshow('G', g)               #intensity of the pixels 
    cv.imshow('R', r)

    #to display them in their colors we can merge each with a blank
    blank = np.zeros(img.shape[:2], dtype = 'uint8')
    cv.imshow('Blank', blank)
    green = cv.merge([blank, b, blank])
    cv.imshow('Green', green)



    cv.waitKey(0)

    
if __name__ == "__main__":
    main()