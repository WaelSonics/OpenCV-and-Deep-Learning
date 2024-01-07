import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main():
    img = cv.imread('OpenCv/Cats/3.jpg')
    img = cv.resize(img, (400, 600))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #laplace edge detection:
    laplace = cv.Laplacian(gray, cv.CV_64F)
    laplace = np.uint8(np.absolute(laplace))        #image specific data type
    # cv.imshow("Lap", laplace)

    #sobel      Most advanced edge detection
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1)
    # cv.imshow('Sobelx', sobelx)
    # cv.imshow('SObely', sobely)
    sobel = cv.bitwise_or(sobelx, sobely)
    # cv.imshow('Sobel', sobel)

    #canny
    canny = cv.Canny(gray, 150, 175)
    # cv.imshow('Canny', canny)

    #Face Detection:::
    wael = cv.imread('OpenCv/Cats/wael.jpg')
    wael = cv.resize(wael, (300, 400))
    # cv.imshow('Wael', wael)
    haar = cv.CascadeClassifier('haarLike.xml')
    face_rect = haar.detectMultiScale(wael, scaleFactor=1.1, minNeighbors=4)
    print(f'Number of faces = \n {len(face_rect)}')   
    for (x, y, w, h) in face_rect:
        cv.rectangle(wael, (x,y), (x+w, y+h), (0,255,0) , thickness=2)
    # cv.imshow('wael', wael)


    #face detection in video:
    capture = cv.VideoCapture(0)        #read from webcam
    while True:
        isTrue, frame = capture.read()
        face_rect = haar.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=4)
        for (x, y, w, h) in face_rect:
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0) , thickness=2)
        cv.imshow('V', frame)
        if not isTrue or (cv.waitKey(20) & 0xFF == ord('a')):       #press 'a' to exit both
            print("End of video.")
            break
    capture.release()
    cv.destroyAllWindows()



    cv.waitKey(0)

if __name__ == "__main__":
    main()
