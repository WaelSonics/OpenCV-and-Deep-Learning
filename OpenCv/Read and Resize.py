import cv2 as cv

# img1 = cv.imread('Cats/1.jpg')
# cv.imshow('1', img1)


def rescaleFrame(frame, scale=0.75):
    '''Function that rescales images or frames of videos 
    (works for images, videos and live vids)'''
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimentions=(width, height)
    return cv.resize(frame, dimentions, interpolation=cv.INTER_AREA)

def changeResolution(livid, width , height):
    '''Works only for LIVE videos'''
    livid.set(3, width)
    livid.set(4, height)


def main():
    capture = cv.VideoCapture(0)        #read from webcam
    while True:
        isTrue, frame = capture.read()
        frame_resized = rescaleFrame(frame)
        cv.imshow('V', frame)
        cv.imshow('V2', frame_resized)
        if not isTrue or (cv.waitKey(20) & 0xFF == ord('a')):       #press 'a' to exit both
            print("End of video.")
            break
    capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
