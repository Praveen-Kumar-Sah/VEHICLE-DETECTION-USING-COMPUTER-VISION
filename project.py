# import the libraries
import cv2 as cv
import numpy as np

# read the data
cap = cv.VideoCapture('video.mp4')

# initialize Substract 
algo = cv.BackgroundSubtractorMOG2()  # use for detect only vehicle no other thrings

while True:
    ret,frame = cap.read() 
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(3,3),5)
    # applying on each frame
    img_sub = algo.apply(blur)
    dilat = cv.dilate(img_sub,np.ones(5,5))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    dilatada = cv.morphologyEx(dilat,cv.MORPH_CLOSE,kernel)
    dilatada = cv.morphologyEx(dilatada,cv.MORPH_CLOSE,kernel)
    counter = cv.findContours(dilatada,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    
    
    cv.imshow('DETECTER',dilatada)
    
    if cv.waitKey(1) == 13:
        break
        
cv.destroyAllWindows()
cap.release()
