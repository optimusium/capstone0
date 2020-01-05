import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import numpy as np
from matplotlib import pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def grayplt(img,title=''):
    '''
    plt.axis('off')
    if np.size(img.shape) == 3:
        plt.imshow(img[:,:,0],cmap='gray',vmin=0,vmax=1)
    else:
        plt.imshow(img,cmap='gray',vmin=0,vmax=1)
    plt.title(title, fontproperties=prop)
    '''
    
    fig,ax = plt.subplots(1)
    ax.set_aspect('equal')
    


    # Show the image
    if np.size(img.shape) == 3:
        ax.imshow(img[:,:,0],cmap='hot',vmin=0,vmax=255)
    else:
        ax.imshow(img,cmap='hot',vmin=0,vmax=255)
   
    plt.show()


cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0

prev=0
while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()
    #cv2.imwrite("frame.jpg",frame)
    #sleep(0.25)

    movement=0
    
    '''
    if prev==0:
        #prev_hist=histr
        prev_frame=frame
        prev=1
        sleep=(0.25)
        continue
    else:
        color = ('b','g','r')
        histr=[]
        for i,col in enumerate(color):
            histr = moving_average( cv2.calcHist([frame-prev_frame],[i],None,[256],[0,256]) )
            #print(histr)
            #histr2 = moving_average( cv2.calcHist([prev_frame],[i],None,[256],[0,256]) )
            #print(histr2)     
            #print(histr-histr2)
            #raise
            #histr3 = histr-histr2 #moving_average( cv2.calcHist([frame-prev_frame],[i],None,[256],[0,256]) )
            #histr3/=histr+1
            #print(histr3)
            #raise
            #plt.plot(histr,color = col)
            #plt.plot(histr2,color = col)
            plt.plot(histr,color = col)
            plt.xlim([0,256])
            plt.show()

        #plt.plot(histr,color = col)
        #plt.plot(histr2,color = col)
        #plt.plot(histr3,color = col)
        #plt.plot(prev_hist,color = col)
        #plt.plot(histr-prev_hist,color = col)
        #plt.xlim([0,256])
        #plt.show()
        #prev_hist=histr
        prev_frame=frame
        sleep=(1)
    '''
    '''    
    negative=frame-prev_frame
    negative=np.where(negative>240,0,negative)
    negative=np.where(negative<15,0,negative)
    '''
    
    
    #grayplt(negative)
    #grayplt(frame)
    #grayplt(prev_frame)
    prev_frame=frame
    #print( np.size(negative))
    #print( np.sum(negative>120)  )
    #sleep(1)
    #print(frame)
    #print(prev_frame)
    #raise

    # Convert BGR to HSV
    '''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imwrite("frame5.jpg",res)
    '''
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if y-h>0 and x-w>0:
            cv2.imwrite("frame.jpg",frame[y:y+h,x:x+w])
        elif y-h>0:
            cv2.imwrite("frame.jpg",frame[y:y+h,x:x+w])
        elif x-w>0:
            cv2.imwrite("frame.jpg",frame[y:y+h,x:x+w])
        else:
            cv2.imwrite("frame.jpg",frame[y:y+h,x:x+w])
            
        #cv2.imwrite("frame.jpg",frame)
        resized = cv2.resize(frame[y:y+h,x:x+w], (160,160), interpolation = cv2.INTER_AREA)
        '''
        if y-h>0 and x-w>0:
            resized = cv2.resize(frame[y:y+h,x:x+w], (160,160), interpolation = cv2.INTER_AREA)
        elif y-h>0:
            resized = cv2.resize(frame[y:y+h,x:x+w], (200,200), interpolation = cv2.INTER_AREA)
        elif x-w>0:
            resized = cv2.resize(frame[y:y+h,x:x+w], (200,200), interpolation = cv2.INTER_AREA)
        else:
            resized = cv2.resize(frame[y:y+h,x:x+w], (200,200), interpolation = cv2.INTER_AREA)
        '''
        
        #resized = cv2.resize(frame[y-h:y+h,x-w:x+w], (200,200), interpolation = cv2.INTER_AREA)
        cv2.imwrite("frame2.jpg",resized)
        print("R")
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        #sleep(0.5)
        
        #raise
        

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        #ret, frame = video_capture.read()
        #cv2.imwrite("frame98.jpg",resized)        
        #image = load_img("frame98.jpg")
        #image = img_to_array(image)
        #image = np.expand_dims(image, axis=0)
        '''
        aug=ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,horizontal_flip=True,fill_mode="nearest")
        print("[INFO] generating images...")
        imageGen = aug.flow(image, batch_size=1, save_to_dir=".",save_prefix="image1", save_format="jpg")
        i=0
        for image in imageGen:
            print(image)
            i+=1
            if i==100: break
        
        break
        '''
        cv2.imwrite("frame5.jpg",resized)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    print("P")

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
