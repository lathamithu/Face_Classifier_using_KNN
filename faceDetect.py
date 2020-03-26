import numpy as np
import cv2

#lets define a video acquition object
name = input("Please Enter the Persons Name: ")
cap = cv2.VideoCapture(0)

cascadePath = "/Users/adarsh/Desktop/Works/haarcascade_frontalface_default.xml"

#create the face cascade classifier using the xml file
faceCascade = cv2.CascadeClassifier(cascadePath)
faceData = []
captureCount = 0
faceCount = 0
while True:
    ret,frame = cap.read()
    if ret is True:
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray= cv2.equalizeHist(gray)

        # The face or faces in an image are detected
        # This section requires the most adjustments to get accuracy on face being detected.
        #the first parameter is the image,then the scale factor and the minimum number of neighbors
        faces = faceCascade.detectMultiScale(gray,1.5,5)
        for (x,y,w,h) in faces:
            #Here we are drawing rectangle over the faces detected using x,y co-ordinates and w,h height and width in blue color
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

            #extract the face out of the frame and resize it
            face = frame[x:x+w,y:y+w,:]
            #grayFace = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            resizedFace = cv2.resize(face,(50,50))
            
            #Skipping consecutive faces images
            captureCount+=1
            if captureCount/20 == 1:
                faceData.append(resizedFace)
                captureCount=0
                faceCount+=1
            #Displaying the number of captured faces Count
            cv2.putText(frame,"captured Faces = "+str(faceCount),(10,20),cv2.FONT_HERSHEY_COMPLEX,1,(72, 52, 212),2)
            cv2.putText(frame,"Collecting "+str(name)+"'s Data",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(72, 52, 212),2)
        if cv2.waitKey(1) == 27 or faceCount >=20:
            cap.release()
            cv2.destroyAllWindows()
            break
        cv2.imshow('img',frame)
#Saving the data in to npy file
data = np.array(faceData)
np.save(name,data)
