import numpy as np
import cv2

cam = cv2.VideoCapture(0)
cascadePath = "/Users/adarsh/Desktop/Works/haarcascade_frontalface_default.xml"

faceCascade = cv2.CascadeClassifier(cascadePath)

adarsh = np.load('Adarsh.npy').reshape((20,50*50*3))
shaa = np.load('sharukh.npy').reshape((20,50*50*3))
print(adarsh.shape)
print(shaa.shape)

names = { 0: 'Adarsh', 1: 'Sharukh'}

label = np.zeros((40,1))
label[21:] = 1

data = np.concatenate([adarsh,shaa])

def distance(x1,x2):
    return np.sqrt(((x1-x2)**2).sum())

def knn(x, train, targets, k=7):
    m = train.shape[0]
    dist = []
    for i in range(m):
        dist.append(distance(x,train[i]))
    dist = np.asarray(dist)
    indx = np.argsort(dist)
    sorted_label = label[indx][:k]
    counts  = np.unique(sorted_label,return_counts=True)
    return counts[0][np.argmax(counts[1])]

while True:
	# get each frame
	ret, frame = cam.read()

	if ret == True:
		# convert to grayscale and get faces
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(gray, 1.3, 5)

		# for each face
		for (x, y, w, h) in faces:
			face_component = frame[y:y+h, x:x+w,:]
			face = cv2.resize(face_component, (50, 50))
            #grayFace = cv2.cvtColor(face,cv2.COLOR_RGB2GRAY)
			# after processing the image and rescaling
			# convert to linear vector using .flatten()
			# and pass to knn function along with all the data

			lab = knn(face.flatten(), data, label)
			# convert this label to int and get the corresponding name
			text = names[int(lab)]

			# display the name
			cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

			# draw a rectangle over the face
			cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
		cv2.imshow('face recognition', frame)

		if cv2.waitKey(1) == 27:
			break
	else:
		print('Error')

cv2.destroyAllWindows()
