size = 4
import cv2, sys, numpy, os

# change the paths below to the location where these files are on your machine
haar_file = '/path/to/project/directory/haarcascade_frontalface_default.xml'
# path to the main faces directory which contains all the sub_datasets
datasets = '/path/to/project/directory/faces'

# Create a list of images and a list of corresponding names along with a unique id
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
	
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
(width, height) = (130, 100)


def rounded_rectangle(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2   
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
 
  
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

(images, labels) = [numpy.array(lists) for lists in [images, labels]]


model = cv2.face.LBPHFaceRecognizer_create()
# Train the model on images
model.train(images, labels)

face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)
print('Classifier trained!')
print('Attempting to recognize faces...')
while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # detect faces using the haar_cacade file
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
       
        face = gray[y:y + h, x:x + w]
        face_resize = cv2.resize(face, (width, height))
      
        prediction = model.predict(face_resize)
        rounded_rectangle(im, (x, y), (x + w, y + h), (0, 255, 255), 2, 15, 30)
       
        if prediction[1] < 74:
            cv2.putText(im,'%s' % (names[prediction[0]].strip()),(x + 5, (y + 25) + h), cv2.FONT_HERSHEY_PLAIN,1.5,(20,185,20), 2)
            # print the confidence level with the person's name to standard output
            confidence = (prediction[1]) if prediction[1] <= 100.0 else 100.0
            print("predicted person: {}, confidence: {}%".format(names[prediction[0]].strip(), round((confidence / 74.5) * 100, 2)))
        
        else:
            cv2.putText(im,'Unknown',(x + 5, (y + 25) + h), cv2.FONT_HERSHEY_PLAIN,1.5,(65,65, 255), 2)
            print("predicted person: Unknown")

    cv2.imshow('OpenCV Face Recognition -  esc to close', im)
    key = cv2.waitKey(10)

    if key == 27:
        break
