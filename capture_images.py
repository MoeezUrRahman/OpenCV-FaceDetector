import cv2, numpy, sys, os, time
# change path accordingly
haar_file = '/path/to/project/directory/haarcascade_frontalface_default.xml'
# Data folder/Images
datasets = '/path/to/project/directory/faces'
# Each individual will have unique folder name
sub_dataset = 'Random'

# join the paths to include the sub_dataset folder
path = os.path.join(datasets, sub_dataset)

if not os.path.isdir(path):
    os.mkdir(path)


(width, height) = (130, 100)

face_cascade = cv2.CascadeClassifier(haar_file)

webcam = cv2.VideoCapture(0)

print("Webcam is open? ", webcam.isOpened())

time.sleep(2)

count = 1
print("Taking pictures...")

while count < 101:
    # im = camera stream
    ret_val, im = webcam.read()
    # if it recieves something from the webcam...
    if ret_val == True:
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # detect face using the haar cascade file
        faces = face_cascade.detectMultiScale(gray, 1.3, 4)
        for (x,y,w,h) in faces:
            # draws a rectangle around your face when taking pictures
            # this is to create a ROI (region of interest) so it only takes pictures of your face
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            # define 'face' as the inside of the rectangle we made above and make it grayscale
            face = gray[y:y + h, x:x + w]
            # resize the face images to the size of the 'face' variable above (i.e: area captured inside of the rectangle)
            face_resize = cv2.resize(face, (width, height))
            # save images with their corresponding number
            cv2.imwrite('%s/%s.png' % (path,count), face_resize)
        count += 1
        # display the openCV window
        cv2.imshow('OpenCV', im)
        key = cv2.waitKey(20)
        # press esc to stop the loop
        if key == 27:
            break
print("Sub dataset for your face has been created.")
webcam.release()
cv2.destroyAllWindows()
