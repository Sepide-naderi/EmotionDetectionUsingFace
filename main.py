# face emotion detection in live camera

#import packages
import cv2
from deepface import DeepFace

#HaarCascade_frontalface detection Algorithm
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#define a video Capture obj
cap = cv2.VideoCapture(0)

while True:
    #capture the vd frame
    ret, frame = cap.read()
    result = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False)

    #convert to grayscale of eachframes
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #returns the coordinates of detected faces in x,y,w,h format
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    #Draw rectangle around the detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    #converting list to str
    emotion = result["dominant_emotion"]
    txt = str(emotion)

    #Display the resulting frame and also flip frame
    frame = cv2.flip(frame, flipCode=1)
    cv2.putText(frame, txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 4)
    cv2.imshow('WebCam', frame)

    #the "e" btn is set as the quitting btn
    if cv2.waitKey(1) & 0xff == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()