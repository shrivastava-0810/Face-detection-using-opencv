#Face detection in a image
import cv2
face_cascade = cv2.CascadeClassifier("D:/Machine Learning A-Z New/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/opencv\haarcascade_frontalface_default.xml")
img = cv2.imread("D:/Machine Learning A-Z New/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/opencv/face.jpg")
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors = 5)
for x, y, w, h in faces:
    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
cv2.imshow("Gray", img)
cv2.waitKey(0)
cv2.destroyAllWindows()



#Face detection using camera
import cv2
video = cv2.VideoCapture(0)
a = 1
while True:
    a += 1
    check, frame = video.read()
    face_cascade = cv2.CascadeClassifier("D:/Machine Learning A-Z New/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/opencv/haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.05, minNeighbors = 5)
    for x, y, w, h in faces:
        img = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    frame = cv2.flip(frame,1)
    cv2.imshow("Capturing", frame)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
print(a)
video.release()
cv2.destroyAllWindows()