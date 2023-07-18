import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# getting_the_image_from_the_file_and_loop_the_image
path = 'Atnimages'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


# getting_encodings_from_the_images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# attendance_register_in_cvs
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
# print(len(encodeListKnown))
print('Encoding Complete')

# videocapture
cap = cv2.VideoCapture(2)

while True:
    success, img = cap.read()

    facesCurFrame = face_recognition.face_locations(img)
    encodeCurFrame = face_recognition.face_encodings(img, facesCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            y1, x2, y2, x1 = faceLoc
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

# # attendance_register_in_cvs
# def markAttendance(name):
#     with open('Attendance.csv','r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#         if name not in nameList:
#             now = datetime.now()
#             dtString = now.strftime('%H:%M:%S')
#             f.writelines(f'\n{name},{dtString}')
#
# encodeListKnown = findEncodings(Images)
# print('Encoding Complete')
#
# #videocapture
# cap = cv2.VideoCapture('http://10.193.0.223:8080/video')
# success, Img = cap.read()
# print(success)
#
# while success is True:
#     Img = cap.read()
#     new_dimension = (416, 520)
#     # ImgS = cv2.resize(Img, new_dimension, 0.5, 0.5, interpolation=cv2.INTER_AREA)
#     ImgS = cv2.cvtColor(Img, cv2.COLOR_BGR2RGB)
#
#     facesCurFrame = face_recognition.face_locations(ImgS)
#     encodeCurFrame = face_recognition.face_encodings(ImgS,facesCurFrame)
#
#     for encodeFace, faceLoc in zip(encodeCurFrame,facesCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
#         #print(faceDis)
#         matchIndex = np.argmin(faceDis)
#
#         name = classNames[matchIndex].upper()
#
#         if matches[matchIndex]:
#             #print(name)
#             y1,x2,y2,x1 = faceLoc
#             y1, x2, y2,x1 = y1*4,x2*4,y2*4,x1*4
#             cv2.rectangle(Img,(x1,y1),(x2,y2),(0,255,0),2)
#             cv2.rectangle(Img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
#             cv2.putText(Img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
#             markAttendance(name)
#
#     cv2.imshow('Webcam', ImgS)
#
#     cv2.waitKey(1)
