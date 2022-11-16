import cv2
import numpy as np
import sqlite3
import csv
import os
import sys
# from PILL import Image
header = ["ID","Name"]
def writeToCSV(id,name):
    f = open("people.csv","a+", newline="")
    writer = csv.writer(f)
    tup=[id,name]
    writer.writerow(tup)
    f.close()
   
# def insertOrUpdate(id,name):
#     conn= sqlite3.connect('data.db')
#     query= "SELECT * FROM people WHERE ID=" + str(id)
#     cusror= conn.execute(query)
#     isRecordExit =0
#     for row in cusror:
#         isRecordExit = 1
#     if (isRecordExit==0):
#         query= "INSERT INTO people(ID,Name) VALUES(" + str(id) + ", '"+str(name)+"')"
#     else:
#         query = "UPDATE people SET name= '" + str(name) + "'WHERE ID = " + str(id)
#     conn.execute(query)
#     conn.commit()
#     conn.close()
    # insertOrUpdate(10,"ABC")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap =cv2.VideoCapture(0)
id = input("Enter Your ID: ")
name = input("Input Your Name: ")
# insertOrUpdate(id,name)
writeToCSV(id,name)
sampleNum = 0
while(True):
    ret,frame =cap.read()
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if not os.path.exists('dataSet'):
            os.makedirs('dataSet')
        sampleNum +=1
        cv2.imwrite('dataSet/User.' + str(id)+'.' + str(sampleNum) + '.jpg', gray[y:y+h,x:x+w])
    cv2.imshow('frame',frame)
    cv2.waitKey(1) #& 0xFF == ord('q')):
    if(sampleNum>=100):
            break
cap.release()
cv2.destroyAllWindows