import os
import pickle
import numpy as np
import cv2
import face_recognition
import datetime
import csv

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(3, 480)

# Load background image
imgBackground = cv2.imread("/home/swapnil/Documents/Face-recognition-Face-identification-on-linux-main/victor-grabarczyk-N04FIfHhv_k-unsplash.jpg")

# Load and encode data
print("Loading Encode File...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
print("Encode File Loaded...")

# Initialize CSV file
csv_filename = "face_attendance.csv"
csv_header = ["Username", "Date", "Time"]
if not os.path.exists(csv_filename):
    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(csv_header)

# Initialize set to keep track of users already recorded today
recorded_users_today = set()

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            username = studentIds[matchIndex]
            now = datetime.datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            if username not in recorded_users_today:
                print("Face detected in database")
                print(username)
                # Save attendance record to CSV
                attendance_record = [username, date_str, time_str]
                with open(csv_filename, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(attendance_record)
                # Add the username to the set of recorded users for today
                recorded_users_today.add(username)

    imgBackground[320:320 + 480, 190:190 + 640] = img
    cv2.imshow("FACE ATTENDANCE", imgBackground)
    cv2.waitKey(1)
