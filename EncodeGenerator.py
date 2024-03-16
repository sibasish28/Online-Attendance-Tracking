import cv2
import face_recognition
import pickle
import os

# Import img
folderPath = '/home/swapnil/Documents/Face-recognition-Face-identification-on-linux-main/Images'
pathList = os.listdir(folderPath)

#print(pathList)
imgList = []
studentIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    studentIds.append(os.path.splitext(path)[0])

#print(imgList)
#print(len(imgList))
print(studentIds)

def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if len(encodings) > 0:
            encode = encodings[0]
            encodeList.append(encode)
            print(encode)

    return encodeList

print("Encoding Started ...")
encodeListKnown = findEncodings(imgList)
#encodeListKnown = findEncodings(imgList)
print(encodeListKnown)
encodeListKnownWithIds = [encodeListKnown, studentIds]

print("Encoding Complete!!!")

#print(encodeListKnown)

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")
