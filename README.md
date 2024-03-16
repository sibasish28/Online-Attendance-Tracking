<p>How to run/execute this:</p>
<p>The project consists of 3 programs namely:</p>
<ol>
<li>Add image to images directory</li>
<li>Encodegenerator.py</li>
<li>Main.py</li>
</ol>
<p>All the required files i.e. two batch files for dlib
‘dlib_face_recognition_resnet_model_v1.dat’ and
‘shape_predictor_68_face_landmarks.dat’ should be present in the same
directory.</p>
<p>Steps:
#Add image to images directory</p>
<ol>
<li>Add the data(image) to images directory.</li>
<li>The name of your image will be the username for the person detected.</li>
<li>After the capture is complete the image data will be stored in pickle files
named ‘faces.pkl’(for faces) and ‘names.pkl’(labels for the corresponding
image) and will be stored within the data directory.</li>
</ol>
<p>#Encodegenerator.py:</p>
<p>The model takes every picture, converts it into some numerical encoding, and
stores it in a list and all the labels(names of persons) in another list. In the
prediction phase when we pass a picture of an unknown person the recognition
model converts the unfamiliar person&#39;s image into encoding.</p>
<p>#main.py:</p>
<p>When you run “main.py”, a window frame opens with webcam capture. Your
face will be detected and contained within a red rectangle (dlib working in
backend for face detection).It will stay till the process is completed.</p>
<p>Once the face is being detected it is stored in csv file “face_attendance.csv”. The data will is stored in the
csv file under the ‘Attendance’ directory where attendance will be stored in the
format of [NAME,DATE,TIME].</p>
