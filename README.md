# Face-Recoginition
This is a simple web application that can: 

* Show a live webcam feed with real-time face recognition.

* Recognize people who have already registered.

* Let new users register by uploading a photo.

* Save user data (images and face encodings) in folders using their names.
  
🧠 What It Does

Live Camera Feed:

* Uses your webcam to detect faces.

* Displays names if the person is already registered.

Register with Image:

* You can upload a photo with your name.

* The app saves the photo and face data in a folder.

Face Matching:

* Compares faces from the camera or uploaded image with saved data.


🚀 How to Run

Install the requirements:
pip install flask flask-cors opencv-python face_recognition numpy

Run the app:
python app.py


📡 Endpoints

* video_feed – Shows live webcam with face recognition

* register – Register a user with name and photo (base64 image)

* health – Check if the server is running

✅ Notes

One face per photo is supported during registration.

No database – all data is saved in folders.

Works on most computers with a webcam.
