# Face Detection web service using Computer Vision

### Aim : To highlight the faces in provided image. This service can be accessed based on web-based API.

##### Tools used: OpenCV, Flask, Python, HaarCascade face detector.

### Usage:
#### Step 1: Start the flask server
1. conda install opencv numpy urllib3 requests
2. git clone https://github.com/shabeer/face_detection.git
3. cd face_detection
4. export FLASK_APP=face_detection.py 
5. python -m flask run

#### Step 2: Invoke the face detection API endpoint using url. It returns an html with faces highlighted in input image.
http://localhost:5000/detect_faces/?image_url=URL

For example:
http://localhost:5000/detect_faces/?image_url=https://upload.wikimedia.org/wikipedia/commons/a/a4/Chaplin_The_Immigrant.jpg

![Identified faces are highlight with green rectangle](../blob/master/charlie_chaplin.jpg?raw=true)

image_url starting with file:// are not allowed to avoid security concerns.
