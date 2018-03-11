from flask import Flask,request,send_file,render_template
import cv2
import urllib
import numpy as np
import os

app = Flask(__name__)

# Pre-trained face detector - provided as haar cascade by opencv
face_detector_cascade = "static/haarcascade_frontalface_default.xml"

# result image is image with rectangles marking faces identified.

# directory to store result image.
local_file_storage="/tmp/"

# url prefix for client to pull result image
result_image_url_prefix="/detect_faces/result/"

# Support both GET and POST.
@app.route("/detect_faces/", methods =["GET","POST"])
def detect_faces_api():
    # A template used to return html with placeholders being replaced dynamically
    error_result = render_template("image_with_faces_identified.html", image_filename=None)

    image_url = None

    # Extract image url from post and get
    if request.method == "POST":
        image_url = request.form["image_url"]
    if request.method == "GET":
        image_url = request.args.get("image_url")

    image_url = image_url.strip("\"")

    # Block passing local files path, to avoid security breach.
    if image_url is None or image_url.startswith("file:/"):
        return error_result

    # Get filename stripping the directories
    image_name = image_url.split("/")[-1]

    try:
        response = urllib.request.urlopen(image_url)
    except Exception:
        # On exception, return error
        return error_result

    image_data = response.read()

    # Convert image data as numpy array.
    image = np.asarray(bytearray(image_data), dtype='uint8')

    # Decode numpy array as colored image.
    image_colored = cv2.imdecode(image, cv2.IMREAD_COLOR)

    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Convert image from color to gray scale
    # If gray scale image is passed, its equivalently a no-op.
    # For face-detection image color is not required. Grayscale image suffices.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize cascade face detector
    detector_value = cv2.CascadeClassifier(face_detector_cascade)

    # Detect faces
    values = detector_value.detectMultiScale(image,
                                             scaleFactor=1.1,
                                             minNeighbors=5,
                                             minSize=(30, 30),
                                             flags=cv2.CASCADE_SCALE_IMAGE)

    # Construct rectangles for the faces detected.
    faces_coordinates = [(int(a), int(b), int(a + c), int(b + d)) for (a, b, c, d) in values]
    for (w, x, y, z) in faces_coordinates:
        cv2.rectangle(image_colored,
                      (w, x), # Top Left corner point
                      (y, z), # Bottom right corner point
                      (0, 255, 0), # rectangle is of Green color
                      2)

    # Store result image in local storage.
    faces_identified_image_filename = local_file_storage + image_name
    cv2.imwrite(faces_identified_image_filename, image_colored)

    # Return a html with dynamically replacing placeholders of result image location.
    return render_template("image_with_faces_identified.html",
                           image_filename = result_image_url_prefix + image_name,
                           random_number = np.random.randint(10000,99999))

#This route is used to pull image in above html
@app.route(result_image_url_prefix+'<name>')
def image_file(name):
    file_to_serve = local_file_storage + name
    response = send_file(file_to_serve)
    # After serving, delete the result image to avoid increase of storage space
    os.remove(file_to_serve)
    return response

