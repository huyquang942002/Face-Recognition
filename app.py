from models import *
from flask import Flask, render_template, Response, request
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
import datetime
import time

app = Flask(__name__)

var_list = []


def capture_by_frames():
    global camera
    camera = cv2.VideoCapture(0)
    sampleNum = 0
    d = var_list.pop()
    name = str(d[2])
    Id = str(d[0])

    while True:
        success, frame = camera.read()  # read the camera frame

        detector = cv2.CascadeClassifier(
            'Haarcascades/haarcascade_frontalface_default.xml')
        faces = detector.detectMultiScale(frame, 1.3, 5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            x1, y1 = x+w, y+h
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.line(frame, (x, y), (x+30, y), (255, 0, 255), 6)  # Top Left
            cv2.line(frame, (x, y), (x, y+30), (255, 0, 255), 6)

            cv2.line(frame, (x1, y), (x1-30, y), (255, 0, 255), 6)  # Top Right
            cv2.line(frame, (x1, y), (x1, y+30), (255, 0, 255), 6)

            cv2.line(frame, (x, y1), (x+30, y1),
                     (255, 0, 255), 6)  # Bottom Left
            cv2.line(frame, (x, y1), (x, y1-30), (255, 0, 255), 6)

            cv2.line(frame, (x1, y1), (x1-30, y1),
                     (255, 0, 255), 6)  # Bottom right
            cv2.line(frame, (x1, y1), (x1, y1-30), (255, 0, 255), 6)
            sampleNum = sampleNum+1
            cv2.imwrite("Image/ "+name + "." + Id + '.' + str(sampleNum) +
                        ".jpg", gray[y:y+h, x:x+w])  # luu anh train vao folderframe
            #gray[y:y+h, x:x+w]
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif sampleNum > 100:  # luu anh cho den khi dc 100 anh
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # print(arrList)
    writeCSV(d)

def getProfileById(id):
    url = open("data/student.csv", "r")
    profile = None
    read_file = csv.reader(url)
    for row in read_file:
        if(row[0] == str(id)):
            profile = row
    url.close()
    return profile

def detect_capture_by_frames():
    global camera
    camera = cv2.VideoCapture(0)
    # sampleNum = 0
    # d = var_list.pop()
    # name = str(d[2])
    # Id = str(d[0])
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    detector = cv2.CascadeClassifier(
        'Haarcascades/haarcascade_frontalface_default.xml')
    df = pd.read_csv("data/student.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    while True:
        success, frame = camera.read()  # read the camera frame

        faces = detector.detectMultiScale(frame, 1.3, 5)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            x1, y1 = x+w, y+h
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.line(frame, (x, y), (x+30, y), (255, 0, 255), 6)  # Top Left
            cv2.line(frame, (x, y), (x, y+30), (255, 0, 255), 6)

            cv2.line(frame, (x1, y), (x1-30, y), (255, 0, 255), 6)  # Top Right
            cv2.line(frame, (x1, y), (x1, y+30), (255, 0, 255), 6)

            cv2.line(frame, (x, y1), (x+30, y1),
                     (255, 0, 255), 6)  # Bottom Left
            cv2.line(frame, (x, y1), (x, y1-30), (255, 0, 255), 6)

            cv2.line(frame, (x1, y1), (x1-30, y1),
                     (255, 0, 255), 6)  # Bottom right
            cv2.line(frame, (x1, y1), (x1, y1-30), (255, 0, 255), 6)
            Id, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if (conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(
                    ts).strftime('%H:%M:%S')
                profile = getProfileById(Id)
                
                if(profile != None):
                    cv2.putText(frame, profile[1] + " " + profile[2], (x, y+h),
                                font, 1, (255, 255, 255), 2)
            else:
                Id = 'Unknown'
                cv2.putText(frame, str(Id), (x, y+h),
                            font, 1, (255, 255, 255), 2)

                #tt = str(Id)
            # if (conf > 75):
            #     noOfFile = len(os.listdir("ImagesUnknown"))+1
            #     cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) +
            #                 ".jpg", frame[y:y+h, x:x+w])
            # cv2.putText(frame, str(Id), (x, y+h), font, 1, (255, 255, 255), 2)
            # attendance = attendance.drop_duplicates(
            #     subset=['Id'], keep='first')
            #cv2.imshow('im', frame)

        #     sampleNum = sampleNum+1
        #     cv2.imwrite("Image/ "+name + "." + Id + '.' + str(sampleNum) +
        #                 ".jpg", gray[y:y+h, x:x+w])  # luu anh train vao folder
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # elif sampleNum > 100:  # luu anh cho den khi dc 100 anh
        #     break
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # print(arrList)
    # writeCSV(d)


@app.route("/takePhotos", methods=['POST'])
def takePhotos():
    # Moving forward code
    number = request.form.get('number')
    firstName = request.form.get('firstName')

    lastName = request.form.get('lastName')
    email = request.form.get('email')
    phone = request.form.get('phone')
    form_class = request.form.get('class')
    # print(form_title, form_number)
    arrList = [number, firstName, lastName, email, phone, form_class]
    # print(arrList)
    # takePhotos(arrList)
    var_list.append(arrList)
    return render_template('index.html')


@app.route('/detectStudent')
def detectStudent():
    return render_template('detectStudent.html')


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/addNewStudent')
def addNewStudent():
    return render_template('addNewStudent.html')


@app.route('/start', methods=['POST'])
def start():
    return render_template('addNewStudent.html')


@app.route("/studentList/")
def studentList():
    result = readCSV()
    return render_template('studentList.html', data=result)


@app.route('/stop', methods=['POST'])
def stop():
    if camera.isOpened():
        camera.release()
    return render_template('stop.html')


@app.route('/stopDetect', methods=['POST'])
def stopDetect():
    if camera.isOpened():
        camera.release()
    return render_template('home.html')


@app.route('/video_capture')
def video_capture():

    return Response(capture_by_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect_video_capture')
def detect_video_capture():
    return Response(detect_capture_by_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/TrainImage', methods=['POST'])
def TrainImage():
    faces, Id = getImagesAndLabels('image')
    # row = request.form.get('number')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(
        'Haarcascades/haarcascade_frontalface_default.xml')

    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel/Trainner.yml")
    return studentList()


def getImagesAndLabels(path):
    # # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]

    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids


if __name__ == '__main__':
    app.run(debug=True)
