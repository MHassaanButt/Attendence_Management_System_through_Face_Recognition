import csv
import os
import sys
import cv2
import dlib
import time
import numpy as np
import pandas as pd
import threading

from datetime import datetime

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import pyqtSlot, QRunnable, pyqtSignal, QObject, QThreadPool
from PyQt5.QtGui import QImage
from keras.models import load_model
from GUI import Ui_MainWindow
from VGG_Utils import VGG_Utils
from Warning import Ui_Dialog
from keras.models import load_model

thread_break = False


class Signals(QObject):
    return_signal = pyqtSignal(str)


class Thread(QRunnable):
    signal = pyqtSignal(str)

    def __init__(self):
        super(Thread, self).__init__()
        self.signal = Signals()

    @pyqtSlot()
    def run(self):
        while True:
            if thread_break:
                break
            time.sleep(1)
            result = "Graphs/training_plot.jpg"
            self.signal.return_signal.emit(result)


# The Main class
class Main:

    # The Constructor
    def __init__(self):

        # Global Variables
        self.images = []
        self.images_folder = "Captured images"
        self.camera_id = 0

        # thread pool
        self.threadpool = QThreadPool()

        # labels for model training
        self.classes = []

        # loading the camera
        self.video = cv2.VideoCapture(self.camera_id)

        # loading the face detector
        self.detector = dlib.get_frontal_face_detector()

        # count of the frames
        self.count = 0

        # default image size
        self.image_size = 224

        # camera stream control flag
        self.camera_stream = False

        # Flag to capture face or not
        self.capture_flag = False

        # Output dataframe time
        if os.path.exists("Records/report.csv"):
            self.report = pd.read_csv('Records/report.csv')
        else:
            self.report = pd.DataFrame(columns=['ID', 'Name', 'Date', 'Time'])

        # Sample Amount to take
        self.samples_amount = 100

        # For model workings
        self.model = VGG_Utils()
        self.learning_rate = 0.1
        self.epochs = 100
        self.batch_size = 20
        self.test_size = 0.5

        # Creating the main window object
        self.main_window = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self.main_window)

        # For warning dialogue
        self.warning_window = QtWidgets.QDialog()
        self.warning_obj = Ui_Dialog()
        self.warning_obj.setupUi(self.warning_window)
        self.warning_obj.pushButton.clicked.connect(
            lambda: self.warning_window.close())

        # All button connections
        self.ui.pushButton.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_2))
        self.ui.pushButton_2.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_3))
        self.ui.pushButton_3.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page_4))
        self.ui.pushButton_10.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page))
        self.ui.pushButton_11.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page))
        self.ui.pushButton_12.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page))
        self.ui.pushButton_13.clicked.connect(
            lambda: self.ui.stackedWidget.setCurrentWidget(self.ui.page))

        # Connecting buttons to functions
        self.ui.pushButton_5.clicked.connect(self.capture_flag_fun)
        self.ui.pushButton_6.clicked.connect(self.capture_faces)
        self.ui.pushButton_7.clicked.connect(self.start_training_thread)
        self.ui.pushButton_8.clicked.connect(self.start_tracking_on_camera)
        self.ui.pushButton_4.clicked.connect(self.view_Report)

        # Start threading of training
        self.graph_thread = Thread()
        self.training_thread = threading.Thread(target=self.start_training)
        self.graph_thread.signal.return_signal.connect(self.update_Graph)

        # Start with initial page
        self.ui.stackedWidget.setCurrentWidget(self.ui.page)

    def capture_flag_fun(self):
        if self.ui.pushButton_5.text() == "Start Capture Face":
            if self.ui.pushButton_6.text() == "Start Camera":
                self.warning_obj.label_3.setText(
                    "Please start the camera first")
                self.warning_window.show()
                return
            self.capture_flag = True
            self.ui.pushButton_5.setText("Stop Capture Face")
        else:
            self.capture_flag = False
            self.ui.pushButton_5.setText("Start Capture Face")

    def capture_faces(self):

        if self.ui.lineEdit.text() == "":
            self.warning_obj.label_3.setText("Please enter person name")
            self.warning_window.show()
            return

        if self.ui.pushButton_6.text() == "Start Camera":

            self.video = cv2.VideoCapture(self.camera_id)
            # Set the button text to stop camera
            self.ui.pushButton_6.setText("Stop Camera")

            # Set the camera stream flag
            self.camera_stream = True

            # Updating the current epochs value
            self.samples_amount = int(self.ui.comboBox_5.currentText())

            # Resetting count
            self.count = 0

            # Loop for the camera frames
            while True:
                # Get the frame from camera
                ret, frame = self.video.read()

                # Resizing the input frame
                frame = cv2.resize(frame, (500, 400))

                # if frame is read
                if ret:
                    # convert the frame into grayscale and get the face Coordinates
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Getting the face coordinates from the image
                    faces = self.detector(gray)

                    # Check face or not found
                    if len(faces) != 0:

                        # Now get all the face in the frame
                        for face in faces:
                            x1 = face.left()
                            y1 = face.top()
                            x2 = face.right()
                            y2 = face.bottom()

                            # Now get the reign of interest of the face and get the prediction over that face
                            roi = frame[y1:y2, x1:x2]

                            # Images resizing
                            try:
                                cv2.resize(
                                    roi, (self.image_size, self.image_size))
                            except:
                                continue

                            if self.capture_flag:
                                # Getting the images
                                self.images.append(roi)

                                # Increment the count
                                self.count += 1

                            # Draw a green box over the face
                            cv2.rectangle(frame, (x1, y1),
                                          (x2, y2), (0, 255, 0), 2)

                    # Show the count of faces Captured
                    cv2.putText(frame, "Face Captured:" + str(self.count), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2)

                    # Show the image
                    result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, channel = result.shape
                    step = channel * width
                    qImg = QImage(result.data, width, height,
                                  step, QImage.Format_RGB888)
                    self.ui.label_3.setPixmap(QtGui.QPixmap(qImg))

                    cv2.waitKey(1)

                    # Break the loop for the 100th frame
                    if self.count == self.samples_amount:
                        break

                    # break if the camera stream is null
                    if not self.camera_stream:
                        break

        elif self.ui.pushButton_6.text() == "Stop Camera":
            self.camera_stream = False
            self.ui.pushButton_6.setText("Start Camera")
            return

        # Release the camera and destroy the windows
        self.video.release()
        self.reset_windows()

        if self.images != []:
            # Extract the person name
            person_name = self.ui.lineEdit.text()

            # Create the id named folder for the image
            if not os.path.exists(os.path.join(self.images_folder, person_name)):
                os.mkdir(os.path.join(self.images_folder, person_name))

            # Save the face images
            for c, img in enumerate(self.images):
                cv2.imwrite(os.path.join(self.images_folder, person_name,
                            person_name + '.' + str(c) + '.jpg'), img)

            # Empty the image list
            self.images = []

        # Now the image capturing is off
        self.capture_flag = False

        # Set the GUI text to back one
        self.ui.pushButton_5.setText("Start Capture Face")
        self.ui.pushButton_6.setText("Start Camera")

    def reset_windows(self):
        self.ui.label_3.setStyleSheet("QFrame{\n"
                                      "background:none;\n"
                                      "background-color: rgb(255, 255, 255);\n"
                                      "border:2px solid #4161AD;\n"
                                      "border-radius:5px\n"
                                      "}\n"
                                      "")
        self.ui.label_5.setStyleSheet("QFrame{\n"
                                      "background:none;\n"
                                      "background-color: rgb(255, 255, 255);\n"
                                      "border:2px solid #4161AD;\n"
                                      "border-radius:5px\n"
                                      "}\n"
                                      "")
        self.ui.label_7.setStyleSheet("QFrame{\n"
                                      "background:none;\n"
                                      "background-color: rgb(255, 255, 255);\n"
                                      "border:2px solid #4161AD;\n"
                                      "border-radius:5px\n"
                                      "}\n"
                                      "")
        _translate = QtCore.QCoreApplication.translate
        self.ui.label_3.setText(_translate("MainWindow",
                                           "<html><head/><body><p><span style=\" font-size:14pt; "
                                           "font-weight:600;\">Image Frame</span></p></body></html>"))
        self.ui.label_5.setText(_translate("MainWindow",
                                           "<html><head/><body><p><span style=\" font-size:14pt; "
                                           "font-weight:600;\">Training Frame</span></p></body></html>"))
        self.ui.label_7.setText(_translate("MainWindow",
                                           "<html><head/><body><p><span style=\" font-size:14pt; "
                                           "font-weight:600;\">Tracking Frame</span></p></body></html>"))

    def update_Graph(self):
        self.ui.label_5.setPixmap(QtGui.QPixmap("Graphs/training_plot.jpg"))

    def start_training_thread(self):
        self.threadpool.start(self.graph_thread)
        self.training_thread.start()

    def start_training(self):

        # check for the person quantity
        if len(os.listdir(self.images_folder)) < 2:
            self.warning_obj.label_3.setText(
                "Atleast two subjects are required.....")
            self.warning_window.show()
            return

        if self.ui.pushButton_7.text() == "Start Training":

            # Disable the start training button
            self.ui.pushButton_7.setText("Training.....")
            self.ui.pushButton_7.setEnabled(False)

            name_dict = {}

            # Loading the images from the folder
            for n, img_folders in enumerate(os.listdir(self.images_folder)):
                name_dict[str(n)] = img_folders
                for img_path in os.listdir(os.path.join(self.images_folder, img_folders)):
                    # Load the image
                    image = cv2.imread(os.path.join(
                        self.images_folder, img_folders, img_path))
                    # Resizing the image
                    image = cv2.resize(
                        image, (self.image_size, self.image_size))
                    # Normalizing
                    image = cv2.normalize(
                        image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    # Saving the image
                    self.images.append(np.array(image))
                    # Saving the label
                    self.classes.append(n)

            # Check if the model exists or not then train the model


# model = load_model(path_to_model)
            if self.model.load_model("Model/vgg16_model.h5.h5") == 0:
                # if self.load_model("Model/vgg16_model.h5.h5") == 0:
                self.warning_obj.label_3.setText("No Trained Model Found...")

            # Get the training tuning values
            self.epochs = self.ui.comboBox.currentText()
            self.learning_rate = self.ui.comboBox_2.currentText()
            self.batch_size = self.ui.comboBox_3.currentText()
            self.test_size = self.ui.comboBox_4.currentText()

            # Inputting the model credentials and starting the training
            self.model.start_training(
                images=self.images,
                classes=self.classes,
                epochs=int(self.epochs),
                learning_rate=float(self.learning_rate),
                batch_size=int(self.batch_size),
                test_size=float(self.test_size)
            )

            # Break the thread
            global thread_break
            thread_break = True

            # Enable the training Button
            self.ui.pushButton_7.setEnabled(True)

            # Change the text
            self.ui.pushButton_7.setText("Start Training")

            # Resetting the training graph
            if os.path.exists("Graphs/new.jpg"):
                img = cv2.imread("Graphs/new.jpg")
                cv2.imwrite("Graphs/training_plot.jpg", img)

            # Save the dictionary as a dataframe
            result = pd.DataFrame(name_dict, index=[0])
            result.to_csv("Records/names.csv", index=False)
            print("training complete")

    def start_tracking_on_camera(self):

        # Current tracking report Dataframe
        current_report = pd.DataFrame(columns=['ID', 'Name', 'Date', 'Time'])

        if self.ui.pushButton_8.text() == "Start Tracking":

            # Change the camera
            self.ui.pushButton_8.setText("Stop Tracking")

            # Camera stream Flag on
            self.camera_stream = True

            # Video capture from the camera
            self.video = cv2.VideoCapture(self.camera_id)

            # count for the unknown person
            unknown_count = 0
            unknown_appear = 0

            # Loading the model trained before
            self.model = load_model("Model/vgg16_model.h5.h5")

            # To calculate the frame rate
            fps_start_time = datetime.now()
            total_frames = 0

            while True:

                # Get the frame from camera
                ret, frame = self.video.read()

                tracking_thresh = float(self.ui.comboBox_6.currentText())

                # Load the names
                names_dict = pd.read_csv("names.csv").values[0]

                # To calculate the fps rate in the video from camera
                total_frames = total_frames + 1
                fps_end_time = datetime.now()
                time_diff = fps_end_time - fps_start_time
                if time_diff.seconds == 0:
                    fps = 0.0
                else:
                    fps = (total_frames / time_diff.seconds)

                fps_text = "FPS: {:.2f}".format(fps)

                # if frame is read
                if ret:
                    # convert the frame into grayscale and get the face Coordinates
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.detector(gray)

                    # Check face or not found
                    if len(faces) != 0:

                        # Now get all the face in the frame
                        for face in faces:
                            x1 = face.left()
                            y1 = face.top()
                            x2 = face.right()
                            y2 = face.bottom()

                            # Now get the reign of interest of the face and get the prediction over that face
                            roi = frame[y1:y2, x1:x2]

                            # Images resizing and preprocessing
                            try:
                                roi = cv2.resize(
                                    roi, (self.image_size, self.image_size))
                                roi = cv2.normalize(roi, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                                    dtype=cv2.CV_32F)
                                roi = np.array(roi)
                            except:
                                continue

                            predictions = self.model.predict(np.array([roi]))
                            score = max(predictions[0])
                            pred = int(np.argmax(predictions))
                            print(predictions)

                            if score >= tracking_thresh:

                                # Draw a green box over the face
                                cv2.rectangle(frame, (x1, y1),
                                              (x2, y2), (0, 255, 0), 2)
                                cv2.rectangle(
                                    frame, (x2 - (x2-x1), y2), (x2, y2 + 50), (0, 255, 0), -1)

                                # Display the id text
                                cv2.putText(frame, str(names_dict[pred]), (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (255, 255, 255), 2)

                                # Save the date and time of the person seen at last
                                ts = time.time()
                                date = datetime.fromtimestamp(
                                    ts).strftime('%Y-%m-%d')
                                timeStamp = datetime.fromtimestamp(
                                    ts).strftime('%H:%M:%S')
                                current_report.loc[len(current_report)] = [
                                    pred, names_dict[pred], date, timeStamp]

                                # Reset the unknown count
                                unknown_count = 0
                            else:
                                # Draw a green box over the face
                                cv2.rectangle(frame, (x1, y1),
                                              (x2, y2), (0, 0, 255), 2)
                                cv2.rectangle(
                                    frame, (x2 - (x2-x1), y2), (x2, y2 + 50), (0, 0, 255), -1)

                                # Display the id text
                                cv2.putText(frame, "unknown" + str(unknown_count * 10), (x1, y2 + 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                                # if unknown person counter 10 times
                                if unknown_count == 10:
                                    ts = time.time()
                                    date = datetime.fromtimestamp(
                                        ts).strftime('%Y-%m-%d')
                                    timeStamp = datetime.fromtimestamp(
                                        ts).strftime('%H:%M:%S')
                                    current_report.loc[len(current_report)] = ['-', 'unknown ' + str(unknown_appear),
                                                                               date,
                                                                               timeStamp]
                                    unknown_appear += 1
                                    unknown_count = 0

                                # increment the unknown person count
                                unknown_count += 1

                            current_report = current_report.drop_duplicates(
                                subset=['ID'], keep='first')

                    # Write the count of the images
                    cv2.putText(frame, fps_text, (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # Show the image
                    result = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, channel = result.shape
                    step = channel * width
                    qImg = QImage(result.data, width, height,
                                  step, QImage.Format_RGB888)
                    self.ui.label_7.setPixmap(QtGui.QPixmap(qImg))

                    cv2.waitKey(1)

                    # Camera stream check
                    if not self.camera_stream:
                        break
        else:
            self.ui.pushButton_8.setText("Start Tracking")
            self.camera_stream = False

        # Save the report
        self.report = pd.concat([self.report, current_report], axis=0)
        self.report.to_csv('Records/report.csv', index=False)

        # Release the camera and destroy the windows
        self.video.release()

        # resetting the windowa
        self.reset_windows()

        # Reset the button text previous
        self.ui.pushButton_8.setText("Start Tracking")

    def view_Report(self):
        if not os.path.exists("Records/report.csv"):
            return

        # Empty the table first
        for i in range(self.ui.table.rowCount()):
            self.ui.table.removeRow(i)

        # Load and add the csv file data into table widget
        with open("Records/report.csv") as f:
            file_data = []
            row = csv.reader(f)

            for x in row:
                file_data.append(x)

            self.ui.table.setRowCount(0)
            file_data = iter(file_data)
            next(file_data)

            for row, rd in enumerate(file_data):
                self.ui.table.insertRow(row)
                for col, data in enumerate(rd):
                    self.ui.table.setItem(
                        row, col, QtWidgets.QTableWidgetItem(str(data)))

        # Navigate to the Last Seen Log Window
        self.ui.stackedWidget.setCurrentWidget(self.ui.page_5)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = Main()
    main.main_window.show()
    sys.exit(app.exec_())
