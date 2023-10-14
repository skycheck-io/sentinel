import os, sys, uuid
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QCheckBox, QLineEdit, QSizePolicy, QPlainTextEdit, 
                             QLabel, QSlider, QHBoxLayout, QSplitter, QFileDialog, QFrame, QRadioButton, QGroupBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter, QTextCursor
from datetime import datetime
import time
import numpy as np
import threading, queue

class StickyRadioButton(QRadioButton):
    """
    Use this class to create radio buttons that can't be toggled off by clicking on them when they're already selected.
    """
    def mousePressEvent(self, event):
        if self.isChecked():
            return
        super().mousePressEvent(event)


class VideoApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.cap = cv2.VideoCapture(0)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.aspect_ratio = self.width / self.height
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.out = None  
        self.save_path = ''
        self.cameras = {}
        self.autorecord = False
        self.recording = False
        self.show_video = True
        self.no_movement_frame_count = 0  # To track how many frames have passed without movement
        self.no_movement_threshold = 30  # Threshold number of frames to stop recording
        self.default_codec = "FFV1"
        self.fourcc = cv2.VideoWriter_fourcc(*self.default_codec)
        self.codecs = {
            "FFV1": "FFV1 (lossless)",
            "MJPG": "MJPG",
            "XVID": "XVID",
            "MP4V": "MP4V",
            "HFYU": "HFYU (lossless)",
        }
        self.codec_extensions = {
            "FFV1": ".avi",
            "MJPG": ".avi",
            "XVID": ".avi",
            "MP4V": ".mp4",
            "HFYU": ".avi",
        }

        self.fgbg_history = 80
        self.fgbg_var_threshold = 20
        self.fgbg_detect_shadows = False
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=self.fgbg_history, varThreshold=self.fgbg_var_threshold, detectShadows=self.fgbg_detect_shadows)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.bb_sensitivity = 20
        self.bounding_box_buffer = 20

        self.all_camera_resolutions = {
            "QVGA": (320, 240),
            "VGA": (640, 480),
            "SVGA": (800, 600),
            "XGA": (1024, 768),
            "HD": (1280, 720),
            "WXGA": (1366, 768),
            "Full HD": (1920, 1080),
            "2K": (2048, 1080),
            "QHD": (2560, 1440),
            "3K": (3072, 1620),
            "UHD": (3840, 2160),
            "4K": (4096, 2160)
        }
        self.resolution_fps_map = {}
        self.prev_time = time.time()
        self.tz_name = time.tzname[time.daylight]
        self.current_video_name = None

        # self.composite_storage = None

        self.composite_storage = None
        self.frame_count = 0



        self.detect_cameras()
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.initUI()

        self.recordings_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self.worker_function)
        self.worker_thread.start()    


    def worker_function(self):
        time.sleep(1)
        while True:
            # Block until a recording is available for processing
            video_filename = self.recordings_queue.get()

            # Check for termination signal
            if video_filename == "TERMINATE":
                break

            self.processRecordedVideo(video_filename)
            self.recordings_queue.task_done()  # Mark the task as done


    def initUI(self):
        """
        Initialize the UI and start the timer to display frames.
        """

        # Create a timer to update the displayed frame
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.nextFrame)
        self.timer.start(int(1000/90)) # 90 FPS timer (may not actually be 90 FPS due to processing time)

        # # Timer for flashing dot
        # self.dot_timer = QTimer(self)
        # self.dot_timer.timeout.connect(self.toggleDot)
        # self.dot_timer.start(500)  # Flash every half a second

        # Create labels for displaying video frames
        self.label_original = QLabel(self)
        self.label_processed = QLabel(self)

        self.message_log = QPlainTextEdit(self)
        self.message_log.setReadOnly(True)  # Make the message log read-only
        self.message_log.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.message_log.setPlaceholderText('Message log will appear here.')
        self.message_log.hide()


        self.recording_button = QPushButton("RECORDING", self)
        # self.recording_button.setAlignment(Qt.AlignCenter)
        self.recording_button.clicked.connect(self.toggleRecording)  # Connect to slot method
        self.setRecordingStatus(False)  # Initially not recording

        # Set the maximum width based on content
        self.recording_button.setMaximumWidth(self.recording_button.sizeHint().width() + 30)  # +30 for some padding

        # Create a horizontal layout to center the button
        centered_layout = QHBoxLayout()
        centered_layout.addStretch(1)  # Spacer on the left
        centered_layout.addWidget(self.recording_button)
        centered_layout.addStretch(1)  # Spacer on the right

        # Splitter for original and processed frames
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.label_original)
        splitter.addWidget(self.label_processed)

        # File picker button
        self.file_button = QPushButton("Select Save Directory", self)
        self.file_button.clicked.connect(self.pickDirectory)
        self.file_label = QLabel("No file selected", self)

        self.frames_checkbox = QCheckBox("Show Video Frames", self)
        self.frames_checkbox.setChecked(True)
        self.frames_checkbox.stateChanged.connect(self.toggleDisplayMode)
        self.autorecord_checkbox = QCheckBox("Autorecord", self)
        self.timestamp_checkbox = QCheckBox("Show Timestamp", self)
        self.timestamp_checkbox.setChecked(True)
        self.fps_display_checkbox = QCheckBox("Show FPS", self)
        self.bbox_checkbox = QCheckBox("Show Bounding Boxes", self)

        

        # Create a group box for background removal settings
        self.bg_group = QGroupBox("Background Removal Settings")
        self.bg_history = QSlider(Qt.Horizontal, self)
        self.bg_history.setFixedWidth(200)
        self.bg_history.setRange(0, 500)
        self.bg_history.setValue(self.fgbg_history)  # Default value
        self.bg_history.valueChanged.connect(self.updateFgbgHistory)
        self.bg_history.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        self.bg_var_threshold = QSlider(Qt.Horizontal, self)
        self.bg_var_threshold.setFixedWidth(200)     
        self.bg_var_threshold.setRange(0, 100)
        self.bg_var_threshold.setValue(self.fgbg_var_threshold)  # Default value
        self.bg_var_threshold.valueChanged.connect(self.updateFgbgVarThreshold)
        self.bg_var_threshold.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        self.bg_history_edit = QLineEdit(self)
        self.bg_history_edit.setFixedWidth(50)
        self.bg_history_edit.setText(str(self.fgbg_history))
        self.bg_history_edit.textChanged.connect(self.updateFgbgHistoryFromEdit)

        self.bg_var_threshold_edit = QLineEdit(self)
        self.bg_var_threshold_edit.setFixedWidth(50)
        self.bg_var_threshold_edit.setText(str(self.fgbg_var_threshold))
        self.bg_var_threshold_edit.textChanged.connect(self.updateFgbgVarThresholdFromEdit)

        # Processing options
        self.processing_group = QGroupBox("Background Processing Settings")
        self.morph_checkbox = QCheckBox("Apply morphological operations", self)
        self.morph_checkbox.setChecked(True)

        self.bb_sensitivity_slider = QSlider(Qt.Horizontal, self)
        self.bb_sensitivity_slider.setFixedWidth(200)
        self.bb_sensitivity_slider.setRange(0, 500)
        self.bb_sensitivity_slider.setValue(self.bb_sensitivity)  # Default value
        self.bb_sensitivity_slider.valueChanged.connect(self.updateBBSensitivity)
        self.bb_sensitivity_slider.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        self.bb_sensitivity_edit = QLineEdit(self)
        self.bb_sensitivity_edit.setFixedWidth(50)
        self.bb_sensitivity_edit.setText(str(self.bb_sensitivity))
        self.bb_sensitivity_edit.textChanged.connect(self.updateBBSensitivityFromEdit)

        self.bb_size_slider = QSlider(Qt.Horizontal, self)
        self.bb_size_slider.setFixedWidth(200)
        self.bb_size_slider.setRange(0, 50)
        self.bb_size_slider.setValue(self.bounding_box_buffer)  # Default value
        self.bb_size_slider.valueChanged.connect(self.updateBBSize)
        self.bb_size_slider.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

        self.bb_size_edit = QLineEdit(self)
        self.bb_size_edit.setFixedWidth(50)
        self.bb_size_edit.setText(str(self.bounding_box_buffer))
        self.bb_size_edit.textChanged.connect(self.updateBBSizeFromEdit)


        # Create radio buttons for different camera options
        self.camera_radios = []
        self.camera_radio_group = QGroupBox("Select Camera")
        radio_layout = QVBoxLayout()

        self.resolutions_radios = []
        self.resolutions_radio_group = QGroupBox("Select Resolution")
        self.resolutions_radio_group.setLayout(QVBoxLayout())

        # self.fps_radios = []
        # self.fps_radio_group = QGroupBox("Select FPS")
        # self.fps_radio_group.setLayout(QVBoxLayout())

        for port, info in self.cameras.items():
            radio = StickyRadioButton(f"Camera {port}") #: {info['width']}x{info['height']} @ {info['fps']}fps")
            radio.camera_port = port  # Attach the port number to the radio button object
            if port == 0:  # Default to the first detected camera
                radio.setChecked(True)
                self.switchCamera(port)
                
            radio.toggled.connect(self.onCameraRadioToggled)
            radio_layout.addWidget(radio)
            self.camera_radios.append(radio)

        self.camera_radio_group.setLayout(radio_layout)    
        self.onCameraRadioToggled()    

        self.codec_radios = []
        self.codec_radio_group = QGroupBox("Select Codec")
        codec_layout = QVBoxLayout()

        for codec, description in self.codecs.items():
            radio = StickyRadioButton(description)
            radio.codec_value = codec  # Attach the codec value to the radio button object
            if codec == self.default_codec:
                radio.setChecked(True)
                self.fourcc = cv2.VideoWriter_fourcc(*codec)
            radio.toggled.connect(self.onCodecRadioToggled)
            codec_layout.addWidget(radio)
            self.codec_radios.append(radio)

        self.codec_radio_group.setLayout(codec_layout)


        # Adjust file picker layout
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.file_button)
        file_layout.addWidget(self.file_label, 1)  # Using stretch factor to push label to fill extra space

        bg_history_layout = QHBoxLayout()
        bg_history_layout.addWidget(QLabel("History:"), 1)
        bg_history_layout.addWidget(self.bg_history, 5)
        # bg_history_layout.addWidget(self.bg_history_label, 1)
        bg_history_layout.addWidget(self.bg_history_edit, 1)
        bg_history_layout.addStretch(5)  # Spacer on the right        

        bg_var_threshold_layout = QHBoxLayout()
        bg_var_threshold_layout.addWidget(QLabel("Var Threshold:"), 1)  # Give less stretch to label
        bg_var_threshold_layout.addWidget(self.bg_var_threshold, 5)  # Give more stretch to slider
        # bg_var_threshold_layout.addWidget(self.bg_var_threshold_label, 1)  # Give less stretch to value label
        bg_var_threshold_layout.addWidget(self.bg_var_threshold_edit, 1)
        bg_var_threshold_layout.addStretch(5)  # Spacer on the right


        sliders_layout = QVBoxLayout()
        sliders_layout.addLayout(bg_history_layout)
        sliders_layout.addLayout(bg_var_threshold_layout)
        bg_group_layout = QVBoxLayout()
        bg_group_layout.addLayout(sliders_layout)
        self.bg_group.setLayout(bg_group_layout)

        bb_sensitivity_slider_layout = QHBoxLayout()
        bb_sensitivity_slider_layout.addWidget(QLabel("Bounding Box Sensitivity:"), 1)
        bb_sensitivity_slider_layout.addWidget(self.bb_sensitivity_slider, 5)
        bb_sensitivity_slider_layout.addWidget(self.bb_sensitivity_edit, 1)


        bb_size_slider_layout = QHBoxLayout()
        bb_size_slider_layout.addWidget(QLabel("Bounding Box Size:"), 1)
        bb_size_slider_layout.addWidget(self.bb_size_slider, 5)
        bb_size_slider_layout.addWidget(self.bb_size_edit, 1)


        proc_layout = QVBoxLayout()
        proc_layout.addWidget(self.morph_checkbox)
        proc_layout.addLayout(bb_sensitivity_slider_layout)
        proc_layout.addLayout(bb_size_slider_layout)
        bg_group_layout = QVBoxLayout()
        bg_group_layout.addLayout(proc_layout)
        self.processing_group.setLayout(bg_group_layout)


        background_layout = QHBoxLayout()
        background_layout.addWidget(self.bg_group)
        background_layout.addWidget(self.processing_group)


        selector_layout = QHBoxLayout()
        selector_layout.addWidget(self.camera_radio_group)  # Adding the camera radio group
        selector_layout.addWidget(self.codec_radio_group)  # Adding the codec radio group        
        selector_layout.addWidget(self.resolutions_radio_group)  # Adding the resolutions radio group
        # selector_layout.addWidget(self.fps_radio_group)  # Adding the FPS radio group

        # Main Control Layout
        control_layout = QVBoxLayout()
        control_layout.addLayout(file_layout)
        control_layout.addWidget(self.frames_checkbox)
        control_layout.addWidget(self.autorecord_checkbox)
        control_layout.addWidget(self.timestamp_checkbox)
        control_layout.addWidget(self.fps_display_checkbox)
        control_layout.addWidget(self.bbox_checkbox)        
        # control_layout.addWidget(self.bg_group)
        # control_layout.addWidget(self.processing_group)
        control_layout.addLayout(background_layout)
        control_layout.addLayout(selector_layout)

        # Adding spacers for better look
        control_layout.addStretch(1)

        # Setting a minimum size for buttons
        self.file_button.setMinimumSize(150, 40)



        # Main Layout
        layout = QVBoxLayout()
        layout.addLayout(centered_layout)
        layout.addWidget(splitter)
        layout.addLayout(control_layout)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Set the size of the window to be smaller
        self.resize(800, int(800 / self.aspect_ratio))
        self.setFocus()
        self.setWindowTitle('Video Display with PyQt5')
        self.prev_time = time.time()
        self.show()

    def logMessage(self, message):
        # Append a message to the message log
        self.message_log.appendPlainText(message)
        # Scroll to the end to always display the latest messages
        cursor = self.message_log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.message_log.setTextCursor(cursor)

    def toggleDisplayMode(self):
        self.show_video = not self.show_video
        if self.show_video:
            self.label_original.show()
            self.label_processed.show()
            self.message_log.hide()
        else:
            self.label_original.hide()
            self.label_processed.hide()
            self.message_log.show()

    ######## Camera Functions ########
    def detect_cameras(self):
        """
        Test the ports and returns a tuple with the available ports and the ones that are working.
        """
        non_working_ports = []
        dev_port = 0
        while len(non_working_ports) < 6: # if there are more than 5 non working ports stop the testing. 
            camera = cv2.VideoCapture(dev_port)
            if not camera.isOpened():
                non_working_ports.append(dev_port)
            else:
                is_reading, img = camera.read()
                w = int(camera.get(3) + 0.5)
                h = int(camera.get(4) + 0.5)
                fps = camera.get(cv2.CAP_PROP_FPS)
                if is_reading:
                    self.cameras[dev_port] = {"width": w, "height": h, "fps": fps}
            dev_port +=1

      
    def switchCamera(self, port):
        """
        Switch the camera to the given port.
        """
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(port)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.aspect_ratio = self.width / self.height


    def onCameraRadioToggled(self):
        """
        Update the camera based on the selected radio button.
        """
        for radio in self.camera_radios:
            if radio.isChecked():
                self.switchCamera(radio.camera_port)
                max_width = self.cameras[radio.camera_port]['width']
                max_height = self.cameras[radio.camera_port]['height']
                self.populateResolutionSelector(max_width, max_height)
                break

    def get_resolutions(self, max_width, max_height):
        """
        Returns a dictionary with the available resolutions for the given max width and height.
        """    
        return {name: res for name, res in self.all_camera_resolutions.items() if res[0] <= max_width and res[1] <= max_height}


    def onResolutionRadioToggled(self):
        """
        Update the resolution based on the selected radio button.
        """
        for radio in self.resolutions_radios:
            if radio.isChecked():
                width, height = radio.resolution_value
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.width = width
                self.height = height
                self.aspect_ratio = self.width / self.height
                self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                # self.populateFPSSelector(self.resolution_fps_map[radio.resolution_value])
                break


    def populateResolutionSelector(self, max_width, max_height):
        # First, clear out existing radio buttons
        for radio in self.resolutions_radios:
            self.resolutions_radio_group.layout().removeWidget(radio)
            radio.deleteLater()
        self.resolutions_radios = []

        resolutions = self.get_resolutions(max_width, max_height)
        # max_available_fps = 0  # Initialize with a low value
        # resolution_fps_map = {}

        # Determine the max FPS for each resolution
        # for name, res in reversed(list(resolutions.items())):
            
        #     self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        #     self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        #     self.cap.set(cv2.CAP_PROP_FPS, 90.0)
        #     current_fps = self.cap.get(cv2.CAP_PROP_FPS)
        #     # print("Testing resolution:", res, "FPS:", current_fps)
        #     max_available_fps = max(current_fps, max_available_fps)
        # #     resolution_fps_map[res] = max_available_fps

        # self.resolution_fps_map = resolution_fps_map

        # Populate the radio buttons for resolutions
        for name, res in reversed(list(resolutions.items())):
            radio = StickyRadioButton(f"{name}: {res[0]}x{res[1]}")
            radio.resolution_value = res
            radio.toggled.connect(self.onResolutionRadioToggled)
            self.resolutions_radio_group.layout().addWidget(radio)
            self.resolutions_radios.append(radio)

        if self.resolutions_radios:
            for radio in self.resolutions_radios:
                width, height = radio.resolution_value
                self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                if width == self.width and height == self.height:
                    radio.setChecked(True)
                    break
            # self.resolutions_radios[0].setChecked(True)
            # self.populateFPSSelector(resolution_fps_map[self.resolutions_radios[0].resolution_value])
             
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)
        # self.cap.set(cv2.CAP_PROP_FPS, 90)



    def onFPSRadioToggled(self):
        """
        Update the FPS based on the selected radio button.
        """
        for radio in self.fps_radios:
            if radio.isChecked():
                self.cap.set(cv2.CAP_PROP_FPS, radio.fps_value)
                break


    # def get_fps_for_resolution(self, width, height):
    #     self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    #     self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    #     time.sleep(0.5)  # Giving a delay of 0.5 seconds to let the camera adjust
    #     return self.cap.get(cv2.CAP_PROP_FPS)


    # def populateFPSSelector(self, max_fps_for_resolution):
    #     """
    #     Populate the FPS radio group with the available FPS values for the given resolution.
    #     """
    #     # First, clear out existing radio buttons
    #     for radio in self.fps_radios:
    #         self.fps_radio_group.layout().removeWidget(radio)
    #         radio.deleteLater()
    #     self.fps_radios = []

    #     available_fps = [int(max_fps_for_resolution)]
    #     # You can customize this list based on standard FPS values you want to support:
    #     possible_fps_values = [120, 60, 30, 24, 15]

    #     for fps in possible_fps_values:
    #         if fps <= max_fps_for_resolution:
    #             available_fps.append(fps)

    #     for fps in available_fps:
    #         radio = StickyRadioButton(f"{fps} FPS")
    #         radio.fps_value = fps
    #         radio.toggled.connect(self.onFPSRadioToggled)
    #         self.fps_radio_group.layout().addWidget(radio)
    #         self.fps_radios.append(radio)

    #     # Check the highest FPS by default
    #     if self.fps_radios:
    #         self.fps_radios[0].setChecked(True)

    def nextFrame(self):
        """
        Read the next frame from the video and process it.
        """
        ret, original_frame = self.cap.read()
        if not ret:
            return

        # original_frame = frame

        # Apply background subtraction
        fgMask = self.fgbg.apply(original_frame)

        # Remove noise with morphological operations
        if self.morph_checkbox.isChecked():
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, self.kernel, iterations=2)
            fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, self.kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        movement_detected = False
        # Draw bounding boxes
        for contour in contours:
            if cv2.contourArea(contour) < self.bb_sensitivity:  # Filter out small movements
                continue
            if self.bbox_checkbox.isChecked():
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(original_frame, (x - self.bounding_box_buffer, y - self.bounding_box_buffer), (x + w + self.bounding_box_buffer, y + h + self.bounding_box_buffer), (0, 0, 255), 2)
            movement_detected = True

        if self.autorecord_checkbox.isChecked():
            if movement_detected:
                self.no_movement_frame_count = 0  # Reset the count
                if not self.recording:  # Start recording if not already doing so
                    self.toggleRecording()
                    self.recording = True
            else:
                self.no_movement_frame_count += 1  # Increment the count

            # If no movement is detected for a certain number of frames, stop recording
            if self.recording and self.no_movement_frame_count > self.no_movement_threshold:
                self.toggleRecording()
                self.recording = False


        # Calculate FPS
        curr_time = time.time()
        self.fps = 1 / (curr_time - self.prev_time)
        # print(self.fps)
        self.prev_time = curr_time

        if self.timestamp_checkbox.isChecked():
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S ') + self.tz_name
            cv2.putText(original_frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.putText(processed_frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw the FPS if the checkbox is checked
        if self.fps_display_checkbox.isChecked():
            fps = f"FPS: {self.fps:.0f}"
            cv2.putText(original_frame, fps, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            # cv2.putText(processed_frame, fps, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if self.save_path and self.out:
            self.out.write(original_frame)

        if self.frames_checkbox.isChecked():
            # Set the frames to the labels
            # self.message_log.hide()
            # self.label_original.show()
            # self.label_processed.show()            
            self.updateLabelWithFrame(self.label_original, original_frame)
            self.updateLabelWithFrame(self.label_processed, fgMask)
        # else:
        #     # Hide video frames and display messages
        #     self.label_original.hide()
        #     self.label_processed.hide()
        #     self.message_log.show()
        #     # Process and log messages here
        #     # self.logMessage('Processing saved videos...')

    def updateLabelWithFrame(self, label, frame):
        """
        Update the given label with the given frame.
        """
        if len(frame.shape) == 2:  # Grayscale image
            height, width = frame.shape
            bytesPerLine = width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qImg)
            label_width = label.width()
            label_height = int(label_width / self.aspect_ratio)
            label.setPixmap(pixmap.scaled(label_width, label_height))
        else:  # RGB image
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_BGR888) # BGR888 because OpenCV uses BGR format
            pixmap = QPixmap.fromImage(qImg)
            label_width = label.width()
            label_height = int(label_width / self.aspect_ratio)
            # painter = QPainter(pixmap)
            # painter.setBrush(QColor('red'))
            # painter.drawEllipse(10, 10, 20, 20)        
            label.setPixmap(pixmap.scaled(label_width, label_height))

    
    ##### Recording functionalities #####
    def setRecordingStatus(self, is_recording):
        """
        Update the recording status and the appearance of the recording button.
        """
        if is_recording:
            # Red color for recording
            self.recording_button.setText("RECORDING")
            self.recording_button.setStyleSheet("""
                QPushButton {
                    color: red;
                    font-weight: bold;
                    font-size: 24px;
                    border: 2px solid;
                    border-radius: 10px;
                    padding: 5px 5px;
                }
                QPushButton:hover {
                    color: red;
                    background-color: lightgray;
                }
            """)
            self.recording = True
        else:
            # Gray color when not recording
            self.recording_button.setText("RECORD")
            self.recording_button.setStyleSheet("""
                QPushButton {
                    color: gray;
                    font-weight: bold;
                    font-size: 24px;
                    border: 2px solid;
                    border-radius: 10px;
                    padding: 5px 10px;
                }
                QPushButton:hover {
                    color: gray;
                    background-color: lightgray;
                }
            """)            
            self.recording = False


    def toggleRecording(self):
        if not self.recording:                   
            if not self.save_path:  # If directory not chosen
                self.pickDirectory()  # This method will handle starting the recording
            if not self.save_path:
                return
            # Generate random UUID for the video name
            self.current_video_name = str(uuid.uuid4())
            codec_extension = self.codec_extensions[self.default_codec]
            self.output_filename = f"{self.current_video_name}{codec_extension}"
            self.video_path = os.path.join(self.save_path, self.output_filename)
            self.out = cv2.VideoWriter(self.video_path, self.fourcc, self.fps, (int(self.width), int(self.height)))
            self.setRecordingStatus(True)
        else:
            self.setRecordingStatus(False)
            if self.out:
                self.out.release()
                self.video_path = os.path.join(self.save_path, self.output_filename)
                self.recordings_queue.put(self.video_path)                
                self.out = None  # Reset the video writer

            # if self.composite_storage is not None:
            #     normalized_composite = cv2.convertScaleAbs(self.composite_storage)
            #     composite_filename = os.path.join(self.save_path, f"{self.current_video_name}_composite.png")
            #     cv2.imwrite(composite_filename, normalized_composite)

            #     # Reset the composite storage and frame count
            #     self.composite_storage = None
            #     self.frame_count = 0

    def processRecordedVideo(self, video_filename):
        print("Processing video:", video_filename)
        # def manual_count(handler):
        #     frames = 0
        #     while True:
        #         status, frame = handler.read()
        #         if not status:
        #             break
        #         frames += 1
        #     return frames 

        cap = cv2.VideoCapture(video_filename)
        # ret, frame = self.cap.read()
        while not cap.isOpened():
            time.sleep(0.5)

        # cap = cv2.VideoCapture(video_filename)
        # Determine the total frame count
        # total_frames = manual_count(cap)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # if total_frames == 0:
        #     total_frames = manual_count(cap) # attempt manual counting
        # print("Total frames:", total_frames)
        weight = 1.0 / total_frames
        
        # cap = cv2.VideoCapture(video_filename)
        ret, first_frame = cap.read()
        composite_storage = np.zeros_like(first_frame, dtype='float')

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            fgMask = self.fgbg.apply(frame)
            foreground = cv2.bitwise_and(frame, frame, mask=fgMask)
            # cv2.accumulateWeighted(foreground, composite_storage, weight)
            composite_storage = np.maximum(composite_storage, foreground)


        composite_image = composite_storage.astype(np.uint8)
        new_filename = os.path.splitext(video_filename)[0] + '_composite.png'
        cv2.imwrite(new_filename, composite_image)
        cap.release()
        

    def pickDirectory(self):
        """
        Open a directory picker dialog to select the directory to save videos.
        """
        directory = QFileDialog.getExistingDirectory(self, "Select Directory to Save Videos")
        if directory:
            self.save_path = directory
            self.file_label.setText(self.save_path)



    # Background subtraction control
    def updateFgbgHistory(self, value):
        """
        Update the history value for the background subtractor.
        """
        self.fgbg_history = value
        self.fgbg.setHistory(self.fgbg_history)
        # self.bg_history_label.setText(str(self.fgbg_history))  # Update the QLabel text
        self.bg_history_edit.setText(str(self.fgbg_history))

    def updateFgbgHistoryFromEdit(self):
        try:
            value = int(self.bg_history_edit.text())
            self.bg_history.setValue(value)
            self.updateFgbgHistory(value)
        except ValueError:
            pass  # User might have entered a non-integer value, we simply ignore it


    def updateFgbgVarThreshold(self, value):
        """
        Update the var threshold value for the background subtractor.
        """
        self.fgbg_var_threshold = value
        self.fgbg.setVarThreshold(self.fgbg_var_threshold)
        # self.bg_var_threshold_label.setText(str(self.fgbg_var_threshold))  # Update the QLabel text
        self.bg_var_threshold_edit.setText(str(self.fgbg_var_threshold))

    def updateFgbgVarThresholdFromEdit(self):
        try:
            value = int(self.bg_var_threshold_edit.text())
            self.bg_var_threshold.setValue(value)
            self.updateFgbgVarThreshold(value)
        except ValueError:
            pass  # User might have entered a non-integer value, we simply ignore it


    def updateBBSensitivity(self, value):
        """
        Update the history value for the background subtractor.
        """
        self.bb_sensitivity = value
        self.bb_sensitivity_edit.setText(str(self.bb_sensitivity))

    def updateBBSensitivityFromEdit(self):
        try:
            value = int(self.bb_sensitivity_edit.text())
            self.bb_sensitivity_slider.setValue(value)
            self.updateBBSensitivity(value)
        except ValueError:
            pass  # User might have entered a non-integer value, we simply ignore it


    def updateBBSize(self, value):
        """
        Update the history value for the background subtractor.
        """
        self.bounding_box_buffer = value
        self.bb_size_edit.setText(str(self.bounding_box_buffer))

    def updateBBSizeFromEdit(self):
        try:
            value = int(self.bb_size_edit.text())
            self.bb_size_slider.setValue(value)
            self.updateBBSize(value)
        except ValueError:
            pass  # User might have entered a non-integer value, we simply ignore it




    # Codec selection
    def onCodecRadioToggled(self):
        """
        Update the codec value based on the selected radio button.
        """
        for radio in self.codec_radios:
            if radio.isChecked():
                self.fourcc = cv2.VideoWriter_fourcc(*radio.codec_value)
                self.default_codec = radio.codec_value
                break



    def mousePressEvent(self, event):
        """ Re-implement mousePressEvent to defocus QLineEdit widgets. """
        self.setFocus()
        super().mousePressEvent(event)


    def closeEvent(self, event):
        self.setRecordingStatus(False)
        self.recordings_queue.put("TERMINATE")
        self.worker_thread.join()        
        if self.cap:    
            self.cap.release()
        if self.out:
            self.out.release()        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoApp()
    sys.exit(app.exec_())
