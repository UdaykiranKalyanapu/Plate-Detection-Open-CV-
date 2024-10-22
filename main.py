from ultralytics import YOLO
import cv2

import base
from sort.sort import *
from base import find_car, LicensePlate, filecreator



results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
YoloLicencePlate = YOLO('./license_plate_detector.pt')

# load video
capture = cv2.VideoCapture('./singlecar.mp4')

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = capture.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        DetectedPlates = YoloLicencePlate(frame)[0]
        for license_plate in DetectedPlates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, carId = find_car(license_plate, track_ids)

            if carId != -1:

                # crop license plate
                PlateTrim = frame[int(y1):int(y2), int(x1): int(x2), :]

                # process license plate
                plate_gray = cv2.cvtColor(PlateTrim, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                plateText, plateScore = LicensePlate(license_plate_crop_thresh)

                if plateText is not None:
                    results[frame_nmr][carId] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': plateText,
                                                                    'bbox_score': score,
                                                                    'text_score': plateScore}}

# write results
filecreator(results, './singlecar.csv')