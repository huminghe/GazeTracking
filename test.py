"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--dataset_folder', default='./face_pics/', type=str, help='dataset path')
parser.add_argument('--save_folder', default='./result2/', type=str,
                    help='Dir to save txt results')
args = parser.parse_args()

gaze = GazeTracking()

if __name__ == '__main__':

    data_folder = args.dataset_folder
    listdir = os.listdir(data_folder)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    for img_name in tqdm(listdir):
        try:
            image_path = os.path.join(data_folder, img_name)
            img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)
            gaze.refresh(img_raw)
            frame = gaze.annotated_frame()
            text = ""

            if gaze.is_blinking():
                text = "Blinking"
            elif gaze.is_right():
                text = "Looking right"
            elif gaze.is_left():
                text = "Looking left"
            elif gaze.is_center():
                text = "Looking center"

            cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
            left_pupil = gaze.pupil_left_coords()
            right_pupil = gaze.pupil_right_coords()
            cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                        (147, 58, 31), 1)
            cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                        (147, 58, 31), 1)
            name = os.path.join(args.save_folder, img_name)
            # if left_pupil != None and right_pupil != None:
            cv2.imwrite(name, frame)

        except Exception as e:
            print(e)
