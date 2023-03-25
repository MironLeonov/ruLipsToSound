import os
import os.path as P
from glob import glob
from multiprocessing import Pool
from functools import partial

import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def extract_frames(video_path, output_dir, glob_width, glob_height):
    save_dir = P.join(output_dir, P.basename(video_path).split('.')[0])
    os.makedirs(save_dir, exist_ok=True)
    video = cv2.VideoCapture(video_path)
    num = 1
    while video.isOpened():
        reg, img = video.read()
        if not reg:
            break
        
        with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            h, w, c = img.shape

            if not results.detections:
                print(num)
                continue

            for detection in results.detections: 
                face_coords = detection.location_data.relative_bounding_box
                x = int(face_coords.xmin*w)
                y = int(face_coords.ymin*h)
                width = int(face_coords.width*w)
                height = int(face_coords.height*h)

                croped_img = img[y:y+height, x:x+width]

                # break

                res_img = cv2.resize(croped_img, (glob_width, glob_height))
                cv2.imwrite(P.join(save_dir, f"img_{num:05d}.jpg"), res_img)
        num += 1




if __name__ == '__main__':
    OUTPUT_DIR = 'data/features_data/pics'
    INPUT_DIR = 'data/features_data/video'

    video_paths = glob(P.join(INPUT_DIR, "*.mp4"))

    with Pool(1) as p:
        p.map(partial(extract_frames, output_dir=OUTPUT_DIR,
                    glob_width=96, glob_height=96), video_paths)