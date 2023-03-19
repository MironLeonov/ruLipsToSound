import os
import os.path as P
from glob import glob
import cv2
from multiprocessing import Pool
from functools import partial

def extract_frames(video_path, output_dir, width, height):
    save_dir = P.join(output_dir, P.basename(video_path).split('.')[0])
    os.makedirs(save_dir, exist_ok=True)
    video = cv2.VideoCapture(video_path)
    num = 1
    while video.isOpened():
        reg, img = video.read()
        if not reg:
            break
        img = cv2.resize(img, (width, height))
        cv2.imwrite(P.join(save_dir, f"img_{num:05d}.jpg"), img)




if __name__ == '__main__':
    OUTPUT_DIR = 'data/features_data/pics'
    INPUT_DIR = 'data/features_data/video'

    video_paths = glob(P.join(INPUT_DIR, "*.mp4"))

    with Pool(1) as p:
        p.map(partial(extract_frames, output_dir=OUTPUT_DIR,
                    width=96, height=96), video_paths)