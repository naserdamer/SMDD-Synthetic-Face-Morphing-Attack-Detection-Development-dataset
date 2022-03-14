import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse
import math

from facenet_pytorch import MTCNN

mtcnn = MTCNN(select_largest=True, post_process=False, min_face_size=64,  device='cuda:0')
OutImageSize = 224  #aligned and saved image size

def align_images(in_folder, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    skipped_imgs = []

    img_names = os.listdir(in_folder)
    for img_name in tqdm(img_names):
        filepath = os.path.join(in_folder, img_name)
        img = cv2.imread(filepath)
        if img is None:
            continue
        boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

        if landmarks is None:
            skipped_imgs.append(img_name)
            continue

        facial5points = landmarks[0]

        x1, y1, x2, y2 = boxes[0]
        w = x2 - x1
        h = y2 - y1
        ew = (OutImageSize - w) / 2
        eh = (OutImageSize - h) / 2

        x1, y1 = round(x1 - ew), round(y1 - eh)
        x2, y2 = x1 + n, y1 + n
        crop_img = img[y1:y2, x1:x2]

        try:
            cv2.imwrite(os.path.join(out_folder, img_name), crop_img)
        except:
            skipped_imgs.append(img_name)
            pass

    print(skipped_imgs)
    print(f"Images with no Face: {len(skipped_imgs)}")

def main():
    parser = argparse.ArgumentParser(description='MTCNN alignment')
    parser.add_argument('--in_folder', type=str, default="./out_large", help='folder with images')
    parser.add_argument('--out_folder', type=str, default="./out_aligned", help="folder to save aligned images")

    args = parser.parse_args()
    align_images(args.in_folder, args.out_folder)

if __name__ == "__main__":
    main()
