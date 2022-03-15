import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

from facenet_pytorch import MTCNN

mtcnn = MTCNN(select_largest=True, post_process=False, min_face_size=64,  device='cuda:0')

def align_images(in_folder, out_folder):
    print('Imges are in :', in_folder)
    os.makedirs(out_folder, exist_ok=True)
    skipped_imgs = []

    img_names = os.listdir(in_folder)
    for img_name in tqdm(img_names):
        #if not (".png" in img_name or ".jpeg" in img_name or ".jpg" in img_name):
        #    continue
        filepath = os.path.join(in_folder, img_name)
        img = cv2.imread(filepath)
        if img is None:
            continue
        boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)

        if landmarks is None:
            skipped_imgs.append(img_name)
            continue

        facial5points = landmarks[0]

        n = 0.05 # add margin for cropping
        x1, y1, x2, y2 = boxes[0]
        w = x2 - x1
        h = y2 - y1
        x1 = x1 - n*w
        y1 = y1 - n*h
        x2 = x2 + n*w
        y2 = y2 + n*h
        crop_img = img[round(y1):round(y2), round(x1):round(x2)]

        try:
            cv2.imwrite(os.path.join(out_folder, img_name), crop_img)
        except:
            skipped_imgs.append(img_name)
            pass

    print(skipped_imgs)
    print(f"Images with no Face: {len(skipped_imgs)}")


def main():
    parser = argparse.ArgumentParser(description='Face detection and crop')
    parser.add_argument('--in_folder', type=str, default="./out_large", help='folder with images')
    parser.add_argument('--out_folder', type=str, default="./out_aligned", help="folder to save aligned images")

    args = parser.parse_args()
    align_images(args.in_folder, args.out_folder)


if __name__ == "__main__":
    main()
