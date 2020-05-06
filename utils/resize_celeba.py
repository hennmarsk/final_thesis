import cv2
import os
from resize_image import resize


def _resize_image(size):
    f = os.listdir("./data/img_align_celeba")
    for src in f:
        print(src)
        img = cv2.imread(f"./data/img_align_celeba/{src}")
        img = resize(img, size)
        cv2.imwrite(f"./data/img_align_celeba_{size}/{src}", img)


def _resize_lfw(size):
    f = os.listdir("./data/lfw")
    for folder in f:
        imgs = os.listdir(f"./data/lfw/{folder}")
        for src in imgs:
            img = cv2.imread(f"./data/lfw/{folder}/{src}")
            img = resize(img, size)
            try:
                os.makedirs(f"./data/lfw_{size}/{folder}")
            except OSError:
                pass
            cv2.imwrite(f"./data/lfw_{size}/{folder}/{src}", img)


_resize_lfw(112)
# img = cv2.imread("./data/img_align_celeba/051522.jpg")
# rs = resize(img, 112)
# cv2.imshow("test", rs)
# cv2.waitKey(0)
