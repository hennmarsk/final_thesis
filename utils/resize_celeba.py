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


def _resize(fl, size):
    f = os.listdir(f"./data/{fl}")
    for folder in f:
        imgs = os.listdir(f"./data/{fl}/{folder}")
        for src in imgs:
            img = cv2.imread(f"./data/{fl}/{folder}/{src}")
            img = resize(img, size)
            try:
                os.makedirs(f"./data/{fl}_{size}/{folder}")
            except OSError:
                pass
            cv2.imwrite(f"./data/{fl}_{size}/{folder}/{src}", img)


_resize('casia', 96)
print('done')
_resize('lfw', 96)
print('done')
