from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np


detector = MTCNN()


def resize(img, size):
    result = detector.detect_faces(img)
    if len(result) == 0:
        rs = cv2.resize(img, (size, size))
        return rs
    else:
        m = 0
        for person in result:
            box = person['box']
            if box[2] + box[3] > m:
                chosen = box
                m = box[2] + box[3]
        box = chosen
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        crop = img[np.max([y, 0]):np.min([y+h, 250]),
                   np.max([x, 0]):np.min([x+w, 250])]
        rs = cv2.resize(crop, (size, size))
        return rs
