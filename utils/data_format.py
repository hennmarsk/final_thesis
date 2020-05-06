import os
import numpy as np

eval_array = np.zeros(202599)
eval_list = open("/home/tung/final-thesis/data/list_eval_partition.txt")
lines = eval_list.readlines()
for ind, line in enumerate(lines):
    parsed = line.split()
    eval_array[ind] = parsed[1]

celeb_list = open("/home/tung/final-thesis/data/identity_CelebA.txt")
lines = celeb_list.readlines()
for ind, line in enumerate(lines):
    parsed = line.split()
    eval_name = ''
    if eval_array[ind] == 0:
        eval_name = 'train'
    elif eval_array[ind] == 1:
        eval_name = 'validate'
    else:
        eval_name = 'test'
    if not os.path.exists(
            f"/home/tung/final-thesis/data/{eval_name}/{parsed[1]}"):
        os.makedirs(f"/home/tung/final-thesis/data/{eval_name}/{parsed[1]}")
    size = len(os.listdir(
        f"/home/tung/final-thesis/data/{eval_name}/{parsed[1]}"))
    os.rename(f"/home/tung/final-thesis/data/img_align_celeba/{parsed[0]}",
              f"/home/tung/final-thesis/data/{eval_name}/{parsed[1]}/{size}")
    print(ind)
