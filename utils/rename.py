import os

def rename(path):
    folders = os.listdir(path)
    l = 0
    for folder in folders:
        l += len(os.listdir(f"{path}/{folder}"))
    return l
print(rename("/home/tung/final-thesis/data/validate"))
