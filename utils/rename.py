import os


def rename(path):
    folders = os.listdir(path)
    x = 0
    for folder in folders:
        x += len(os.listdir(f"{path}/{folder}"))
    return x


print(rename("/home/tung/final-thesis/data/validate"))
