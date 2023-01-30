import os
from sklearn.model_selection import train_test_split
import random

path_images = '/home/neptun/PycharmProjects/datasets/data-science-bowl-2018/stage1_train'

all_images = os.listdir(path_images)

train, val = train_test_split(all_images, test_size=10)

zero = random.sample(train, k=3)

with open('/home/neptun/PycharmProjects/datasets/data-science-bowl-2018/al/train.txt', 'w') as f:
    for name in zero:
        f.write(name+'\n')

with open('/home/neptun/PycharmProjects/datasets/data-science-bowl-2018/al/val.txt', 'w') as f:
    for name in val:
        f.write(name+'\n')
