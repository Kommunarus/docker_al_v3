import os
from sklearn.model_selection import train_test_split
import random

path_images = '/home/alex/PycharmProjects/dataset/data-science-bowl-2018/stage1_train'

all_images = os.listdir(path_images)

train, val = train_test_split(all_images, test_size=100)

zero = random.sample(train, k=100)

with open('/home/alex/PycharmProjects/dataset/data-science-bowl-2018/al2/train/zero.txt', 'w') as f:
    for name in zero:
        f.write(name+'\n')

# with open('/home/alex/PycharmProjects/dataset/data-science-bowl-2018/al2/val/val.txt', 'w') as f:
#     for name in val:
#         f.write(name+'\n')
