import os
import random

from scripts.segmentation.train_unet import find_err
from scripts.segmentation.train_unet import train_al

def save_id(listi, path, prefix):
    with open(os.path.join(path, 'train', 'step_{}.txt'.format(prefix)), 'w') as f:
        for row in listi:
            f.write(row[0]+'\n')



if __name__ == '__main__':
    path1 = '/home/alex/PycharmProjects/dataset/data-science-bowl-2018/stage1_train'
    path2 = '/home/alex/PycharmProjects/dataset/data-science-bowl-2018/al'
    # N = 3

    all_id = os.listdir(path1)

    val_img = []

    for file in os.listdir(os.path.join(path2, 'val')):
        with open(os.path.join(path2, 'val', file)) as f:
            for line in f.readlines():
                val_img.append(line.strip())

    all_id = list(set(all_id) - set(val_img))
    for num in [50, 100, 200, 300, 400]:
    # for num in range(3, 28, 3):
        meanscore = []
        for n in range(5):
            sampl = random.sample(all_id, k=num)

            old_files = os.listdir(os.path.join(path2, 'train'))
            for file in old_files:
                os.remove(os.path.join(path2, 'train', file))

            with open(os.path.join(path2, 'train', 'zero.txt'), 'w') as f:
                for name in sampl:
                    f.write(name + '\n')

            _, score = train_al(path1, path2, n_gpu=1)
            meanscore.append(score)
        print(num, meanscore)

