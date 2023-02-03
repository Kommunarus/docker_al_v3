import os
import random
import matplotlib.pyplot as plt
import PIL.Image as Image

from scripts.segmentation.train_unet import find_err
from scripts.segmentation.train_unet import train_al, eval

def save_id(listi, path, prefix):
    with open(os.path.join(path, 'train', 'step_{}.txt'.format(prefix)), 'w') as f:
        for row in listi:
            f.write(row[0]+'\n')


def plot_img(path, listik, score, pred_mask):
    fig, axs = plt.subplots(2, len(listik), figsize=(15, 5))
    fig.suptitle(score)
    for i in range(len(listik)):
        id = listik[i]
        img = Image.open(os.path.join(path, id, 'images', id+'.png'))
        axs[0, i].imshow(img)
        axs[1, i].imshow(pred_mask[i])

    plt.show()


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
    for num in [3, ]:
    # for num in range(3, 28, 3):
        meanscore = []
        for n in range(10):
            sampl = random.sample(all_id, k=num)

            old_files = os.listdir(os.path.join(path2, 'train'))
            for file in old_files:
                os.remove(os.path.join(path2, 'train', file))

            with open(os.path.join(path2, 'train', 'zero.txt'), 'w') as f:
                for name in sampl:
                    f.write(name + '\n')

            model, score = train_al(path1, path2, n_gpu=1)
            meanscore.append(score)
            pred_mask = eval(model, path1, sampl, n_gpu=1)
            plot_img(path1, sampl, score, pred_mask)
        print(num, meanscore)

