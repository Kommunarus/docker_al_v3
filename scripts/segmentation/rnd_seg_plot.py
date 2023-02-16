import os
import random
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
import torch
from torchvision.utils import draw_segmentation_masks
from scripts.segmentation.train_rcnn import train_al, eval

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

def al(out):
    t = torch.nonzero(out['masks']).shape[0]
    t1 = (out['masks'][:, 0] > 0.5).sum()
    c = 100 * (t - t1) / t / len(out['masks'])
    print('scores', [round(x, 3) for x in out['scores'].cpu().numpy().tolist()])
    a = []
    nump = out['masks'][:, 0].cpu().numpy()
    for j in range(10):
        mask = (nump > j*0.1) & (nump <= (j+1)*0.1)
        a.append(round(np.sum(mask) / t, 3))

    print('mask', a)

    return c


def plot_img_small(images, listik, score0, score, targets, out):
    fig, axs = plt.subplots(3, len(listik), figsize=(3, 10))
    fig.suptitle('{:.02f} -> {:.02f}'.format(score0, score))
    for i in range(len(listik)):
        id = listik[i]
        # img = Image.open(os.path.join(path, id, 'images', id+'.png'))
        img = (255 * images[i]).cpu().to(torch.uint8)
        axs[0].imshow(torch.movedim(img, 0, 2))
        axs[0].set_axis_off()

        img = (255 * torch.ones((3, 224, 224))).cpu().to(torch.uint8)
        img = draw_segmentation_masks(img, targets[i]['masks'] > 0.5, alpha=1)
        axs[1].imshow(torch.movedim(img, 0, 2))
        axs[1].set_axis_off()

        img = (255 * torch.ones((3, 224, 224))).cpu().to(torch.uint8)
        img = draw_segmentation_masks(img, out[i]['masks'][:, 0] > 0.5, alpha=1)
        m = al(out[i])
        axs[2].set_title('{:.05f}'.format(m))
        axs[2].imshow(torch.movedim(img, 0, 2))
        axs[2].set_axis_off()



    plt.show()

if __name__ == '__main__':
    path1 = '/home/alex/PycharmProjects/dataset/data-science-bowl-2018/stage1_train'
    path2 = '/home/alex/PycharmProjects/dataset/data-science-bowl-2018/al'
    # N = 3
    model0 = None

    all_id = os.listdir(path1)

    val_img = []
    for file in os.listdir(os.path.join(path2, 'val')):
        with open(os.path.join(path2, 'val', file)) as f:
            for line in f.readlines():
                val_img.append(line.strip())

    zero = []
    with open(os.path.join(path2, 'train', 'zero.txt')) as f:
        for line in f.readlines():
            zero.append(line.strip())

    all_id = list(set(all_id) - set(val_img) - set(zero))
    for num in [1, ]:
    # for num in range(3, 28, 3):
        meanscore = []
        for n in range(10):
            sampl = random.sample(all_id, k=num)

            old_files = os.listdir(os.path.join(path2, 'train'))
            for file in old_files:
                if file != 'zero.txt':
                    os.remove(os.path.join(path2, 'train', file))

            if model0 is None:
                model0, score0 = train_al(path1, path2, n_gpu=1)
            images, targets, out = eval(model0, path1, sampl, n_gpu=1)


            with open(os.path.join(path2, 'train', 'step_1.txt'), 'w') as f:
                for name in sampl:
                    f.write(name + '\n')

            model, score = train_al(path1, path2, n_gpu=1)
            meanscore.append(score)
            print(score0, score)
            plot_img_small(images, sampl, score0, score, targets, out)
        print(num, meanscore)

