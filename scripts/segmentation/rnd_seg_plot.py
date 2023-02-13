import os
import random
import matplotlib.pyplot as plt
import PIL.Image as Image
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


def plot_img_small(images, listik, score0, score, targets, out):
    fig, axs = plt.subplots(3, len(listik), figsize=(10, 10))
    fig.suptitle('{:.02f} -> {:.02f}'.format(score0, score))
    for i in range(len(listik)):
        id = listik[i]
        # img = Image.open(os.path.join(path, id, 'images', id+'.png'))
        img = (255 * images[i]).cpu().to(torch.uint8)
        axs[0].imshow(torch.movedim(img, 0, 2))
        axs[0].set_axis_off()

        img = (255 * torch.ones((3, 224, 224))).cpu().to(torch.uint8)
        img = draw_segmentation_masks(img, targets[i]['masks'] > 0.7, alpha=1)
        axs[1].imshow(torch.movedim(img, 0, 2))
        axs[1].set_axis_off()

        # t = torch.nonzero(targets[i]['masks']).shape[0]
        # t1 = (targets[i]['masks'] > 0.7).sum()
        # axs[1].set_title('{:.02f}'.format((t-t1)/t))


        img = (255 * torch.ones((3, 224, 224))).cpu().to(torch.uint8)
        img = draw_segmentation_masks(img, out[i]['masks'][:, 0] > 0.7, alpha=1)
        axs[2].imshow(torch.movedim(img, 0, 2))
        axs[2].set_axis_off()

        t = torch.nonzero(out[i]['masks']).shape[0]
        t1 = (out[i]['masks'][:, 0] > 0.7).sum()
        axs[2].set_title('{:.05f}'.format(100 * (t - t1) / t / len(out[i]['masks'])))


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

            model0, score0 = train_al(path1, path2, n_gpu=1)
            images, targets, out = eval(model0, path1, sampl, n_gpu=1)


            with open(os.path.join(path2, 'train', 'step_1.txt'), 'w') as f:
                for name in sampl:
                    f.write(name + '\n')

            model, score = train_al(path1, path2, n_gpu=1)
            meanscore.append(score)
            plot_img_small(images, sampl, score0, score, targets, out)
        print(num, meanscore)

