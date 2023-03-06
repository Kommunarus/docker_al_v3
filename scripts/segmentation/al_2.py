import os
import random
import torch
import numpy as np
from scripts.segmentation.train_rcnn import find_err, find_err_with_val
from scripts.segmentation.train_rcnn import train_al
import matplotlib.pyplot as plt
import PIL.Image as Image

def save_id(listi, path, prefix):
    with open(os.path.join(path, 'train', 'step_{}.txt'.format(prefix)), 'w') as f:
        for row in listi:
            f.write(row[0]+'\n')


def plot_img(path, listik, score, score2, pred_mask):
    fig, axs = plt.subplots(2, len(listik), figsize=(15, 10))
    fig.suptitle('{} {}'.format(score, score2), fontsize=20)
    for i in range(len(listik)):
        id = listik[i]
        img = Image.open(os.path.join(path, id, 'images', id+'.png'))
        axs[0, i].imshow(img, vmin=0, vmax=1)
        im = axs[1, i].imshow(pred_mask[i], vmin=0, vmax=1)
    fig.colorbar(im)
    plt.show()

if __name__ == '__main__':
    path1 = '/home/alex/PycharmProjects/dataset/data-science-bowl-2018/stage1_train'
    path2 = '/home/alex/PycharmProjects/dataset/data-science-bowl-2018/al'
    n_gpu = 0
    # n_gpu_anti = 0
    N = 40

    all_id = os.listdir(path1)
    for opyt in range(10):
        old_files = os.listdir(os.path.join(path2, 'train'))
        for file in old_files:
            if file.find('step') > -1:
                os.remove(os.path.join(path2, 'train', file))
        sc = []
        for n in range(12):

            use_img = []
            val_list = []
            for type_file in ['train']:
                for file in os.listdir(os.path.join(path2, type_file)):
                    with open(os.path.join(path2, type_file, file)) as f:
                        for line in f.readlines():
                            use_img.append(line.strip())
            for type_file in ['val']:
                for file in os.listdir(os.path.join(path2, type_file)):
                    with open(os.path.join(path2, type_file, file)) as f:
                        for line in f.readlines():
                            use_img.append(line.strip())
                            val_list.append(line.strip())

            model, score = train_al(path1, path2, n_gpu=n_gpu)
            print(score, end=' ')
            sc.append(score)

            not_label = list(set(all_id) - set(use_img))

            err_not_lab = find_err_with_val(model, path1, not_label, val_list, n_gpu=n_gpu)
            # err_not_lab = find_err(model, path1, not_label, n_gpu=n_gpu)
            q = [0.3, 0.7]
            # d = (q[-1] - q[0]) / (N - 1)
            # for i in range(N-2):
            #     q.append(q[0] + d * (i + 1))
            # q.sort()
            quantile = np.quantile([x[1] for x in err_not_lab], q)

            ind0 = len([row for row in err_not_lab if row[1] < quantile[0]])
            # ind1 = len([row for row in err_not_lab if row[1] < quantile[1]])

            if len(err_not_lab[:ind0]) > N:
                out = random.sample(err_not_lab[:ind0], k=N)
            else:
                out = err_not_lab[:N]
            # out = random.sample(err_not_lab[ind0:ind1], k=N)
            # out.append(new_l[0])

            save_id(out, path2, n)
            # out2 = [x[0] for x in out]
            # score2 = [x[1] for x in out]
            # pred_mask_al = eval(model, path1, out2, n_gpu=1)
            # plot_img(path1, out2, score, score2, pred_mask_al)
        else:
            model, score = train_al(path1, path2, n_gpu=n_gpu)
            sc.append(score)

        print(opyt, sc)
