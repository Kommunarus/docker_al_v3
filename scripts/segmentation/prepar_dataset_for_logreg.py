import os
from scripts.segmentation.train_rcnn import train_al, data_loaders, eval
import torch
import json
import numpy as np

path1 = '/home/alex/PycharmProjects/dataset/data-science-bowl-2018/stage1_train'
path2 = '/home/alex/PycharmProjects/dataset/data-science-bowl-2018/al'

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

old_files = os.listdir(os.path.join(path2, 'train'))
for file in old_files:
    if file != 'zero.txt':
        os.remove(os.path.join(path2, 'train', file))

all_id = list(set(all_id) - set(val_img) - set(zero))

model0, score0 = train_al(path1, path2, n_gpu=1)

with open('dataset_for_logres.txt', 'w') as file_ds:
    for id in all_id:
        sampl = [id, ]

        old_files = os.listdir(os.path.join(path2, 'train'))
        for file in old_files:
            if file != 'zero.txt':
                os.remove(os.path.join(path2, 'train', file))

        images, targets, out = eval(model0, path1, sampl, n_gpu=1)

        with open(os.path.join(path2, 'train', 'step_1.txt'), 'w') as f:
            for name in sampl:
                f.write(name + '\n')

        _, score = train_al(path1, path2, n_gpu=1)

        t = torch.nonzero(out[0]['masks']).shape[0]
        nump = out[0]['masks'][:, 0].cpu().numpy()

        a = []
        for j in range(10):
            mask = (nump > j * 0.1) & (nump <= (j + 1) * 0.1)
            a.append(round(np.sum(mask) / t, 3))

        dict_out = {'id': id, 'reward': score-score0, 'mask': a}
        print(json.dumps(dict_out))

        file_ds.write(json.dumps(dict_out) + '\n')