import os
from scripts.segmentation.train_rcnn import ensemble_find_err
from scripts.segmentation.train_rcnn import ensemble
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
    path1 = '/home/neptun/PycharmProjects/datasets/data-science-bowl-2018/stage1_train'
    path2 = '/home/neptun/PycharmProjects/datasets/data-science-bowl-2018/al'
    n_gpu = 1
    N = 3

    all_id = os.listdir(path1)
    for opyt in range(10):
        old_files = os.listdir(os.path.join(path2, 'train'))
        for file in old_files:
            if file.find('step') > -1:
                os.remove(os.path.join(path2, 'train', file))
        sc = []
        for n in range(8):

            use_img = []
            for type_file in ['train', 'val']:
                for file in os.listdir(os.path.join(path2, type_file)):
                    with open(os.path.join(path2, type_file, file)) as f:
                        for line in f.readlines():
                            use_img.append(line.strip())

            models, score = ensemble(3, path1, path2, n_gpu=n_gpu)
            print(score, end=' ')
            sc.append(score)

            not_label = list(set(all_id) - set(use_img))

            err_not_lab = ensemble_find_err(models, path1, not_label, n_gpu=n_gpu)
            out = [err_not_lab[-i] for i in range(1, N*5, 5)]

            save_id(out, path2, n)
            # out2 = [x[0] for x in out]
            # score2 = [x[1] for x in out]
            # pred_mask_al = eval(model, path1, out2, n_gpu=1)
            # plot_img(path1, out2, score, score2, pred_mask_al)
        else:
            model, score = ensemble(3, path1, path2, n_gpu=n_gpu)
            sc.append(score)

        print(opyt, sc)
