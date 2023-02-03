import os
import random
import torch
from tqdm import tqdm

from torch.utils.data import DataLoader

from dataset import Dataset_mask as Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

from scripts.segmentation.model_rcnn import get_model_instance_segmentation
import utils

from scripts.segmentation.engine import evaluate
from scripts.segmentation.eval import _summarize

def data_loaders(param, test=False, shuffle=True):
    dataset_train = datasets(param, test)

    loader_train = DataLoader(
        dataset_train,
        batch_size=param['batch_size'],
        shuffle=shuffle, collate_fn=utils.collate_fn)

    return loader_train

def datasets(parametres, test):
    train = Dataset(parametres['pathdataset'], parametres['img'], test)
    return train

def plot_train(model, images, targets, images_val, targets_val, score_threshold):
    model.eval()
    loss_dict = model(images)

    fig, ax = plt.subplots(2, 2)
    # train
    ax[0, 0].imshow(torch.movedim(images[0].cpu(), 0, 2))
    ax[0, 1].imshow(torch.movedim(images[0].cpu(), 0, 2))
    for row in range(targets[0]['boxes'].shape[0]):
        bbox = targets[0]['boxes'][row].cpu()
        rect = patches.Rectangle((bbox[0], bbox[1]), (bbox[2]-bbox[0]), (bbox[3]-bbox[1]), linewidth=1, edgecolor='g',
                                 facecolor='none')
        ax[0, 0].add_patch(rect)
    for row in range(loss_dict[0]['boxes'].shape[0]):
        if loss_dict[0]['scores'][row] > score_threshold:
            bbox = loss_dict[0]['boxes'][row].cpu().detach().numpy()
            rect = patches.Rectangle((bbox[0], bbox[1]), (bbox[2]-bbox[0]), (bbox[3]-bbox[1]), linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax[0, 1].add_patch(rect)
    # val
    loss_dict = model(images_val)
    ax[1, 0].imshow(torch.movedim(images_val[0].cpu(), 0, 2))
    ax[1, 1].imshow(torch.movedim(images_val[0].cpu(), 0, 2))
    for row in range(targets_val[0]['boxes'].shape[0]):
        bbox = targets_val[0]['boxes'][row].cpu()
        rect = patches.Rectangle((bbox[0], bbox[1]), (bbox[2]-bbox[0]), (bbox[3]-bbox[1]), linewidth=1, edgecolor='g',
                                 facecolor='none')
        ax[1, 0].add_patch(rect)
    for row in range(loss_dict[0]['boxes'].shape[0]):
        if loss_dict[0]['scores'][row] > score_threshold:
            bbox = loss_dict[0]['boxes'][row].cpu().detach().numpy()
            rect = patches.Rectangle((bbox[0], bbox[1]), (bbox[2]-bbox[0]), (bbox[3]-bbox[1]), linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax[1, 1].add_patch(rect)

    plt.show()


def train_al(path_to_images, path_to_split, n_gpu, ploting=False):
    num_classes = 2
    device = torch.device("cpu" if not torch.cuda.is_available() else f'cuda:{n_gpu}')

    param_train = dict()
    param_train['pathdataset'] = path_to_images
    param_train['batch_size'] = 4

    train_img = []
    for file in os.listdir(os.path.join(path_to_split, 'train')):
        with open(os.path.join(path_to_split, 'train', file)) as f:
            for line in f.readlines():
                train_img.append(line.strip())

    val_img = []
    for file in os.listdir(os.path.join(path_to_split, 'val')):
        with open(os.path.join(path_to_split, 'val', file)) as f:
            for line in f.readlines():
                val_img.append(line.strip())

    param_train['img'] = train_img

    loader_train = data_loaders(param_train)

    param_val = dict()
    param_val['pathdataset'] = path_to_images
    param_val['batch_size'] = 1
    param_val['img'] = val_img
    loader_val = data_loaders(param_val, shuffle=True)

    model = get_model_instance_segmentation(num_classes)

    model.to(device)

    best_validation_dsc = 0.0
    best_model = None
    epochs = 1000
    lr = 5e-5
    score_threshold = 0


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    earling = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data in loader_train:
            images, targets = data
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            losses.backward()
            optimizer.step()
            # lr_scheduler.step()
        if epoch % 5 == 0 and epoch != 0:
            print(total_loss)
            # iter_val = iter(loader_val)
            # data_val = next(iter_val)
            # images_val, targets_val = data_val
            # images_val = list(image.to(device) for image in images_val)
            # targets_val = [{k: v.to(device) for k, v in t.items()} for t in targets_val]

            # plot_train(model, images, targets, images_val, targets_val, score_threshold)
            val = evaluate(model, loader_val, device=device)
            print('segmentation', end=' ')
            metr_s = _summarize(val.coco_eval['segm'])
            print('box', end=' ')
            _summarize(val.coco_eval['bbox'])
            if metr_s > best_validation_dsc:
                best_validation_dsc = metr_s
                best_model = copy.deepcopy(model)
                torch.save(model, 'rcnn.pth')
                earling = 0
            else:
                earling += 1
        if earling == 10:
            pass
            # break

    # test
    if ploting:
        iter_b = iter(loader_val)
        b, tr = next(iter_b)
        out = best_model(b.to(device))
        fig, axs = plt.subplots(param_train['batch_size'], 2, figsize=(5, 15))
        for i in range(param_train['batch_size']):
            axs[i, 0].imshow(tr[i, 0].cpu().detach().numpy())
            axs[i, 1].imshow(out[i, 0].cpu().detach().numpy())
        plt.show()
    # print('best dice {}'.format(best_validation_dsc))

    return best_model, best_validation_dsc


def find_err(unet, path_to_images, ids, n_gpu):
    param_test = dict()
    param_test['pathdataset'] = path_to_images
    param_test['batch_size'] = 16
    param_test['img'] = ids

    loader_test = data_loaders(param_test, test=True)
    device = torch.device("cpu" if not torch.cuda.is_available() else f'cuda:{n_gpu}')

    unet.eval()
    ids = []
    mag = []
    for i, data in enumerate(loader_test):
        x, id = data
        x = x.to(device)
        y_pred = unet(x)
        vpred = y_pred[:, 0]

        # 1
        v1pred = 1 - vpred
        margin = 1 - (torch.abs(vpred - v1pred))
        marg_i = torch.mean(margin, (1, 2))

        # 2
        # vpred[vpred > 0.7] = 1
        # vpred[vpred < 0.3] = 0
        # v1pred = 1 - vpred
        # margin = 1 - (torch.abs(vpred - v1pred))
        # marg_i = torch.mean(margin, (1, 2))

        # 3
        # T_03_07 = ((vpred > 0.4) & (vpred < 0.6)).float()
        # T_07_1 = (vpred >= 0.7).float()
        # marg_i = torch.sum(T_03_07, (1, 2)) / 224**2

        ids = ids + id.tolist()
        mag = mag + marg_i.tolist()
        # margin_img = margin.view(-1, 1, 224, 224)
    err = [(loader_test.dataset.indxx[i], e) for i, e in zip(ids, mag)]
    err2 = sorted(err, key=lambda x: x[1])
    return err2



if __name__ == '__main__':
    path_to_images = '/home/neptun/PycharmProjects/datasets/data-science-bowl-2018/stage1_train'
    path_to_split = '/home/neptun/PycharmProjects/datasets/data-science-bowl-2018/al'
    n_gpu = 0
    train_al(path_to_images, path_to_split, n_gpu, ploting=False)
