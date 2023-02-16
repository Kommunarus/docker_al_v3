import os
import random
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

from dataset import Dataset_mask as Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy

from scripts.segmentation.model_rcnn import get_model_instance_segmentation
import utils

from torchmetrics.detection.mean_ap import MeanAveragePrecision


def data_loaders(param, test=False, shuffle=True, usetransform=True):
    dataset_train = datasets(param, test, usetransform)

    loader_train = DataLoader(
        dataset_train,
        batch_size=param['batch_size'],
        shuffle=shuffle, collate_fn=utils.collate_fn)

    return loader_train


def datasets(parametres, test, usetransform):
    train = Dataset(parametres['pathdataset'], parametres['img'], test, usetransform)
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
    device_val = torch.device('cuda:0' if n_gpu == 1 else 'cuda:1')

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
    param_val['batch_size'] = 4
    param_val['img'] = val_img
    loader_val = data_loaders(param_val, shuffle=True)

    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    best_validation_dsc = 0.0
    best_model = None
    epochs = 50
    lr = 1e-4
    score_threshold = 0


    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    earling = 0


    for epoch in range(epochs):
        model.train()
        model.to(device)

        total_loss = 0
        for data in loader_train:
            images, targets = data
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            # print(targets)
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            losses.backward()
            optimizer.step()
        # if True:
        if epoch == epochs - 1:
        # if epoch % 500 == 0 and epoch != 0:
            # model.to(device_val)
            model.eval()
            # if epoch == epochs - 1:
            with torch.no_grad():
                # print(total_loss)
                # model.to(device_val)
                all_mape = []
                for data in loader_val:
                    images, targets = data
                    images = list(image.to(device) for image in images)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    out = model(images)

                    pred_map = []
                    target_map = []
                    for r in range(len(images)):
                        pred_map.append({'boxes': out[r]['boxes'],
                                 'scores': out[r]['scores'],
                                 'labels': out[r]['labels'],
                                 'masks': (out[r]['masks'][:, 0] > 0.5)})
                        target_map.append({'boxes': targets[r]['boxes'],
                                    'labels': targets[r]['labels'],
                                    'masks': (targets[r]['masks'] > 0.5)})

                    segm = True
                    if not segm:
                        metric = MeanAveragePrecision(iou_type='bbox')
                    else:
                        metric = MeanAveragePrecision(iou_type='segm')
                    metric.update(pred_map, target_map)
                    out = metric.compute()

                    all_mape.append(out['map'].item() * len(images) / len(loader_val.dataset))

            metr_s = sum(all_mape)

            if metr_s >= best_validation_dsc:
                best_validation_dsc = metr_s
                best_model = copy.deepcopy(model)
                # torch.save(model, 'rcnn.pth')
                earling = 0
                # print('best mape {:.03f} in epoch {}'.format(best_validation_dsc, epoch))
            else:
                # print('mape {:.03f}'.format(metr_s))
                earling += 1
        # else:
        #     best_model = copy.deepcopy(model)

        if earling == 5:
            # print('stop on {} epoch'.format(epoch+1))
            break
        lr_scheduler.step()

    return best_model, best_validation_dsc


def ensemble(n, path_to_images, path_to_split, n_gpu):
    models = []
    path_dir_models = './models'
    path_dir_models_copy = './copy model'
    old_model = os.listdir(path_dir_models)
    for m in old_model:
        os.remove(os.path.join(path_dir_models, m))
    vv = []
    for i in range(n):
        best_model, score = train_al(path_to_images, path_to_split, n_gpu)
        path_model1 = os.path.join(path_dir_models, f'model {i+1}.pth')
        torch.save(best_model, path_model1)
        path_model2 = os.path.join(path_dir_models_copy, f'model {i+1}.pth')
        torch.save(best_model, path_model2)
        models.append(path_model1)
        vv.append(score)

    return models, sum(vv)/len(vv), vv


def find_err(model, path_to_images, ids, n_gpu):
    param_test = dict()
    param_test['pathdataset'] = path_to_images
    param_test['batch_size'] = 8
    param_test['img'] = ids

    loader_test = data_loaders(param_test, test=True, shuffle=False)
    device = torch.device("cpu" if not torch.cuda.is_available() else f'cuda:{n_gpu}')

    model.eval()
    model.to(device)
    ids = []
    mag = []
    with torch.no_grad():
        for i, data in enumerate(loader_test):
            x, id = data
            images = list(image.to(device) for image in x)
            y_pred = model(images)
            koef_mask = []
            for row in range(len(y_pred)):
                score_box = y_pred[row]['scores']
                mask = y_pred[row]['masks']


                t = torch.nonzero(mask).shape[0]
                t_high = (mask[:, 0] > 0.5).sum().item()

                k = 100*(t - t_high) / t / mask.shape[0] #* torch.mean(score_box).item()
                koef_mask.append(k)



            ids = ids + list(id)
            mag = mag + koef_mask
            # margin_img = margin.view(-1, 1, 224, 224)
        err = [(loader_test.dataset.indxx[i], e) for i, e in zip(ids, mag)]
        err2 = sorted(err, key=lambda x: x[1])
    return err2


def ensemble_find_err(models, path_to_images, ids, n_gpu):
    param_test = dict()
    param_test['pathdataset'] = path_to_images
    param_test['batch_size'] = 16
    param_test['img'] = ids

    loader_test = data_loaders(param_test, test=True, shuffle=False)
    device = torch.device("cpu" if not torch.cuda.is_available() else f'cuda:{n_gpu}')
    maskss = {}
    for path_model in models:
        model = torch.load(path_model)
        model.eval()
        model.to(device)
        with torch.no_grad():
            for i, data in enumerate(loader_test):
                x, id = data
                images = list(image.to(device) for image in x)
                y_pred = model(images)

                koef_box = []
                koef_mask = []
                for image_in_batch in range(len(y_pred)):
                    score_box = y_pred[image_in_batch]['scores']
                    mask = y_pred[image_in_batch]['masks']
                    box = y_pred[image_in_batch]['boxes']
                    areabox = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
                    big_bbox = (areabox > 100)
                    score_box = score_box[big_bbox]
                    mask = mask[big_bbox]

                    if mask.shape[0] != 0 and len(mask.shape) == 4:
                        # t1 = torch.movedim(score_box * torch.movedim(mask[:, 0], 0, 2), 2, 0)
                        # t1 = (mask > 0.1)[:, 0].int()
                        t1 = mask[:, 0]
                    else:
                        t1 = torch.zeros((1, 224, 224)).to(int)

                    total_mask = torch.max(t1, 0)[0].cpu().numpy()
                    nnn = loader_test.dataset.indxx[id[image_in_batch]]
                    if maskss.get(nnn) is None:
                        maskss[nnn] = [total_mask]
                    else:
                        e = maskss[nnn]
                        new_e = e + [total_mask, ]
                        maskss[nnn] = new_e

    mag = []
    mig = []
    for k, v in maskss.items():
        a = fI(v)
        b = OI(v)
        if b == 0:
            koeff = 0
        else:
            koeff = 1 - a / b

        mag.append(koeff)
        mig.append(k)
    err = [(i, e) for i, e in zip(mig, mag)]
    err2 = sorted(err, key=lambda x: x[1])
    return err2


def fI(masks):
    s1 = np.prod(np.array(masks), 0)
    s2 = np.log(s1 + 0.01)
    s3 = s2 / len(masks)
    x = np.exp(s3)
    out = (x > 0.5).sum()
    return out


def OI(masks):
    s1 = np.max(np.array(masks), 0)
    out = (s1 > 0.5).sum()
    return out


def eval(model, path1, zero, n_gpu):
    device = torch.device("cpu" if not torch.cuda.is_available() else f'cuda:{n_gpu}')
    param_val = dict()
    param_val['pathdataset'] = path1
    param_val['batch_size'] = 8
    param_val['img'] = zero
    loader_val = data_loaders(param_val, shuffle=False)
    with torch.no_grad():
        # print(total_loss)
        model.eval()
        model.to(device)
        all_mape = []
        for data in loader_val:
            images, targets = data
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            out = model(images)
    return images, targets, out



if __name__ == '__main__':
    path_to_images = '/home/neptun/PycharmProjects/datasets/data-science-bowl-2018/stage1_train'
    path_to_split = '/home/neptun/PycharmProjects/datasets/data-science-bowl-2018/al'
    n_gpu = 0
    # train_al(path_to_images, path_to_split, n_gpu, ploting=False)
    # models = ensemble(2, path_to_images, path_to_split, n_gpu)
    models = ['./models/model 1.pth', './models/model 2.pth']
    ids = ['0bf33d3db4282d918ec3da7112d0bf0427d4eafe74b3ee0bb419770eefe8d7d6',
           '6af82abb29539000be4696884fc822d3cafcb2105906dc7582c92dccad8948c5']
    ensemble_find_err(models, path_to_images, ids, n_gpu)