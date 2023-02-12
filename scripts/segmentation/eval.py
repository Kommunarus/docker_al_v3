import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL.Image as Image
import os
from scripts.segmentation.train_rcnn import data_loaders
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.utils import draw_segmentation_masks

def inference(ids, segm, usetransform, namemodel, namefile, truska):
    path_to_dir = '/home/alex/PycharmProjects/dataset/data-science-bowl-2018/stage1_train'
    score_threshold = 0.1
    model = torch.load(namemodel)
    model.eval()
    device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda:1')
    param_train = dict()
    param_train['pathdataset'] = path_to_dir
    param_train['batch_size'] = 1
    param_train['img'] = ids
    loader_train = data_loaders(param_train, test=False, usetransform=usetransform)
    model.to(device)

    pred_map = []
    target_map = []

    for data in loader_train:
        images, targets = data
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            out = model(images)

        pred_map.append({'boxes': out[0]['boxes'],
              'scores': out[0]['scores'],
              'labels': out[0]['labels'],
              'masks': (out[0]['masks'][:, 0] > 0.5)})

        target_map.append({'boxes': targets[0]['boxes'],
               'labels':targets[0]['labels'],
               'masks': (targets[0]['masks'] > 0.5)})

    if not segm:
        fig, ax = plt.subplots()
        ax.imshow(torch.movedim(images[0].cpu(), 0, 2))

        for row in range(out[0]['boxes'].shape[0]):
            if out[0]['scores'][row] > score_threshold:
                bbox = out[0]['boxes'][row].cpu().detach().numpy()
                rect = patches.Rectangle((bbox[0], bbox[1]), (bbox[2]-bbox[0]), (bbox[3]-bbox[1]), linewidth=1,
                                         edgecolor='r',  facecolor='none')
                ax.add_patch(rect)
        ax.set_axis_off()
        plt.savefig(namefile, bbox_inches='tight')

        if truska:
            fig, ax = plt.subplots()
            ax.imshow(torch.movedim(images[0].cpu(), 0, 2))
            for row in range(targets[0]['boxes'].shape[0]):
                bbox = targets[0]['boxes'][row].cpu()
                rect = patches.Rectangle((bbox[0], bbox[1]), (bbox[2] - bbox[0]), (bbox[3] - bbox[1]), linewidth=1,
                                         edgecolor='g',
                                         facecolor='none')
                ax.add_patch(rect)
            ax.set_axis_off()
            plt.savefig('img_tr.jpg', bbox_inches='tight')
        fig.clf()
        plt.close('all')
    else:
        fig, ax = plt.subplots()
        img = (255*images[0]).cpu().to(torch.uint8)
        img = draw_segmentation_masks(img, (out[0]['masks'] > 0.5)[:, 0], alpha=0.8,)
        ax.imshow(torch.movedim(img, 0, 2))
        ax.set_axis_off()
        plt.savefig(namefile, bbox_inches='tight')

        if truska:
            fig, ax = plt.subplots()
            img = (255*images[0]).cpu().to(torch.uint8)
            img = draw_segmentation_masks(img, targets[0]['masks']>0.5, alpha=0.8)
            ax.imshow(torch.movedim(img, 0, 2))
            ax.set_axis_off()
            plt.savefig('img_tr.jpg', bbox_inches='tight')

        fig.clf()
        plt.close('all')

    # mape

    if not segm:
        metric = MeanAveragePrecision(iou_type='bbox')
    else:
        metric = MeanAveragePrecision(iou_type='segm')
    metric.update(pred_map, target_map)
    out = metric.compute()
    return out, pred_map, target_map
