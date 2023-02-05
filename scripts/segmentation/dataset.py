import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import albumentations as A
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# mean = [0.4914, 0.4822, 0.4465]
# std = [0.2470, 0.2435, 0.2616]
#
# data_transform = transforms.Compose([
#     # transforms.ToPILImage(),
#     # transforms.Resize(224),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=mean,  std=std)
# ])

class Dataset_mask(Dataset):
    def __init__(self, path_to_dataset, id_images, test=False, usetransform=True):
        self.data = {}
        self.path_to_dataset = path_to_dataset
        self.indxx = []
        self.use_mask = not test
        self.usetransform = usetransform

        for idx in id_images:
            file1 = os.path.join(path_to_dataset, idx, 'images')
            files = os.listdir(file1)
            images = os.path.join(file1, files[0])

            if self.use_mask:
                file2 = os.path.join(path_to_dataset, idx, 'masks')
                files = os.listdir(file2)
                masks = [os.path.join(file2, x) for x in files]
                self.data[idx] = (images, masks)
            else:
                self.data[idx] = (images, None)

            self.indxx.append(idx)
        self.transform = A.Compose([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            # A.RandomBrightnessContrast(p=0.2),
        ])


    def __getitem__(self, idx):
        data = self.data[self.indxx[idx]]
        path_img = data[0]
        paths_mask = data[1]

        # image
        pil_img = Image.open(path_img).convert("RGB")
        pil_img = pil_img.resize((224, 224))

        # mask
        if self.use_mask:
            np_arr = np.array(pil_img)

            mask = []
            for row in paths_mask:
                ma = Image.open(row)
                ma = ma.resize((224, 224))
                mask.append(np.array(ma))

            if self.usetransform:
                transformed = self.transform(image=np_arr, masks=mask)
                transformed_image = transformed['image']
                transformed_mask = transformed['masks']
                transformed_mask = np.stack(transformed_mask, 0) / 255
            else:
                transformed_image = np_arr
                transformed_mask = np.stack(mask, 0) / 255


            num_objs = len(transformed_mask)
            boxes = []
            for i in range(num_objs):
                pos = np.where(transformed_mask[i])
                xmin = np.min(pos[1])
                xmax = np.max(pos[1])
                ymin = np.min(pos[0])
                ymax = np.max(pos[0])
                if ymax - ymin > 0 and xmax - xmin > 0:
                    boxes.append([xmin, ymin, xmax, ymax])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones((num_objs,), dtype=torch.int64)
            masks = torch.as_tensor(transformed_mask, dtype=torch.uint8)

            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd

            transformed_image = transforms.ToTensor()(transformed_image)
            # transformed_mask = transforms.ToTensor()(transformed_mask)

            return transformed_image, target
        else:
            transformed_image = transforms.ToTensor()(pil_img)
            return transformed_image, idx

    def __len__(self):
        return len(self.indxx)


class Dataset_objdetect(Dataset):
    def __init__(self, path_to_dataset, id_images, test=False):
        self.data = {}
        self.path_to_dataset = path_to_dataset
        self.indxx = []
        self.use_mask = not test
        for idx in id_images:
            file1 = os.path.join(path_to_dataset, idx, 'images')
            files = os.listdir(file1)
            images = os.path.join(file1, files[0])

            if self.use_mask:
                file2 = os.path.join(path_to_dataset, idx, 'masks')
                files = os.listdir(file2)
                masks = [os.path.join(file2, x) for x in files]
                self.data[idx] = (images, masks)
            else:
                self.data[idx] = (images, None)

            self.indxx.append(idx)
        self.transform = A.Compose([
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])


    def __getitem__(self, idx):
        data = self.data[self.indxx[idx]]
        path_img = data[0]
        paths_mask = data[1]

        # image
        pil_img = Image.open(path_img)
        pil_img = pil_img.resize((224, 224))

        # mask
        if self.use_mask:
            masks = []
            for row in paths_mask:
                ma = Image.open(row)
                ma = ma.resize((224, 224))
                masks.append(np.array(ma))

            s_mask = np.sum(masks, 0)/255
            np_arr = np.array(pil_img)
            transformed = self.transform(image=np_arr, mask=s_mask)
            transformed_image = transformed['image']
            transformed_mask = transformed['mask']

            transformed_image = transforms.ToTensor()(transformed_image)
            transformed_mask = transforms.ToTensor()(transformed_mask)


            return transformed_image, transformed_mask
        else:
            transformed_image = transforms.ToTensor()(pil_img)
            return transformed_image, idx

    def __len__(self):
        return len(self.indxx)

if __name__ == '__main__':
    listfile = ['0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9',
                '0acd2c223d300ea55d0546797713851e818e5c697d073b7f4091b96ce0f3d2fe']
    ds = Dataset_mask('/home/neptun/PycharmProjects/datasets/data-science-bowl-2018/stage1_train',
                           listfile)
    iterds = iter(ds)
    batch = next(iterds)

    fig, ax = plt.subplots()
    ax.imshow(torch.movedim(batch[0], 0, 2))
    for row in range(batch[1]['boxes'].shape[0]):
        bbox = batch[1]['boxes'][row]
        rect = patches.Rectangle((bbox[0], bbox[1]), (bbox[2]-bbox[0]), (bbox[3]-bbox[1]), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()