import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

# mean = [0.4914, 0.4822, 0.4465]
# std = [0.2470, 0.2435, 0.2616]

data_transform = transforms.Compose([
    # transforms.ToPILImage(),
    # transforms.Resize(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=mean,  std=std)
])

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


    def __getitem__(self, idx):
        data = self.data[self.indxx[idx]]
        path_img = data[0]
        paths_mask = data[1]

        # image
        pil_img = Image.open(path_img)
        pil_img = pil_img.resize((224, 224))
        transformed_image = data_transform(pil_img)

        # mask
        if self.use_mask:
            masks = []
            for row in paths_mask:
                ma = Image.open(row)
                ma = ma.resize((224, 224))
                masks.append(np.array(ma))

            s_mask = np.sum(masks, 0)/255
            transformed_mask = data_transform(s_mask)
            return transformed_image, transformed_mask
        else:
            return transformed_image, idx

    def __len__(self):
        return len(self.indxx)

if __name__ == '__main__':
    listfile = ['0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9',
                '0acd2c223d300ea55d0546797713851e818e5c697d073b7f4091b96ce0f3d2fe']
    ds = Dataset_objdetect('/home/neptun/PycharmProjects/datasets/data-science-bowl-2018/stage1_train',
                           listfile)
    iterds = iter(ds)
    batch = next(iterds)
    import matplotlib.pyplot as plt
    plt.imshow(batch[1][0])
    plt.show()