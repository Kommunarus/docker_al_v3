import os
import random
import torch
from tqdm import tqdm
from unet import UNet
from torch.utils.data import DataLoader
import torch.optim as optim
from loss import DiceLoss
from dataset import Dataset_objdetect as Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import copy
from scripts.segmentation.metric import BinaryMetrics

def data_loaders(param, test=False, shuffle=True):
    dataset_train = datasets(param, test)

    loader_train = DataLoader(
        dataset_train,
        batch_size=param['batch_size'],
        shuffle=shuffle)

    return loader_train

def datasets(parametres, test):
    train = Dataset(parametres['pathdataset'], parametres['img'], test)
    return train

def train_al(path_to_images, path_to_split, ploting=False):

    device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda:0')

    param_train = dict()
    param_train['pathdataset'] = path_to_images
    param_train['batch_size'] = 8

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

    param_val = copy.copy(param_train)
    param_val['img'] = val_img
    loader_val = data_loaders(param_val, shuffle=False)

    unet = UNet(in_channels=4, out_channels=1)
    # unet = torch.hub.load('Kommunarus/unet', 'model1',
    #                       in_channels=4, out_channels=1, init_features=32,
    #                       pretrained=False, force_reload=False)
    unet.to(device)

    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0
    best_model = None
    epochs = 2500
    lr = 5e-4
    metric = BinaryMetrics()

    optimizer = optim.Adam(unet.parameters(), lr=lr)
    lossic_train = []
    lossic_val = []
    earling = 0
    for epoch in tqdm(range(epochs), total=epochs):
        unet.train()
        train_loss = 0
        val_loss = 0
        for i, data in enumerate(loader_train):
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)
            optimizer.zero_grad()

            y_pred = unet(x)

            loss = dsc_loss(y_pred, y_true)
            # print(epoch, loss.item())
            train_loss += loss.item()/len(data)

            loss.backward()
            optimizer.step()
        lossic_train.append(train_loss/len(train_img))
        unet.eval()
        y_t = []
        y_p = []
        for i, data in enumerate(loader_val):
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)

            y_pred = unet(x)

            loss = dsc_loss(y_pred, y_true)
            val_loss += loss.item()/len(data)

            y_t.append(y_true[:, 0].cpu().detach())
            y_p.append(y_pred.cpu().detach())
            # print(epoch, loss.item())
        pass
        a = torch.concatenate(y_t)
        b = torch.concatenate(y_p)
        m = metric(a, b)
        if m[1] > best_validation_dsc:
            best_validation_dsc = m[1]
            best_model = copy.deepcopy(unet)
            earling = 0
        else:
            earling += 1
        if earling == 50:
            # print('stop learning for step {}'.format(epoch+1))
            break
        lossic_val.append(val_loss/len(val_img))
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

    return best_model, best_validation_dsc.item()


def find_err(unet, path_to_images, ids):
    param_test = dict()
    param_test['pathdataset'] = path_to_images
    param_test['batch_size'] = 8
    param_test['img'] = ids

    loader_test = data_loaders(param_test, test=True)
    device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda:0')

    unet.eval()
    ids = []
    mag = []
    for i, data in enumerate(loader_test):
        x, id = data
        x = x.to(device)
        y_pred = unet(x)
        vpred = y_pred[:, 0]
        v1pred = 1 - vpred
        margin = 1 - (torch.abs(vpred - v1pred))
        marg_i = torch.mean(margin, (1, 2))
        ids = ids + id.tolist()
        mag = mag + marg_i.tolist()
        # margin_img = margin.view(-1, 1, 224, 224)
    err = [(loader_test.dataset.indxx[i], e) for i, e in zip(ids, mag)]
    err2 = sorted(err, key=lambda x: x[1])
    return err2


def train():
    random.seed(42)

    device = torch.device("cpu" if not torch.cuda.is_available() else 'cuda:0')

    param_train = dict()
    param_train['pathdataset'] = '/home/neptun/PycharmProjects/datasets/data-science-bowl-2018/stage1_train'
    param_train['batch_size'] = 8

    all_images = os.listdir(param_train['pathdataset'])
    train_img, val_img = train_test_split(all_images, test_size=5)
    zero_img = random.sample(train_img, k=3)

    param_train['first_labels'] = zero_img

    loader_train = data_loaders(param_train)

    param_val = copy.copy(param_train)
    param_val['first_labels'] = val_img
    loader_val = data_loaders(param_val)

    unet = UNet(in_channels=4, out_channels=1)
    unet.to(device)

    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0
    epochs = 50
    lr = 5e-4

    optimizer = optim.Adam(unet.parameters(), lr=lr)
    lossic_train = []
    lossic_val = []
    for epoch in tqdm(range(epochs), total=epochs):
        unet.train()
        train_loss = 0
        val_loss = 0
        for i, data in enumerate(loader_train):
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)
            optimizer.zero_grad()

            y_pred = unet(x)

            loss = dsc_loss(y_pred, y_true)
            # print(epoch, loss.item())
            train_loss += loss.item()/len(data)

            loss.backward()
            optimizer.step()
        lossic_train.append(train_loss/len(zero_img))
        unet.eval()
        for i, data in enumerate(loader_val):
            x, y_true = data
            x, y_true = x.to(device), y_true.to(device)

            y_pred = unet(x)

            loss = dsc_loss(y_pred, y_true)
            val_loss += loss.item()/len(data)
            # print(epoch, loss.item())
        lossic_val.append(val_loss/len(val_img))

    plt.plot(lossic_train, label='loss train')
    plt.plot(lossic_val, label='loss val')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    train()
