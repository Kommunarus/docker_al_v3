import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from scripts.segmentation.dataset import Dataset_atlas, Dataset_atlas_eval
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, f1_score

backbone = 'mobilenet'
n_f = 1280

class Feature:
    def __init__(self, device, backbone):
        if backbone == 'b0':
            net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0',
                                          pretrained=True, verbose=False)
        # elif backbone == 'b4':
        #     net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4',
        #                                   pretrained=True)
        elif backbone == 'resnet50':
            net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True,
                                 verbose=False)
        # elif backbone == 'vgg16':
        #     net = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
        elif backbone == 'mobilenet':
            net = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', weights='MobileNet_V2_Weights.DEFAULT',
                                 verbose=False)
        else:
            net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0',
                                          pretrained=True, verbose=False)


        if backbone in ['b0', 'b4', '']:
            fc = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=1),
                nn.Flatten()
            )
            self.old_in = net.classifier[3].in_features
            net.classifier = fc
        elif backbone in ['resnet50']:
            fc = nn.Sequential(
                nn.Flatten()
            )
            self.old_in = net.fc.in_features
            net.fc = fc
        elif backbone in ['vgg16']:
            fc = net.classifier[:4]
            self.old_in = fc[3].in_features
            net.classifier = fc
        elif backbone in ['mobilenet']:
            fc = nn.Sequential(
                nn.Flatten()
            )
            self.old_in = net.classifier[1].in_features
            net.classifier = fc
        net.eval().to(device)
        self.net = net

    def predict(self, x):
        return self.net(x)

class NeuralNetwork(nn.Module):
    def __init__(self, n_in):
        super().__init__()

        self.fc1 = nn.Linear(n_in, 640)
        self.fc2 = nn.Linear(640, 256)
        self.fc3 = nn.Linear(256, 1)

        self.relu = nn.ReLU()

        # self.sm = nn.Softmax(dim=1)
        self.sm = nn.Sigmoid()

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        y = self.relu(self.fc1(x))
        y = self.relu(self.fc2(y))
        out = self.fc3(y)
        map = self.sm(out)

        return map


def train_model_map(labeled_data, path_to_images):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_feacher_resnet = Feature(device, backbone)

    train_ds, val_ds = train_test_split(labeled_data, test_size=0.2, random_state=42)

    ds0 = Dataset_atlas(train_ds, path_to_images, model_feacher_resnet, device)
    train_dataloader = DataLoader(ds0, batch_size=16, shuffle=True)

    loss_func = nn.MSELoss()
    # loss_func = nn.CrossEntropyLoss()
    model = NeuralNetwork(n_f).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    bestf1 = 1000
    best_model = None

    dsv = Dataset_atlas(val_ds, path_to_images, model_feacher_resnet, device)
    val_dataloader = DataLoader(dsv, batch_size=16, shuffle=True)

    for ep in range(1, 30):
        model.train()
        for batch in train_dataloader:
            fea = batch[0].to(device)
            labs = batch[1].to(device).to(torch.float)
            # labs = batch[1].to(device).to(torch.uint8)

            pred = model(fea)

            loss = loss_func(torch.squeeze(pred), labs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_true = []
            y_pred = []

            for batch in val_dataloader:
                fea = batch[0].to(device)
                labs = batch[1].to(device).to(torch.float)
                # labs = batch[1].to(device).to(torch.uint8)
                pred = model(fea)

                y_true = y_true + labs.tolist()
                y_pred = y_pred + pred.tolist()
                # y_pred = y_pred + torch.argmax(pred, 1).tolist()
        # acc = f1_score(y_true, y_pred)
        acc = mean_absolute_error(y_true, y_pred)
        if bestf1 > acc:
            bestf1 = acc
            # print(ep, acc)
            best_model = copy.deepcopy(model)

    return best_model

def eval_model_map(model, unlabeled_data, path_to_images):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_feacher_resnet = Feature(device, backbone)
    model = model.to(device)
    dsv = Dataset_atlas_eval(unlabeled_data, path_to_images, model_feacher_resnet, device)
    val_dataloader = DataLoader(dsv, batch_size=16, shuffle=False)

    model.eval()
    with torch.no_grad():
        y_pred = []

        for batch in val_dataloader:
            fea = batch.to(device)
            pred = model(fea)

            y_pred = y_pred + torch.squeeze(pred).tolist()
            # y_pred = y_pred + pred[:, 1].tolist()

    return y_pred
