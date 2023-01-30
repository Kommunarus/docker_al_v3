import numpy as np
from PIL import Image
from torchvision import transforms
import urllib
import torch
import matplotlib.pyplot as plt

url, filename = ("https://github.com/mateuszbuda/brain-segmentation-pytorch/raw/master/assets/TCGA_CS_4944.png",
                 "TCGA_CS_4944.png")
try:
    urllib.URLopener().retrieve(url, filename)
except:
    urllib.request.urlretrieve(url, filename)


model = torch.hub.load('Kommunarus/unet', 'model1',
    in_channels=3, out_channels=1, init_features=32, pretrained=True, force_reload=False)

input_image = Image.open(filename)
m, s = np.mean(input_image, axis=(0, 1)), np.std(input_image, axis=(0, 1))
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=m, std=s),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model = model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

plt.imshow(torch.round(output[0]).cpu().numpy()[0])
plt.show()

print(torch.round(output[0]))
