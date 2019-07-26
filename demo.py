import os
import glob
import argparse

from network import feature_net

import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
from PIL import Image

demo_path = '/home/weizhaoyu/PycharmProjects/Dogs_vs_Cats_Pytorch/demo/'
resume_file = '/home/weizhaoyu/PycharmProjects/Dogs_vs_Cats_Pytorch/model/net_  1.pth'

# get parameter
parser = argparse.ArgumentParser(description='demo')
# parser.add_argument('--pre_epoch', default=0, help='begin epoch')
# parser.add_argument('--total_epoch', default=1, help='time for ergodic')
parser.add_argument('--model', default='vgg', help='model for training')
# parser.add_argument('--outf', default='/home/weizhaoyu/PycharmProjects/Dogs_vs_Cats_Pytorch/model/', help='folder to output images and model checkpoints')  # 输出结果保存路径
# parser.add_argument('--pre_model', default=False, help='use pre-model')  # 恢复训练时的模型路径
args = parser.parse_args()
# define model
model = args.model
# use gpu
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

demodir = demo_path


class TestImageFolder(data.Dataset):
    def __init__(self, root, transform=None):
        images = []
        for filename in os.listdir(root):
            images.append(root+filename)
        # for filename in sorted(glob.glob(root + "*.jpg")):
        #     images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transform = transform

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = Image.open(os.path.join(self.root, filename))
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)



transform = transforms.Compose([transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

demo_image = TestImageFolder(demodir, transform=transform)

demo_loader = data.DataLoader(
            demo_image,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=False)

print('There are %d testing images.'% len(demo_image))

classes = ['cat', 'dog']

# initialize network
use_model = feature_net(model, dim=512, n_classes=2)

# load parameters
checkpoint = torch.load(resume_file)
use_model.load_state_dict(checkpoint)

for parma in use_model.feature.parameters():
    parma.requires_grad = False

if use_cuda:
    use_model = use_model.to(device)



with torch.no_grad():
    accuracy = 0
    total = 0
    for data in demo_loader:
        use_model.eval()

        image = data
        if use_cuda:
            image = Variable(image.to(device))
        else:
            image = Variable(image)

        label_prediction = use_model(image)

        _, prediction = torch.max(label_prediction.data, 1)
        print(prediction)
        print(torch.squeeze(prediction))
        print(torch.squeeze(prediction).cpu().numpy())
