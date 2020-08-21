import torch

import math
import numbers
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

from PIL import Image, ImageOps
from tqdm import tqdm
from torchvision import datasets, transforms
from functions import rotation_invariant_encoding
from modules.ORConv import ORConv2d

# Training settings
parser = argparse.ArgumentParser(description='ORN.PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1234, metavar='S',
                    help='random seed (default: 1234)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--use-arf', action='store_true', default=False,
                    help='upgrading to ORN')
parser.add_argument('--orientation', type=int, default=8, metavar='O',
                    help='nOrientation for ARFs (default: 8)')


class toyDataset(Dataset):
  def __init__(self, num_samples = 10000):
    super(toyDataset, self).__init__()
    self.num_samples = num_samples
  
  def __len__(self):
    return self.num_samples

  def __getitem__(self, index):
    assert index < len(self), 'index range error'
    image = torch.randn(1, 32, 32)
    label = torch.randint(0,10,(1,)).item()
    return image, label

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
  torch.cuda.manual_seed(args.seed)


train_loader = torch.utils.data.DataLoader(toyDataset(10000),
                                           batch_size=args.batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(toyDataset(100),
                                          batch_size=args.batch_size,
                                          shuffle=False)

class Net(nn.Module):
  def __init__(self, use_arf=False, nOrientation=8):
    super(Net, self).__init__()
    self.use_arf = use_arf
    self.nOrientation = nOrientation
    if use_arf:
      self.conv1 = ORConv2d(1, 10, arf_config=(1,nOrientation), kernel_size=3)
      self.conv2 = ORConv2d(10, 20, arf_config=nOrientation,kernel_size=3)
      self.conv3 = ORConv2d(20, 40, arf_config=nOrientation,kernel_size=3, stride=1, padding=1)
      self.conv4 = ORConv2d(40, 80, arf_config=nOrientation,kernel_size=3)
    else:
      self.conv1 = nn.Conv2d(1, 80, kernel_size=3)
      self.conv2 = nn.Conv2d(80, 160, kernel_size=3)
      self.conv3 = nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1)
      self.conv4 = nn.Conv2d(320, 640, kernel_size=3)
    self.fc1 = nn.Linear(640, 1024)
    self.fc2 = nn.Linear(1024, 10)


  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = F.max_pool2d(F.relu(self.conv3(x)), 2)
    x = F.relu(self.conv4(x))
    if self.use_arf:
        x = rotation_invariant_encoding(x, self.nOrientation)
    x = x.view(-1, 640)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)

model = Net(args.use_arf, args.orientation)
print(model)
if args.cuda:
  model.cuda()

optimizer = optim.Adadelta(model.parameters())
best_test_acc = 0.

def train(epoch):
  model.train()
  for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
    if args.cuda:
      data, target = data.cuda(), target.cuda()
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    # if batch_idx % args.log_interval == 0:
  print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    epoch, batch_idx * len(data), len(train_loader.dataset),
    100. * batch_idx / len(train_loader), loss.item()))

def test(epoch):
  global best_test_acc
  model.eval()
  test_loss = 0
  correct = 0
  for data, target in tqdm(test_loader):
    with torch.no_grad():
      if args.cuda:
        data, target = data.cuda(), target.cuda()
      output = model(data)
    test_loss += F.nll_loss(output, target).item()
    pred = output.data.max(1)[1] # get the index of the max log-probability
    correct += pred.eq(target.data).cpu().sum()

  test_loss = test_loss
  test_loss /= len(test_loader) # loss function already averages over batch size
  test_acc = 100. * correct / len(test_loader.dataset)
  if test_acc > best_test_acc:
    best_test_acc = test_acc
    print('best test accuracy: {:.2f}%'.format(best_test_acc))

  print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset), test_acc))

for epoch in range(1, args.epochs + 1):
  train(epoch)
  test(epoch)

print('best test accuracy: {:.2f}%'.format(best_test_acc))
