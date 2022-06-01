import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.transforms as transforms

import tarfile, requests, os
from torchvision import transforms
from torchvision.datasets import MNIST

from helpers.setDevice import set_device
from helpers.setSeed import set_seed
from helpers.plottingFunctions import plot_mnist, multi_task_barplot
from helpers.helperFunctions import load_mnist
from helpers.helperFunctions import permute_mnist


SEED = 2021
set_seed(seed=SEED)
DEVICE = set_device()

# @title Data-loader MNIST dataset


name = 'MNIST'
fname = name + '.tar.gz'
url = 'https://www.di.ens.fr/~lelarge/MNIST.tar.gz'

if not os.path.exists(name):
  print('\nDownloading and unpacking MNIST data. Please wait a moment...')
  r = requests.get(url, allow_redirects=True)
  with open(fname, 'wb') as fh:
    fh.write(r.content)
  with tarfile.open(fname) as tar:
    tar.extractall('./')  # Specify which folder to extract to
  os.remove(fname)
  print('\nDownloading MNIST completed.')
else:
  print('MNIST has been already downloaded.')


# Load the Data
mnist_train = MNIST('./', download=False,
                    transform=transforms.Compose([transforms.ToTensor(), ]),
                    train=True)
mnist_test = MNIST('./', download=False,
                    transform=transforms.Compose([transforms.ToTensor(), ]),
                   train=False)

class Net(nn.Module):
  """
  Simple MultiLayer CNN with following attributes and structure.
  nn.Conv2d(1, 10, kernel_size=5) # First Convolutional Layer
  nn.Conv2d(10, 20, kernel_size=5) # Second Convolutional Layer [add dropout]
  nn.Linear(320, 50) # First Fully Connected Layer
  nn.Linear(50, 10) # Second Fully Connected Layer
  """

  def __init__(self):
    """
    Initialize Multilayer CNN parameters

    Args:
      None

    Returns:
      Nothing
    """
    super(Net, self).__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    """
    Forward pass of network

    Args:
      x: np.ndarray
        Input data

    Returns:
      x: np.ndarray
        Output from final fully connected layer
    """
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return x

# @title Model Training and Testing Functions [RUN ME!]

# @markdown `train(model, x_train, t_train, optimizer, epoch, device)`
def train(model, x_train, t_train, optimizer, epoch, device):
  """
  Train function

  Args:
    model: Net() type
      Instance of the multilayer CNN
    x_train: np.ndarray
      Training data
    t_train: np.ndarray
      Labels corresponding to the training data
    optimizer: torch.optim type
      Implements Adam algorithm.
    epoch: int
      Number of epochs
    device: string
      CUDA/GPU if available, CPU otherwise

  Returns:
    Nothing
  """
  model.train()

  for start in range(0, len(t_train)-1, 256):
    end = start + 256
    x = torch.from_numpy(x_train[start:end])
    if torch.cuda.is_available():
      x = x.type(torch.cuda.FloatTensor)
    else:
      x = x.type(torch.FloatTensor)
    y = torch.from_numpy(t_train[start:end]).long()
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()

    output = model(x)
    loss = F.cross_entropy(output, y)
    loss.backward()
    optimizer.step()
  print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))


# @markdown `test(model, x_test, t_test, device)`
def test(model, x_test, t_test, device):
    """
    Test function.

    Args:
      model: Net() type
        Instance of the multilayer CNN
      x_test: np.ndarray
        Test data
      t_test: np.ndarray
        Labels corresponding to the test data
      device: string
        CUDA/GPU if available, CPU otherwise

    Returns:
      Nothing
    """
    model.eval()
    correct, test_loss = 0, 0
    for start in range(0, len(t_test)-1, 256):
      end = start + 256
      with torch.no_grad():
        x = torch.from_numpy(x_test[start:end])
        if torch.cuda.is_available():
          x = x.type(torch.cuda.FloatTensor)
        else:
          x = x.type(torch.FloatTensor)
        y = torch.from_numpy(t_test[start:end]).long()
        x, y = x.to(device), y.to(device)
        output = model(x)
        test_loss += F.cross_entropy(output, y).item()  # Sum up batch loss
        pred = output.max(1, keepdim=True)[1]  # Get the index of the max logit
        correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss /= len(t_train)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(t_test),
        100. * correct / len(t_test)))
    return 100. * correct / len(t_test)

class simpNet(nn.Module):
  """
  Defines a simple neural network with following configuration:
  nn.Linear(28*28, 320) # First Fully Connected Layer
  nn.Linear(320, 10) + ReLU # Second Fully Connected Layer
  """

  def __init__(self):
    """
    Initialize SimpNet Parameters

    Args:
      None

    Returns:
      Nothing
    """
    super(simpNet,self).__init__()
    self.linear1 = nn.Linear(28*28, 320)
    self.out = nn.Linear(320, 10)
    self.relu = nn.ReLU()

  def forward(self, img):
    """
    Forward pass of SimpNet

    Args:
      img: np.ndarray
        Input data

    Returns:
      x: np.ndarray
        Output from final fully connected layer
    """
    x = img.view(-1, 28*28)
    x = self.relu(self.linear1(x))
    x = self.out(x)
    return x

# Load in MNIST and create an additional permuted dataset
x_train, t_train, x_test, t_test = load_mnist(mnist_train, mnist_test,
                                              verbose=True)
x_train2, x_test2 = permute_mnist([x_train, x_test], 0, verbose=False)

# Plot the data to see what we're working with
print('\nTask 1: MNIST Training data:')
plot_mnist(x_train, nPlots=10)
print('\nTask 2: Permuted MNIST data:')
plot_mnist(x_train2, nPlots=10)


# Define a new model and set params
model = Net().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Train the model on MNIST
nEpochs = 3
print(f"Training model on {nEpochs} epochs...")
for epoch in range(1, nEpochs+1):
  train(model, x_train, t_train, optimizer, epoch, device=DEVICE)
  test(model, x_test, t_test, device=DEVICE)


# Test the model's accuracy on both the regular and permuted dataset

# Let's define a dictionary that holds each of the task
# datasets and labels
tasks = {'MNIST':(x_test, t_test),
         'Perm MNIST':(x_test2, t_test)}
t1_accs = []
for ti, task in enumerate(tasks.keys()):
  print(f"Testing on task {ti + 1}")
  t1_accs.append(test(model, tasks[task][0], tasks[task][1], device=DEVICE))

# And then let's plot the testing accuracy on both datasets
multi_task_barplot(t1_accs, tasks, t='Accuracy after training on Task 1 \nbut before Training on Task 2')


# Train the previously trained model on Task 2, the permuted MNIST dataset
for epoch in range(1, 3):
  train(model, x_train2, t_train, optimizer, epoch, device=DEVICE)
  test(model, x_test2, t_test, device=DEVICE)

# Same data as before, stored in a dict
tasks = {'MNIST':(x_test, t_test),
         'Perm MNIST':(x_test2, t_test)}
# Test the model on both datasets, same as before
t12_accs = []
for ti, task in enumerate(tasks.keys()):
  print(f"Testing on task {ti + 1}")
  t12_accs.append(test(model, tasks[task][0], tasks[task][1], device=DEVICE))

# And then let's plot each of the testing accuracies after the new training
multi_task_barplot(t12_accs, tasks, t='Accuracy after training on Task 1 and then Training on Task 2')