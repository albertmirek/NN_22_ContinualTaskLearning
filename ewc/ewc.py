import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

#import torchvision.datasets as datasets
import torchvision.transforms as transforms

import seaborn as sns

from helpers.setDevice import set_device
from helpers.setSeed import set_seed
from helpers.helperFunctions import load_mnist
from helpers.helperFunctions import permute_mnist

from helpers.plottingFunctions import plot_mnist
#from helpers.setSeed import seed_worker
# @title Data-loader MNIST dataset
import tarfile, requests, os
from torchvision import transforms
from torchvision.datasets import MNIST
from helpers.plottingFunctions import plot_task
#from train import train




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




SEED = 2021
set_seed(seed=SEED)
DEVICE = set_device()

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
mnist_train = MNIST('./',
                    download=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                    ]),
                    train=True)
mnist_test = MNIST('./',
                   download=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                   ]),
                   train=False)

x_train, t_train, x_test, t_test = load_mnist(mnist_train, mnist_test,
                                              verbose=True)
x_train2, x_test2 = permute_mnist([x_train, x_test], 0, verbose=False)

# Specify which classes should be part of which task
task_classes_arr = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
tasks_num = len(task_classes_arr)

# Divide the data over the different tasks
task_data_with_overlap = []
for task_id, task_classes in enumerate(task_classes_arr):
  train_mask = np.isin(t_train, task_classes)
  test_mask = np.isin(t_test, task_classes)
  x_train_task, t_train_task = x_train[train_mask], t_train[train_mask]
  x_test_task, t_test_task = x_test[test_mask], t_test[test_mask]
  # Convert the original class labels (i.e., the digits 0 to 9) to
  # "within-task labels" so that within each task one of the digits is labeled
  # as '0' and the other as '1'.
  task_data_with_overlap.append((x_train_task, t_train_task - (task_id * 2),
                                 x_test_task, t_test_task - (task_id * 2)))

# Display tasks
for sample in range(len(task_classes_arr)):
  print(f"Task: {sample + 1}")
  plot_task(task_data_with_overlap[sample][0], len(task_classes_arr))

class FBaseNet(nn.Module):
  """
  Base network that is shared between all tasks
  """

  def __init__(self, hsize=512):
    """
    Initialize parameters of base network

    Args:
      hsize: int
        Size of head in the multi-headed layout

    Returns:
      Nothing
    """
    super(FBaseNet, self).__init__()
    self.l1 = nn.Linear(784, hsize)

  def forward(self, x):
    """
    Forward pass of FBaseNet

    Args:
      x: np.ndarray
        Input data

    Returns:
      x: np.ndarray
        Outputs after passing x through first fully connected layer
    """
    x = x.view(x.size(0), -1)
    x = F.relu(self.l1(x))
    return x

class FHeadNet(nn.Module):
  """
  Output layer of FBaseNet which will be separate for each task
  """

  def __init__(self, base_net, input_size=512):
    """
    Initialize parameters of base network

    Args:
      input_size: int
        Size of input [default: 512]

    Returns:
      Nothing
    """
    super(FHeadNet, self).__init__()

    self.base_net = base_net
    self.output_layer = nn.Linear(input_size, 2)

  def forward(self, x):
    """
    Forward pass of FHeadNet

    Args:
      x: np.ndarray
        Input data

    Returns:
      x: np.ndarray
        Outputs after passing x through output layer
    """
    x = self.base_net.forward(x)
    x = self.output_layer(x)
    return x

# Define the base network (a new head is defined when we encounter a new task)
base = FBaseNet().to(DEVICE)
heads = []

# Define a list to store test accuracies for each task
accs_naive = []

# Set the number of epochs to train each task for
epochs = 3

# Loop through all tasks
for task_id in range(tasks_num):
  # Collect the training data for the new task
  x_train, t_train, _, _ = task_data_with_overlap[task_id]

  # Define a new head for this task
  model = FHeadNet(base).to(DEVICE)
  heads.append(model)

  # Set the optimizer
  optimizer = optim.SGD(heads[task_id].parameters(), lr=0.01)

  # Train the model (with the new head) on the current task
  train(heads[task_id], x_train, t_train, optimizer, epochs, device=DEVICE)

  # Test the model on all tasks seen so far
  accs_subset = []
  for i in range(0, task_id + 1):
    _, _, x_test, t_test = task_data_with_overlap[i]
    test_acc = test(heads[i], x_test, t_test, device=DEVICE)
    accs_subset.append(test_acc)
  # For unseen tasks, we don't test
  if task_id < (tasks_num - 1):
    accs_subset.extend([np.nan] * (4 - task_id))
  # Collect all test accuracies
  accs_naive.append(accs_subset)



def on_task_update(task_id, x_train, t_train, model, shared_model, fisher_dict,
                   optpar_dict, device):
  """
  Helper function to accumulate gradients to further calculate fisher scores

  Args:
    task_id: int
      ID of the task to be updated
    x_train: np.ndarray
      Training data
    t_train: np.ndarray
      Corresponding ground truth of training data
    shared_model: FBaseNet instance
      Instance of the part of the model that is shared amongst all tasks
    fisher_dict: dict
      Dictionary with fisher values
    optpar_dict: dict
      Dictionary with optimal parameter values
    device: string
      CUDA/GPU if available, CPU otherwise

  Returns:
    Nothing
  """
  model.train()
  optimizer.zero_grad()

  # Accumulating gradients
  for start in range(0, len(t_train) - 1, 256):
    end = start + 256
    x = torch.from_numpy(x_train[start:end])
    if torch.cuda.is_available():
      x = x.type(torch.cuda.FloatTensor)
    else:
      x = x.type(torch.FloatTensor)
    y = torch.from_numpy(t_train[start:end]).long()
    x, y = x.to(device), y.to(device)
    output = model(x)
    loss = F.cross_entropy(output, y)
    loss.backward()

  fisher_dict[task_id] = {}
  optpar_dict[task_id] = {}

  # Gradients accumulated can be used to calculate fisher
  for name, param in shared_model.named_parameters():
    optpar_dict[task_id][name] = param.data.clone()
    fisher_dict[task_id][name] = param.grad.data.clone().pow(2)

def train_ewc(model, shared_model, task_id, x_train, t_train, optimizer,
              epoch, ewc_lambda, fisher_dict, optpar_dict, device):
  """
  Adding Regularisation loss to training function

  Args:
    model: FHeadNet instance
      Initates a new head network for task
    task_id: int
      ID of the task to be updated
    x_train: np.ndarray
      Training data
    t_train: np.ndarray
      Corresponding ground truth of training data
    shared_model: FBaseNet instance
      Instance of the part of the model that is shared amongst all tasks
    fisher_dict: dict
      Dictionary to store fisher values
    optpar_dict: dict
      Dictionary to store optimal parameter values
    device: string
      CUDA/GPU if available, CPU otherwise
    optimizer: torch.optim type
      Implements Adam algorithm.
    num_epochs: int
      Number of epochs
    ewc_lambda: float
      EWC hyperparameter

  Returns:
    Nothing
  """
  model.train()
  for start in range(0, len(t_train) - 1, 256):
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

    for task in range(task_id):
      for name, param in shared_model.named_parameters():
        fisher = fisher_dict[task][name]
        optpar = optpar_dict[task][name]
        loss += (fisher * (optpar - param).pow(2)).sum() * ewc_lambda

    loss.backward()
    optimizer.step()

  print(f"Train Epoch: {epoch} \tLoss: {loss.item():.6f}")

# Define the base network (a new head is defined when we encounter a new task)
base = FBaseNet().to(DEVICE)
heads = []

# Define a list to store test accuracies for each task
accs_ewc = []

# Set number of epochs
epochs = 2

# Set EWC hyperparameter
ewc_lambda = 0.2

# Define dictionaries to store values needed by EWC
fisher_dict = {}
optpar_dict = {}

# Loop through all tasks
for task_id in range(tasks_num):
    # Collect the training data for the new task
    x_train, t_train, _, _ = task_data_with_overlap[task_id]

    # Define a new head for this task
    model = FHeadNet(base).to(DEVICE)
    heads.append(model)

    # Set the optimizer
    optimizer = optim.SGD(heads[task_id].parameters(), lr=0.01)

    # Train the model (with the new head) on the current task
    for epoch in range(1, epochs+1):
        train_ewc(heads[task_id], heads[task_id].base_net, task_id, x_train,
                  t_train, optimizer, epoch, ewc_lambda, fisher_dict,
                  optpar_dict, device=DEVICE)
    on_task_update(task_id, x_train, t_train, heads[task_id],
                   heads[task_id].base_net, fisher_dict, optpar_dict,
                   device=DEVICE)

    # Test the model on all tasks seen so far
    accs_subset = []
    for i in range(0, task_id + 1):
        _, _, x_test, t_test = task_data_with_overlap[i]
        test_acc = test(heads[i], x_test, t_test, device=DEVICE)
        accs_subset.append(test_acc)
    # For unseen tasks, we don't test
    if task_id < (tasks_num - 1):
        accs_subset.extend([np.nan] * (4 - task_id))
    # Collect all test accuracies
    accs_ewc.append(accs_subset)

# @title Plot Naive vs EWC results



fig, axes = plt.subplots(1, 3, figsize=(15, 6))
accs_fine_grid = np.array(accs_naive)
nan_mask = np.isnan(accs_naive)

sns.heatmap(accs_naive, vmin=0, vmax=100, mask=nan_mask, annot=True,fmt='.0f',
            yticklabels=range(1, 6), xticklabels=range(1, 6), ax=axes[0],
            cbar=False)
sns.heatmap(accs_ewc, vmin=0, vmax=100, mask=nan_mask, annot=True,fmt='.0f',
            yticklabels=range(1, 6), xticklabels=range(1, 6), ax=axes[1],
            cbar=False)

axes[0].set_ylabel('Tested on Task')

axes[0].set_xlabel('Naive')
axes[1].set_xlabel('EWC')

axes[2].plot(range(1, 6), np.nanmean(accs_naive, axis=1), linewidth=2.0)
axes[2].plot(range(1, 6), np.nanmean(accs_ewc, axis=1), linewidth=2.0)

axes[2].legend(['Naive', 'EWC'])
axes[2].set_ylabel('Accumulated Accuracy for Seen Tasks')
axes[2].set_xlabel('Task Number')
plt.show()


# !
# !
# !
# !
# !
# !
# ! Class incremental

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

# Load the MNIST dataset
x_train, t_train, x_test, t_test = load_mnist(mnist_train, mnist_test,
                                              verbose=True)

# Define which classes are part of each task
classes_per_task = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]

# Divide the MNIST dataset in tasks
task_data = []
for _, classes_in_this_task in enumerate(classes_per_task):

  # Which data-points belong to the classes in the current task?
  train_mask = np.isin(t_train, classes_in_this_task)
  test_mask = np.isin(t_test, classes_in_this_task)
  x_train_task, t_train_task = x_train[train_mask], t_train[train_mask]
  x_test_task, t_test_task = x_test[test_mask], t_test[test_mask]

  # Add the data for the current task
  task_data.append((x_train_task, t_train_task, x_test_task, t_test_task))

# In contrast to the task-incremental version of Split MNIST explored in the
# last section, now task identity information will not be provided to the model

# Define the model and the optimzer
model = Net().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Set 'lambda', the hyperparameter of EWC
ewc_lambda = 0.2

# Define dictionaries to store values needed by EWC
fisher_dict = {}
optpar_dict = {}

# Prepare list to store average accuracies after each task
ewc_accs = []

# Loop through all tasks
for id, task in enumerate(task_data):

  # Collect training data
  x_train, t_train, _, _ = task

  # Training with EWC
  print("Training on task: ", id)
  for epoch in range(1, 2):
    train_ewc(model, model, id, x_train, t_train, optimizer, epoch,
              ewc_lambda, fisher_dict, optpar_dict, device=DEVICE)

  on_task_update(id, x_train, t_train, model, model, fisher_dict,
                 optpar_dict, device=DEVICE)

  # Evaluate performance after training on this task
  avg_acc = 0
  for id_test, task in enumerate(task_data):
    print(f"Testing on task: {id_test}")
    _, _, x_test, t_test = task
    acc = test(model, x_test, t_test, device=DEVICE)
    avg_acc = avg_acc + acc

  print(f"Avg acc: {avg_acc / len(task_data)}")
  ewc_accs.append(avg_acc / len(task_data))


# !
# !
# !
# !
# !
# ! REPLAY
def shuffle_datasets(dataset, seed, in_place=False):
  """
  Shuffle a list of two (or more) datasets.

  Args:
    dataset: np.ndarray
      Dataset
    seed: Integer
      A non-negative integer that defines the random state.
    in_place: boolean
      If True, shuffle datasets in place

  Returns:
    Nothing
  """

  np.random.seed(seed)
  rng_state = np.random.get_state()
  new_dataset = []
  for x in dataset:
    if in_place:
      np.random.shuffle(x)
    else:
      new_dataset.append(np.random.permutation(x))
    np.random.set_state(rng_state)

  if not in_place:
    return new_dataset
# Define the model and the optimizer
model = Net().to(DEVICE)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Prepare list to store average accuracies after each task
rehe_accs = []

# Loop through all tasks
for id, task in enumerate(task_data):

  # Collect training data
  x_train, t_train, _, _ = task

  # Add replay
  for i in range(id):
    past_x_train, past_t_train, _, _ = task_data[i]
    x_train = np.concatenate((x_train, past_x_train))
    t_train = np.concatenate((t_train, past_t_train))

  x_train, t_train = shuffle_datasets([x_train, t_train], seed=SEED)

  # Training
  print(f"Training on task: {id}")
  for epoch in range(1, 3):
    train(model, x_train, t_train, optimizer, epoch, device=DEVICE)

  # Evaluate performance after training on this task
  avg_acc = 0
  for id_test, task in enumerate(task_data):
    print(f"Testing on task: {id_test}")
    _, _, x_test, t_test = task
    acc = test(model, x_test, t_test, device=DEVICE)
    avg_acc = avg_acc + acc

  print(f"Avg acc: {avg_acc / len(task_data)}")
  rehe_accs.append(avg_acc/len(task_data))

# @title Plot EWC vs. Replay
plt.plot([1, 2, 3, 4, 5], rehe_accs, '-o', label="Replay")
plt.plot([1, 2, 3, 4, 5], ewc_accs, '-o', label="EWC")
plt.xlabel('Tasks Encountered', fontsize=14)
plt.ylabel('Average Accuracy', fontsize=14)
plt.title('CL Strategies on Class-incremental version of Split MNIST',
          fontsize=14);
plt.xticks([1, 2, 3, 4, 5])
plt.legend(prop={'size': 16})
plt.show()