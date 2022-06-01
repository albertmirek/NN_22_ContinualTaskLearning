import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from helpers.helperFunctions import load_mnist, permute_mnist

x_train, t_train, x_test, t_test = load_mnist(mnist_train, mnist_test,
                                              verbose=True)
x_train2, x_test2 = permute_mnist([x_train, x_test], 0, verbose=False)

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