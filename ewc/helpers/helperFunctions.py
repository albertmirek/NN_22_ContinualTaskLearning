# @title Helper functions
import torch
import numpy as np

def load_mnist(mnist_train, mnist_test, verbose=False, asnumpy=True):
  """
  Helper function to load MNIST data
  Note: You can try an alternate implementation with torchloaders

  Args:
    mnist_train: np.ndarray
      MNIST training data
    mnist_test: np.ndarray
      MNIST test data
    verbose: boolean
      If True, print statistics
    asnumpy: boolean
      If true, MNIST data is passed as np.ndarray

  Returns:
   X_test: np.ndarray
      Test data
    y_test: np.ndarray
      Labels corresponding to above mentioned test data
    X_train: np.ndarray
      Train data
    y_train: np.ndarray
      Labels corresponding to above mentioned train data
  """

  x_traint, t_traint = mnist_train.data, mnist_train.targets
  x_testt, t_testt = mnist_test.data, mnist_test.targets

  if asnumpy:
    # Fix dimensions and convert back to np array for code compatability
    # We aren't using torch dataloaders for ease of use
    x_traint = torch.unsqueeze(x_traint, 1)
    x_testt = torch.unsqueeze(x_testt, 1)
    x_train, x_test = x_traint.numpy().copy(), x_testt.numpy()
    t_train, t_test = t_traint.numpy().copy(), t_testt.numpy()
  else:
    x_train, t_train = x_traint, t_traint
    x_test, t_test = x_testt, t_testt

  if verbose:
    print(f"x_train dim: {x_train.shape} and type: {x_train.dtype}")
    print(f"t_train dim: {t_train.shape} and type: {t_train.dtype}")
    print(f"x_train dim: {x_test.shape} and type: {x_test.dtype}")
    print(f"t_train dim: {t_test.shape} and type: {t_test.dtype}")

  return x_train, t_train, x_test, t_test


def permute_mnist(mnist, seed, verbose=False):
    """
    Given the training set, permute pixels of each
    image.

    Args:
      mnist: np.ndarray
        MNIST Data to be permuted
      seed: int
        Set seed for reproducibility
      verbose: boolean
        If True, print statistics

    Returns:
      perm_mnist: List
        Permutated set of pixels for each incoming image
    """

    np.random.seed(seed)
    if verbose: print("Starting permutation...")
    h = w = 28
    perm_inds = list(range(h*w))
    np.random.shuffle(perm_inds)
    perm_mnist = []
    for set in mnist:
        num_img = set.shape[0]
        flat_set = set.reshape(num_img, w * h)
        perm_mnist.append(flat_set[:, perm_inds].reshape(num_img, 1, w, h))
    if verbose: print("done.")
    return perm_mnist