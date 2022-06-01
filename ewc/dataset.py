# @title Data-loader MNIST dataset
import tarfile, requests, os
from torchvision import transforms
from torchvision.datasets import MNIST

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