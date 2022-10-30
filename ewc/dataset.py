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
        
        import os
        
        def is_within_directory(directory, target):
        	
        	abs_directory = os.path.abspath(directory)
        	abs_target = os.path.abspath(target)
        
        	prefix = os.path.commonprefix([abs_directory, abs_target])
        	
        	return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
        	for member in tar.getmembers():
        		member_path = os.path.join(path, member.name)
        		if not is_within_directory(path, member_path):
        			raise Exception("Attempted Path Traversal in Tar File")
        
        	tar.extractall(path, members, numeric_owner=numeric_owner) 
        	
        
        safe_extract(tar, "./")
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