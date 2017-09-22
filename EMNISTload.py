import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
import os
import cntk as C
import __future__
import urllib
import zipfile
import csv
import re
import gzip
import shutil
import struct

##############Download the dataset
def report(count, blockSize, totalSize):
  	percent = int(count*blockSize*100/totalSize)
  	sys.stdout.write("\r%d%%" % percent + ' complete')
  	sys.stdout.flush()

#sys.stdout.write('\rFetching ' + name + '...\n')

print("download our datasets")
urllib.urlretrieve ("http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip", "/tmp/file.zip", reporthook=report)
#pbar.finish()



########################

#zip_ref = zipfile.ZipFile('/home/hans/Desktop/CNTKtutorial/CNN/2/gzip.zip', 'r')
#zip_ref.extractall('/home/hans/Desktop/CNTKtutorial/CNN/2/')
#zip_ref.close()
########################



zip_ref = zipfile.ZipFile('/tmp/file.zip', 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()


try: 
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve

# Config matplotlib for inline plotting
# Functions to load MNIST images and unpack into train and test set.
# - loadData reads image data and formats into a 28x28 long array
# - loadLabels reads the corresponding labels data, 1 for each image
# - load packs the downloaded image and labels data into a combined format to be read later by 
#   CNTK text reader 

def loadData(src, cimg):
    print ('Downloading ' + src)
    gzfname, h = urlretrieve(src, './delete.me')
    print ('Done.')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x3080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))[0]
            print(n)
            if n != cimg:
                raise Exception('Invalid file: expected {0} entries.'.format(cimg))
            crow = struct.unpack('>I', gz.read(4))[0]
            ccol = struct.unpack('>I', gz.read(4))[0]
            if crow != 28 or ccol != 28:
                raise Exception('Invalid file: expected 28 rows/cols per image.')
            # Read data.
            res = np.fromstring(gz.read(cimg * crow * ccol), dtype = np.uint8)
    finally:
        os.remove(gzfname)
    return res.reshape((cimg, crow * ccol))

def loadLabels(src, cimg):
    print ('Downloading ' + src)
    gzfname, h = urlretrieve(src, './delete.me')
    print ('Done.')
    try:
        with gzip.open(gzfname) as gz:
            n = struct.unpack('I', gz.read(4))
            # Read magic number.
            if n[0] != 0x1080000:
                raise Exception('Invalid file: unexpected magic number.')
            # Read number of entries.
            n = struct.unpack('>I', gz.read(4))
            if n[0] != cimg:
                raise Exception('Invalid file: expected {0} rows.'.format(cimg))
            # Read labels.
            res = np.fromstring(gz.read(cimg), dtype = np.uint8)
    finally:
        os.remove(gzfname)
    return res.reshape((cimg, 1))

def try_download(dataSrc, labelsSrc, cimg):
    data = loadData(dataSrc, cimg)
    labels = loadLabels(labelsSrc, cimg)
    return np.hstack((data, labels))

#url_train_image = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
url_train_image = '/tmp/gzip/emnist-byclass-train-images-idx3-ubyte.gz'

url_train_labels = '/tmp/gzip/emnist-byclass-train-labels-idx1-ubyte.gz'
num_train_samples = 697932


print("Downloading train data")
train = try_download(url_train_image, url_train_labels, num_train_samples)

url_test_image = '/tmp/gzip/emnist-byclass-test-images-idx3-ubyte.gz'
url_test_labels = '/tmp/gzip/emnist-byclass-test-labels-idx1-ubyte.gz'
num_test_samples = 116323

print("Downloading test data")
test = try_download(url_test_image, url_test_labels, num_test_samples)

# Plot a random image
sample_number = 5001
plt.imshow(train[sample_number,:-1].reshape(28,28), cmap="gray_r")
plt.show()
plt.axis('off')
print("Image Label: ", train[sample_number,-1])

# Save the data files into a format compatible with CNTK text reader
def savetxt(filename, ndarray):
    with open(filename, 'w') as f:
        labels = list(map(' '.join, np.eye(62, dtype=np.uint).astype(str)))
        for row in ndarray:
            row_str = row.astype(str)
            label_str = labels[row[-1]]
            feature_str = ' '.join(row_str[:-1])
            f.write('|labels {} |features {}\n'.format(label_str, feature_str))



print ('Writing train text file...')
savetxt( "/tmp/gzip/Train-28x28_cntk_text.txt", train)

print ('Writing test text file...')
savetxt("/tmp/gzip/Test-28x28_cntk_text.txt", test)

print('Done with data loader')
