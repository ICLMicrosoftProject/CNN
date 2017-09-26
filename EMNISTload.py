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
try: 
    from urllib.request import urlretrieve 
except ImportError: 
    from urllib import urlretrieve
####################################################Impoet All packages
##############Download the dataset
def report(count, blockSize, totalSize):
  	percent = int(count*blockSize*100/totalSize)
  	sys.stdout.write("\r%d%%" % percent + ' complete')
  	sys.stdout.flush()


print("download our datasets")
urllib.urlretrieve ("http://biometrics.nist.gov/cs_links/EMNIST/gzip.zip", "/tmp/file.zip", reporthook=report)
#################################################### Download Datasets into tmp
##############unzip the datast

zip_ref = zipfile.ZipFile('/tmp/file.zip', 'r')
zip_ref.extractall('/tmp/')
zip_ref.close()
##################################################### Unzip all the data sets


def loadData(src, cimg):
    print ('Loading ' + src)
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

###############################################Define function for Loading data, where row is 28 and column 28
###############################################Making sure that the format is correct

def loadLabels(src, cimg):
    print ('Loading ' + src)
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


######################################################

def try_readout(dataSrc, labelsSrc, cimg):
    data = loadData(dataSrc, cimg)
    labels = loadLabels(labelsSrc, cimg)
    return np.hstack((data, labels))


#################################
url_train_image = '/tmp/gzip/emnist-byclass-train-images-idx3-ubyte.gz'
url_train_labels = '/tmp/gzip/emnist-byclass-train-labels-idx1-ubyte.gz'
num_train_samples = 697932
train = try_readout(url_train_image, url_train_labels, num_train_samples)
url_test_image = '/tmp/gzip/emnist-byclass-test-images-idx3-ubyte.gz'
url_test_labels = '/tmp/gzip/emnist-byclass-test-labels-idx1-ubyte.gz'
num_test_samples = 116323
test = try_readout(url_test_image, url_test_labels, num_test_samples)


sample_number = 501
plt.imshow(train[sample_number,:-1].reshape(28,28), cmap="gray_r")
plt.show()
plt.axis('off')
print("Image Label: ", train[sample_number,-1])

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
