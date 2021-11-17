import pandas as pd
import numpy as np
import idx2numpy
import gzip
import matplotlib.pyplot as plt


def load_data(filename):
    train_dset = pd.read_csv(f'data/{filename}', sep=';')
    X = train_dset.drop(['label'], axis=1).to_numpy()
    Y = train_dset['label'].to_numpy()

    return X, Y

def print_data(X, Y):
    print(f'X_train: {X}')
    print(f'Y_train: {Y}')

def read_images_labels(X_filename, Y_filename):
    with open(Y_filename, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
    if magic != 2049:
        raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
    labels = array("B", file.read())      
    
    with open(Y_filename, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
        image_data = array("B", file.read())        
    images = []

    for i in range(size):
        images.append([0] * rows * cols)

    for i in range(size):
        img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[i][:] = img            
            
    return images, labels
                
def load_gz_img(filename, flatten):
    with gzip.open(filename, 'r') as f:
        image_size = 28
        num_images = 5

        f.read(16)
        buf = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        X_data = data.reshape(num_images, image_size, image_size, 1)
        
        if flatten:
            return data.ravel()
        else:
            return data

def load_mnist_data(X_filename, Y_filename, flatten=False):#, X_test_filename, Y_test_filename):
    X_filename = f'data/{X_filename}'
    Y_filename = f'data/{Y_filename}'
    
    X_train = idx2numpy.convert_from_file(X_filename)
    Y_train = idx2numpy.convert_from_file(Y_filename)
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])

#     X_train = load_gz_img(X_filename, flatten)
#     print(X_train.shape)
#     with gzip.open(Y_filename, 'r') as f:
#         f.read()
#         buf = f.read()
#         Y_train = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
       
#     print(X_train)
#     print(Y_train)

    return X_train, Y_train

def plot_mnist_data(data, index):
    image = np.asarray(data[index]).squeeze()
    plt.imshow(image)
    plt.show()
    
def scale_min_max_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
    
def normalize_data(data):
    return (data - np.mean(data)) / (np.max(data) - np.min(data))

    # X, Y = read_images_labels(X_filename, Y_filename)
    # #X_test, Y_test = self.read_images_labels(X_test_filename, Y_test_filename)
    # return X, Y#, X_test, Y_test        