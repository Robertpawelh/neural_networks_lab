import pandas as pd

def load_data(filename):
    train_dset = pd.read_csv(f'data/{filename}', sep=';')
    X = train_dset.drop(['label'], axis=1).to_numpy()
    Y = train_dset['label'].to_numpy()

    return X, Y

def print_data(X, Y):
    print(f'X_train: {X}')
    print(f'Y_train: {Y}')
