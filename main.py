import numpy as np
import torch
from torchvision import transforms, datasets, utils
#y = 0 is mask on
#y = 1 is mask off
NUM_PX = 128
BATCH_SIZE = 500


def initialize_params(feature_ct):
    w = np.zeros((feature_ct, 1))
    b = 0
    return w, b


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def load_and_standardize_images(path):
    p = transforms.Compose([transforms.Resize((NUM_PX, NUM_PX)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
    data = datasets.ImageFolder(root=path, transform=p)
    dl = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    return dl


def propagate(w, b, X, Y):
    m = X.shape[1]

    #calc Z
    Z = np.dot(w.T, X) + b

    #calc A
    A = sigmoid(Z)

    #calc cost
    cost = (-1 / m) * (np.dot(Y, np.log(A).T) + np.dot(1 - Y, np.log(1 - A).T))

    #calc dZ
    dZ = A - Y

    #calc dw
    dw = (1 / m) * np.dot(X, dZ.T)

    #calc db
    db = (1 / m) * np.sum(dZ)

    return dw, db, cost


def train(w, b, X, Y, num_iter=2000, learning_rate=0.05):

    for i in range(num_iter):

        dw, db, cost = propagate(w, b, X, Y)
        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)

    return w, b


def predict(w, b, X):
    '''
    :param w: numpy array containing the weights for each feature
    :param b: scalar offset of logistic regression formula
    :param X: numpy array containing the input features of each image
    :return Y_prediction: numpy array of predicted classifications for each image (0 == mask, 1 == no_mask)
    '''

    #calc Z
    Z = np.dot(w.T, X) + b

    #calc A
    A = sigmoid(Z)

    #calc Y_prediction
    Y_prediction = np.round(A)

    return Y_prediction


def model():
    dl = load_and_standardize_images("./images")
    dl_iter = iter(dl)
    X_train_torch, Y_train_torch = next(dl_iter)
    X_test_torch, Y_test_torch = next(dl_iter)
    X_train = X_train_torch.numpy()
    Y_train = Y_train_torch.numpy()
    X_test = X_test_torch.numpy()
    Y_test = Y_test_torch.numpy()

    print(X_train.shape[0])
    print(X_train.shape[1])
    print(X_train.shape[2])
    m_train = X_train.shape[0]
    m_test = X_test.shape[0]

    X_train_reg = X_train.reshape(m_train, NUM_PX**2 * X_train.shape[1]).T
    X_test_reg = X_test.reshape(m_test, NUM_PX**2 * X_test.shape[1]).T

    w, b = initialize_params(X_train_reg.shape[0])
    w, b = train(w, b, X_train_reg, Y_train, num_iter=2000, learning_rate=0.0005)

    Y_prediction_test = predict(w, b, X_test_reg)
    Y_prediction_train = predict(w, b, X_train_reg)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    # p = transforms.Compose([transforms.Resize((NUM_PX, NUM_PX)), transforms.Grayscale(num_output_channels=3), transforms.ToTensor()])
    # data = datasets.ImageFolder(root="./test_images", transform=p)
    # d_iter = iter(data)
    # for datum in data:
    #     x_new_torch, y_new_torch = next(d_iter)
    #     x_new = x_new_torch.numpy()
    #     #y_new = y_new_torch.numpy()
    #     m_new = 1
    #     x_new_reg = x_new.reshape(m_new, NUM_PX**2 * x_new.shape[0]).T
    #     prediction = predict(w, b, x_new_reg)
    #     print(prediction)


model()