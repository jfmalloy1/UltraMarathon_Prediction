#pytorch mlp for regression
from numpy import vstack
from numpy import sqrt
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear
from torch.nn import Sigmoid
from torch.nn import Module
from torch.optim import SGD
from torch.nn import MSELoss
from torch.nn.init import xavier_uniform_
"""
Going through https://machinelearningmastery.com/pytorch-tutorial-develop-deep-learning-models/ tutorial

Goal is to understand regression techniques used in PyTorch, and apply them to ultramarathon prediction tool

"""


# dataset definition
class CSVDataset(Dataset):
    """ 
    Defines a python class to access/manipulate the dataset

    Loads the csv file, creates methods to return the size & specific items,
    as well as to randomly split the data into training & test sets

    """

    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path, header=None)
        # store the inputs and outputs
        self.X = df.values[:, :-1].astype('float32')
        self.y = df.values[:, -1].astype('float32')
        # ensure target has the right shape
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


# model definition
class MLP(Module):
    """ 
    Defines the NN model - in this case, there are 3 hidden layers,
    13 inputs (defined by data) in the 1st, 10 inputs in the second, 
    and 8 in the 3rd (with 1 output). The first two are activated by a sigmoid function,
    weighted by a xavier initalization scheme.

    """

    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, 10)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        # second hidden layer
        self.hidden2 = Linear(10, 8)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        # third hidden layer and output (this is new!)
        self.hidden3 = Linear(8, 6)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()
        # fourth hidden layer
        self.hidden4 = Linear(6, 1)
        xavier_uniform_(self.hidden4.weight)

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        #third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # third hidden layer and output
        X = self.hidden4(X)
        return X


# prepare the dataset
def prepare_data(path):
    """ Prepares the data for ML training & testing

    Args:
        path (str): url to the csv file containing the relevnt data

    Returns:
        array, array: returns training and test data
    """
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=1024, shuffle=False)
    return train_dl, test_dl


# train the model
def train_model(train_dl, model):
    """ Trains the NN model on the training data

    Args:
        train_dl (pytorch data): training split of full data
        model (MPL object): model defined here for NN 
    """
    # define the optimization
    #Use Mean-Squared Loss as the error criteria
    criterion = MSELoss()
    #Use standard SGD stochastic gradient descent algorithm (default parameters)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()


# evaluate the model
def evaluate_model(test_dl, model):
    """ Evaluates the trained NN model

    Args:
        test_dl (pytorch data): training split
        model (MPL object): trained NN model

    Returns:
        float: accuracy score of the model when run on the test data
    """
    # evaluate the model
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mse - from sklearn
    mse = mean_squared_error(actuals, predictions)
    return mse


# make a class prediction for one row of data
def predict(row, model):
    """ Predict a house value based on a given set of data

    Args:
        row (array): given values of data
        model (MPL object): trained NN object

    Returns:
        float: predicted value
    """
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat


# prepare the data
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))

#Define the NN
model = MLP(13)

#Train the NN model
train_model(train_dl, model)

#Evaluate the NN model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)

# make a single prediction (expect class=1)
row = [
    0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30,
    396.90, 4.98
]
yhat = predict(row, model)
print('Predicted: %.3f' % yhat)
