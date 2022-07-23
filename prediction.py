import pandas as pd
import os
from datetime import timedelta
from numpy import vstack
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
from torch import from_numpy
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
import numpy as np


def remove_GSheets_labels(weeks):
    """ Removes the annoying " - Sheet1" file extension given through GSheets

    Args:
        weeks (list): list of all weeks to be included in data cleaning
    """
    for week in weeks:
        #Targets
        targets = os.listdir("Data/Targets_" + week)
        for race in targets:
            os.rename("Data/Targets_" + week + "/" + race,
                      "Data/Targets_" + week + "/" + race[:-13] + ".csv")
        #Results
        results = os.listdir("Data/Results_" + week)
        for race in results:
            os.rename("Data/Results_" + week + "/" + race,
                      "Data/Results_" + week + "/" + race[:-13] + ".csv")


def merge(weeks):
    """ Merges target & results dataframes into one dataframe based on names

    Args:
        weeks (list): weeks (well, technically weekends) to merge
    """
    ### Created one merged dataframe of all targets + results, merged on First & Last name for each race
    merged_df = pd.DataFrame()
    for week in weeks:
        print(week)
        for race in tqdm(os.listdir("Data/Targets_" + week)):
            if race.endswith(".csv"):
                #print(race)
                target_df = pd.read_csv("Data/Targets_" + week + "/" + race)
                results_df = pd.read_csv("Data/Results_" + week + "/" + race)

                merged_df = merged_df.append(
                    pd.merge(target_df, results_df, on=["First", "Last"]))

    merged_df.to_csv("Data/fullRaceData.csv")


def get_seconds(t):
    """ Turn the string target & finish times into seconds

    Args:
        t (str): time, in HH:MM:SS form

    Returns:
        float: number of seconds denoted by time string
    """
    t = [int(x) for x in t.split(":")]
    d = timedelta(hours=t[0], minutes=t[1], seconds=t[2])
    return d.seconds


def clean_data(weeks):
    """ Takes separate target & result data, merges it, and 
        calculates the seconds elapsed for both target and result data
    """

    ### Merge targets & results dataframe - should only be run once
    merge(weeks)

    #Change HH:MM:SS finish times to seconds
    df = pd.read_csv("Data/fullRaceData.csv")

    df["Target_Seconds"] = df["Target"].apply(get_seconds)
    df["Time_Seconds"] = df["Time"].apply(get_seconds)

    df.to_csv("Data/fullRaceData.csv")


def split(df, percent_train):
    test_size = round(len(df) * percent_train)
    train_size = len(df) - test_size
    print(test_size, train_size)

    return random_split(df, [train_size, test_size])


class Data(Dataset):
    """ Loads dataset (in "Data/fullRaceData.csv) & properly converts time data into seconds

    Args:
        Dataset (pytorch Dataset)
    """

    def __init__(self):
        #Read in dataframe
        df = pd.read_csv("Data/fullRaceData.csv",
                         usecols=[
                             "Rank_x", "Age Rank", "Results", "Finishes",
                             "Age_y", "GP", "Rank_y", "Target_Seconds",
                             "Time_Seconds"
                         ])

        #Make sure data is only numbers
        df["Rank_x"] = df["Rank_x"].str[:-1]
        df["Age Rank"] = df["Age Rank"].str[:-1]
        df = df.apply(pd.to_numeric, errors='coerce')
        print(df.dtypes)
        df = df.astype(float)
        print(df.dtypes)

        # self.X = [df["Rank_x"], df["Age Rank"], df["Results"], df["Finishes"], df["Age_y"],
        #                  df["GP"], df["Rank_y"], df["Target_Seconds"]]
        # self.y = df["Time_Seconds"]
        self.X = df.values[:, :-1]  #All except last column
        self.y = df.values[:, -1]  #Last column (Target_Seconds)

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.25):  #Using 25% for now
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


# model definition
class MLP(Module):
    """ 
    Defines the NN model - in this case, there are 3 hidden layers,
    8 inputs (defined by data) in the 1st, 6 inputs in the second, 
    and 6 in the 3rd (with 1 output). The first two are activated by a sigmoid function,
    weighted by a xavier initalization scheme.

    """

    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, n_inputs)
        xavier_uniform_(self.hidden1.weight)
        self.act1 = Sigmoid()
        # second hidden layer
        self.hidden2 = Linear(n_inputs, n_inputs - 2)
        xavier_uniform_(self.hidden2.weight)
        self.act2 = Sigmoid()
        # third hidden layer and output (this is new!)
        self.hidden3 = Linear(n_inputs - 2, n_inputs - 3)
        xavier_uniform_(self.hidden3.weight)
        self.act3 = Sigmoid()
        # fourth hidden layer
        self.hidden4 = Linear(n_inputs - 3, 1)
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


# train the model - taken directly from machinelearningmastery.com
def train_model(train_dl, model):
    # define the optimization
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(100):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs.float())
            targets = targets.unsqueeze(1)
            targets = targets.float()
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()


# evaluate the model  - taken directly from machinelearningmastery.com
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs.float())
        print(yhat)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # store
        predictions.append(yhat)
        actuals.append(actual)

    # print(predictions)
    # print(actuals)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate mse
    mse = mean_squared_error(actuals, predictions)
    return mse


def main():
    # #Remove " - Sheet1" from specific weeks - ###NOTE: should only be run once per file
    # weeks = ["April09"]
    # remove_GSheets_labels(weeks)

    # ### Clean data - should only be needed once every time a new week is added
    # weeks = ["March05", "March12", "March19", "March26", "April09"]
    # clean_data(weeks)

    dataset = Data()

    train, test = dataset.get_splits()
    print(type(train))
    # train = from_numpy(train), float()
    # test = from_numpy(test), float()
    print("\n------ Training ----- ")
    print(train)

    print("\n------ Test ----- ")
    print(test)

    #Prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    print(train_dl)

    test_dl = DataLoader(test, batch_size=1024, shuffle=False)

    #Define the NN
    model = MLP(8)

    # train the model
    train_model(train_dl, model)
    # evaluate the model
    acc = evaluate_model(test_dl, model)
    print('Accuracy: %.3f' % acc)


if __name__ == "__main__":
    main()
