import pandas as pd
import os
from datetime import timedelta
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from tqdm import tqdm


def remove_GSheets_labels(months):
    """ Removes the annoying " - Sheet1" file extension given through GSheets

    Args:
        months (list): list of all months to be included in data cleaning
    """
    for month in months:
        #Targets
        targets = os.listdir("Data/Targets_" + month)
        for race in targets:
            os.rename("Data/Targets_" + month + "/" + race,
                      "Data/Targets_" + month + "/" + race[:-13] + ".csv")
        #Results
        results = os.listdir("Data/Results_" + month)
        for race in results:
            os.rename("Data/Results_" + month + "/" + race,
                      "Data/Results_" + month + "/" + race[:-13] + ".csv")


def merge(months):
    """ Merges target & results dataframes into one dataframe based on names

    Args:
        months (list): months (well, technically weekends) to merge
    """
    ### Created one merged dataframe of all targets + results, merged on First & Last name for each race
    merged_df = pd.DataFrame()
    for month in months:
        print(month)
        for race in tqdm(os.listdir("Data/Targets_" + month)):
            if race.endswith(".csv"):
                #print(race)
                target_df = pd.read_csv("Data/Targets_" + month + "/" + race)
                results_df = pd.read_csv("Data/Results_" + month + "/" + race)

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


def clean_data():
    """ Takes separate target & result data, merges it, and 
        calculates the seconds elapsed for both target and result data
    """
    #Remove " - Sheet1" from specific months - should only be run once
    months = ["March05", "March12", "March19", "March26"]
    #remove_GSheets_labels(months)

    ### Merge targets & results dataframe - should only be run once
    merge(months)

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


def main():
    ### Clean data - should only be needed once
    clean_data()

    # df = pd.read_csv("Data/fullRaceData.csv")

    # train, test = split(df, 0.33)

    # #Prepare data loaders
    # train_dl = DataLoader(train, batch_size=32, shuffle=True)
    # test_dl = DataLoader(test, batch_size=1024, shuffle=False)

    # # train the model
    # for i, (inputs, targets) in enumerate(train_dl):
    #     print(i)


if __name__ == "__main__":
    main()
