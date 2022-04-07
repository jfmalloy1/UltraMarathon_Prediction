import pandas as pd
import os
from datetime import timedelta
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from tqdm import tqdm


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


def main():
    #Remove " - Sheet1" from specific weeks - ###NOTE: should only be run once per file
    weeks = ["April09"]
    remove_GSheets_labels(weeks)

    ### Clean data - should only be needed once every time a new week is added
    weeks = ["March05", "March12", "March19", "March26", "April09"]
    clean_data(weeks)

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
