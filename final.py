import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import precision_score
import kagglehub
import pickle

# Download latest version via kaggle api
path = kagglehub.dataset_download("ajaxianazarenka/premier-league")

# Importing data
matches = pd.read_csv(path+'\PremierLeague.csv', index_col=0)

matches["Date"] = pd.to_datetime(matches["Date"])
# Restricting data to matches where all attributes available for each match
matches = matches[matches["Date"] > '2019-08-01']

# Creating a format for the attributes that can be accessed by the model
matches["opp_code"] = matches["AwayTeam"].astype("category").cat.codes
matches["home_code"] = matches["HomeTeam"].astype("category").cat.codes
matches["hour"] = matches["Time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["Date"].dt.day_of_week

def result(x):
    if x == "H":
        return 2
    # elif x == "D":
    #     return 1
    else:
        return 0

matches["target"] = matches["FullTimeResult"].apply(result)

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)


# Involve teams performances in the last few games into our prediction model
grouped_matches = matches.groupby("HomeTeam")

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("Date")
    # closed = 'left' stops the rolling data to be included in the current match we are predicting
    rolling_stats = group[cols].rolling(10, closed='left').mean()
    group[new_cols] = rolling_stats
    # removes all rows with missing values
    group = group.dropna(subset=new_cols)
    return group

# Adding more data for our model to use, including team form statistics
matches["ref_code"] = matches["Referee"].astype("category").cat.codes

cols = ["FullTimeHomeTeamGoals", "FullTimeAwayTeamGoals", "HalfTimeHomeTeamGoals", "HalfTimeAwayTeamGoals", "HomeTeamShots", "AwayTeamShots", "HomeTeamShotsOnTarget", "AwayTeamShotsOnTarget", "HomeTeamCorners", "AwayTeamCorners", 
        "HomeTeamFouls", "AwayTeamFouls", "HomeTeamYellowCards", "AwayTeamYellowCards", "HomeTeamRedCards", "AwayTeamRedCards", "HomeTeamPoints", "AwayTeamPoints"]
new_cols = [f"{c}_rolling" for c in cols]

matches_rolling = matches.groupby("HomeTeam").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling.index = range(matches_rolling.shape[0])

# Trains the model and tests the model, splits data into train and test data
def make_predictions(data, predictors):
    train = data[data["Date"] < '2022-07-01']
    test = data[data["Date"] > '2022-07-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    precision = precision_score(test["target"], preds, average="weighted")
    return combined, precision

predictors = ["home_code", "opp_code", "hour", "day_code"] + new_cols

combined, precision = make_predictions(matches_rolling, predictors)
print(f"Precision: {precision.round(2)}")

combined = combined.merge(matches_rolling[["Date", "HomeTeam", "AwayTeam", "FullTimeResult"]], left_index=True, right_index=True)
print(combined)
