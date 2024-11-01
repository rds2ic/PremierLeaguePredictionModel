import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import precision_score
import kagglehub
import pickle

# Download latest version via kaggle api
path = kagglehub.dataset_download("ajaxianazarenka/premier-league")

# print("Path to dataset files:", path)

matches = pd.read_csv(path+'\PremierLeague.csv', index_col=0)
print(matches.tail(10))
matches["Season"].value_counts()
matches["Date"] = pd.to_datetime(matches["Date"])
matches = matches[matches["Date"] > '2019-08-01']

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


# we want to involve teams performances in the last few games into our prediction model
grouped_matches = matches.groupby("HomeTeam")

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("Date")
    # closed = 'left' stops the rolling data to be included in the current match we are predicting
    rolling_stats = group[cols].rolling(10, closed='left').mean()
    group[new_cols] = rolling_stats
    # removes all rows with missing values
    group = group.dropna(subset=new_cols)
    return group

matches["ref_code"] = matches["Referee"].astype("category").cat.codes

cols = ["FullTimeHomeTeamGoals", "FullTimeAwayTeamGoals", "HalfTimeHomeTeamGoals", "HalfTimeAwayTeamGoals", "HomeTeamShots", "AwayTeamShots", "HomeTeamShotsOnTarget", "AwayTeamShotsOnTarget", "HomeTeamCorners", "AwayTeamCorners", 
        "HomeTeamFouls", "AwayTeamFouls", "HomeTeamYellowCards", "AwayTeamYellowCards", "HomeTeamRedCards", "AwayTeamRedCards", "HomeTeamPoints", "AwayTeamPoints"]
new_cols = [f"{c}_rolling" for c in cols]

matches_rolling = matches.groupby("HomeTeam").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling.index = range(matches_rolling.shape[0])

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

# with open('rf-'+str(precision.round(2)*100).split('.')[0]+'.pkl', 'wb') as f:
#     pickle.dump(rf, f)

# precision = '63'
# with open(f'rf-{precision}.pkl', 'rb') as f:
#     rf = pickle.load(f)

# print(rf)

# upcoming_matches = pd.DataFrame({
#     "Date": pd.to_datetime(['2024-10-19', '2024-10-19', '2024-10-19', '2024-10-19', '2024-10-19', '2024-10-19', '2024-10-19', '2024-10-20', '2024-10-20', '2024-10-21']),
#     "HomeTeam": ['Tottenham', 'Fullham', 'Ipswich', 'Man United', 'Newcastle', 'Southampton', 'Bournemouth', 'Wolves', 'Liverpool', 'Nott\'m Forest'],
#     "AwayTeam": ['West Ham', 'Aston Villa', 'Everton', 'Brentford', 'Brighton', 'Leicster', 'Arsenal', 'Man City', 'Chelsea', 'Crystal Palace'],
#     "Time": ['15:00', '15:00', '15:00', '15:00', '15:00', '15:00', '17:30', '14:00', '16:30', '20:00']
# })

# upcoming_matches["home_code"] = upcoming_matches["HomeTeam"].astype("category").cat.codes
# upcoming_matches["opp_code"] = upcoming_matches["AwayTeam"].astype("category").cat.codes
# upcoming_matches["hour"] = upcoming_matches["Time"].str.replace(":.+", "", regex=True).astype("int")
# upcoming_matches["day_code"] = upcoming_matches["Date"].dt.day_of_week

# # Get rolling averages for these teams from their recent matches
# def get_team_rolling_stats(team_name, rolling_data):
#     # Filter the last available game stats for the home and away teams
#     team_data = rolling_data[rolling_data["HomeTeam"] == team_name].sort_values("Date", ascending=False).head(1)
#     return team_data[new_cols].values[0] if not team_data.empty else np.zeros(len(new_cols))

# # Append rolling averages to the upcoming matches
# for i, row in upcoming_matches.iterrows():
#     home_team_stats = get_team_rolling_stats(row["HomeTeam"], matches_rolling)
#     # away_team_stats = get_team_rolling_stats(row["AwayTeam"], matches_rolling)
#     # combined_stats = np.concatenate([home_team_stats, away_team_stats])  # Combine home and away stats
#     upcoming_matches.loc[i, new_cols] = home_team_stats

# upcoming_preds = rf.predict(upcoming_matches[predictors])
# upcoming_matches["PredictedResult"] = upcoming_preds
# print(upcoming_matches[['HomeTeam', 'AwayTeam', 'PredictedResult']])

# # print(upcoming_matches)