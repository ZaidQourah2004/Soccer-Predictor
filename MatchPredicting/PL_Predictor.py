## Zaid Qourah
## PL Predictor using scikit-learn to predict from the matches.csv stat sheet containing data from all matches from 2020-2024 Premier League Seasons

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score

### Data Preparation Module

def load_and_prepare_data(file_path):
    matches = pd.read_csv(file_path, index_col=0)
    matches["date"] = pd.to_datetime(matches["date"])
    matches["h/a"] = matches["venue"].astype("category").cat.codes
    matches["opp"] = matches["opponent"].astype("category").cat.codes
    matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
    matches["day"] = matches["date"].dt.dayofweek
    matches["target"] = (matches["result"] == "W").astype("int")
    return matches

### Rolling Averages Module

def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

def apply_rolling_averages(matches, cols):
    new_cols = [f"{c}_rolling" for c in cols]
    matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
    matches_rolling = matches_rolling.droplevel('team')
    matches_rolling.index = range(matches_rolling.shape[0])
    return matches_rolling, new_cols

### Model Training and Prediction Module

def train_and_predict(train_data, test_data, predictors):
    rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)
    rf.fit(train_data[predictors], train_data["target"])
    preds = rf.predict(test_data[predictors])
    accuracy = accuracy_score(test_data["target"], preds)
    precision = precision_score(test_data["target"], preds)
    return preds, accuracy, precision

def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    preds, _, precision = train_and_predict(train, test, predictors)
    combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)
    return combined, precision

### Mapping and Merging Module

class MissingDict(dict):
    __missing__ = lambda self, key: key

def create_mapping():
    map_values = {
        "Brighton and Hove Albion": "Brighton",
        "Manchester United": "Manchester Utd",
        "Tottenham Hotspur": "Tottenham",
        "West Ham United": "West Ham",
        "Wolverhampton Wanderers": "Wolves"
    }
    return MissingDict(**map_values)

def merge_predictions(combined, matches_rolling):
    combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
    mapping = create_mapping()
    combined["new_team"] = combined["team"].map(mapping)
    merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])
    return merged

### Main Script

if __name__ == "__main__":
    file_path = "matches.csv"
    matches = load_and_prepare_data(file_path)

    cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
    matches_rolling, new_cols = apply_rolling_averages(matches, cols)

    predictors = ["h/a", "opp", "hour", "day"]
    combined, precision = make_predictions(matches_rolling, predictors + new_cols)

    merged = merge_predictions(combined, matches_rolling)
    print(merged)
