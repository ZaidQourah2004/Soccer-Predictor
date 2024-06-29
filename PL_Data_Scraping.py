import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import requests
import time
from bs4 import BeautifulSoup

class PLPredictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.matches = pd.read_csv(data_path, index_col=0)
        self.rf = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=1)
        self.prepare_data()

    def prepare_data(self):
        self.matches["date"] = pd.to_datetime(self.matches["date"])
        self.matches["h/a"] = self.matches["venue"].astype("category").cat.codes
        self.matches["opp"] = self.matches["opponent"].astype("category").cat.codes
        self.matches["hour"] = self.matches["time"].str.replace(":.+", "", regex=True).astype("int")
        self.matches["day"] = self.matches["date"].dt.dayofweek
        self.matches["target"] = (self.matches["result"] == "W").astype("int")

    def train_model(self):
        train = self.matches[self.matches["date"] < '2022-01-01']
        test = self.matches[self.matches["date"] >= '2022-01-01']
        predictors = ["h/a", "opp", "hour", "day"]
        self.rf.fit(train[predictors], train["target"])
        preds = self.rf.predict(test[predictors])
        accuracy = accuracy_score(test["target"], preds)
        precision = precision_score(test["target"], preds)
        combined = pd.DataFrame(dict(actual=test["target"], prediction=preds))
        crosstab = pd.crosstab(index=combined["actual"], columns=combined["prediction"])
        return accuracy, precision, combined, crosstab

    @staticmethod
    def rolling_averages(group, cols, new_cols):
        group = group.sort_values("date")
        rolling_stats = group[cols].rolling(3, closed='left').mean()
        group[new_cols] = rolling_stats
        group = group.dropna(subset=new_cols)
        return group

    def apply_rolling_averages(self):
        cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
        new_cols = [f"{c}_rolling" for c in cols]
        matches_rolling = self.matches.groupby("team").apply(lambda x: self.rolling_averages(x, cols, new_cols))
        matches_rolling = matches_rolling.droplevel('team')
        matches_rolling.index = range(matches_rolling.shape[0])
        self.matches = matches_rolling
        return new_cols

    def make_predictions(self, predictors):
        train = self.matches[self.matches["date"] < '2022-01-01']
        test = self.matches[self.matches["date"] >= '2022-01-01']
        self.rf.fit(train[predictors], train["target"])
        preds = self.rf.predict(test[predictors])
        combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)
        precision = precision_score(test["target"], preds)
        return combined, precision

    def merge_predictions(self, combined, new_cols):
        map_values = {
            "Brighton and Hove Albion": "Brighton",
            "Manchester United": "Manchester Utd",
            "Tottenham Hotspur": "Tottenham",
            "West Ham United": "West Ham",
            "Wolverhampton Wanderers": "Wolves"
        }
        combined["new_team"] = combined["team"].map(lambda x: map_values.get(x, x))
        merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])
        return merged

def main():
    data_path = "matches.csv"
    predictor = PLPredictor(data_path)

    # Initial training and evaluation
    accuracy, precision, combined, crosstab = predictor.train_model()
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print("Crosstab:")
    print(crosstab)

    # Apply rolling averages and make predictions with new features
    new_cols = predictor.apply_rolling_averages()
    combined, precision = predictor.make_predictions(predictors=["h/a", "opp", "hour", "day"] + new_cols)
    print(f"Precision with rolling averages: {precision}")

    combined = combined.merge(predictor.matches[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
    merged = predictor.merge_predictions(combined, new_cols)
    print(merged)

if __name__ == "__main__":
    main()
