from pathlib import Path
import io
import zipfile

import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor

from evidently import Report, Dataset, DataDefinition, Regression
from evidently.presets import DataDriftPreset, RegressionPreset
from evidently.metrics import ValueDrift

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"


def load_data() -> pd.DataFrame:
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        raw_data = pd.read_csv(
            archive.open("hour.csv"),
            header=0,
            sep=",",
            parse_dates=["dteday"],
            index_col="dteday",
        )

    return raw_data


def build_datasets(raw_data: pd.DataFrame):
    target = "cnt"
    prediction = "prediction"
    numerical_features = ["temp", "atemp", "hum", "windspeed", "hr", "weekday"]
    categorical_features = ["season", "holiday", "workingday"]

    reference = raw_data.loc["2011-01-01 00:00:00":"2011-01-28 23:00:00"].copy()
    current = raw_data.loc["2011-01-29 00:00:00":"2011-02-28 23:00:00"].copy()

    model = RandomForestRegressor(
        random_state=0,
        n_estimators=50,
    )

    feature_columns = numerical_features + categorical_features

    model.fit(reference[feature_columns], reference[target])

    reference[prediction] = model.predict(reference[feature_columns])
    current[prediction] = model.predict(current[feature_columns])

    regression_definition = DataDefinition(
        numerical_columns=numerical_features + [target, prediction],
        categorical_columns=categorical_features,
        regression=[Regression(target=target, prediction=prediction)],
    )

    drift_definition = DataDefinition(
        numerical_columns=numerical_features
    )

    reference_regression = Dataset.from_pandas(
        reference, data_definition=regression_definition
    )
    current_regression = Dataset.from_pandas(
        current, data_definition=regression_definition
    )

    reference_drift = Dataset.from_pandas(
        reference[numerical_features], data_definition=drift_definition
    )
    current_drift = Dataset.from_pandas(
        current[numerical_features], data_definition=drift_definition
    )

    return reference_regression, current_regression, reference_drift, current_drift


def make_slice(dataset: Dataset, start: str, end: str) -> Dataset:
    df = dataset.as_dataframe().loc[start:end].copy()
    return Dataset.from_pandas(df, data_definition=dataset.data_definition)


def save_regression_reports(reference_dataset: Dataset, current_dataset: Dataset) -> None:
    report = Report([RegressionPreset()])

    result = report.run(current_data=reference_dataset, reference_data=None)
    result.save_html(str(STATIC_DIR / "index.html"))

    week1 = make_slice(current_dataset, "2011-01-29 00:00:00", "2011-02-07 23:00:00")
    result = report.run(current_data=week1, reference_data=reference_dataset)
    result.save_html(str(STATIC_DIR / "regression_performance_after_week1.html"))

    week2 = make_slice(current_dataset, "2011-02-08 00:00:00", "2011-02-14 23:00:00")
    result = report.run(current_data=week2, reference_data=reference_dataset)
    result.save_html(str(STATIC_DIR / "regression_performance_after_week2.html"))

    week3 = make_slice(current_dataset, "2011-02-15 00:00:00", "2011-02-21 23:00:00")
    result = report.run(current_data=week3, reference_data=reference_dataset)
    result.save_html(str(STATIC_DIR / "regression_performance_after_week3.html"))


def save_target_drift_reports(reference_dataset: Dataset, current_dataset: Dataset) -> None:
    report = Report([ValueDrift(column="cnt")])

    week1 = make_slice(current_dataset, "2011-01-29 00:00:00", "2011-02-07 23:00:00")
    result = report.run(reference_data=reference_dataset, current_data=week1)
    result.save_html(str(STATIC_DIR / "target_drift_after_week1.html"))

    week2 = make_slice(current_dataset, "2011-02-08 00:00:00", "2011-02-14 23:00:00")
    result = report.run(reference_data=reference_dataset, current_data=week2)
    result.save_html(str(STATIC_DIR / "target_drift_after_week2.html"))

    week3 = make_slice(current_dataset, "2011-02-15 00:00:00", "2011-02-21 23:00:00")
    result = report.run(reference_data=reference_dataset, current_data=week3)
    result.save_html(str(STATIC_DIR / "target_drift_after_week3.html"))


def save_data_drift_reports(reference_dataset: Dataset, current_dataset: Dataset) -> None:
    report = Report([DataDriftPreset()])

    week1 = make_slice(current_dataset, "2011-01-29 00:00:00", "2011-02-07 23:00:00")
    result = report.run(reference_data=reference_dataset, current_data=week1)
    result.save_html(str(STATIC_DIR / "data_drift_dashboard_after_week1.html"))

    week2 = make_slice(current_dataset, "2011-02-08 00:00:00", "2011-02-14 23:00:00")
    result = report.run(reference_data=reference_dataset, current_data=week2)
    result.save_html(str(STATIC_DIR / "data_drift_dashboard_after_week2.html"))


def main() -> None:
    raw_data = load_data()
    reference_regression, current_regression, reference_drift, current_drift = build_datasets(raw_data)

    save_regression_reports(reference_regression, current_regression)
    save_target_drift_reports(reference_regression, current_regression)
    save_data_drift_reports(reference_drift, current_drift)

    print(f"Reports saved to: {STATIC_DIR}")


if __name__ == "__main__":
    main()
