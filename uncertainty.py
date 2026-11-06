import os

import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split


# Input and output paths
base_dir = "/mnt/private1/luansl/Basic_Data_All"
training_file = os.path.join(base_dir, "final_filtered_data.csv")

# Adjust these templates if the directory or filenames differ.
water_files = {
    "Lake": os.path.join(base_dir, "Lake", "water{year}{month:02d}.csv"),
    "Reservoir": os.path.join(base_dir, "Reservoir", "water{year}{month:02d}.csv"),
}
output_dir = os.path.join(base_dir, "Flux_predictions_100_runs")

# Model settings
selected_features = ["DO", "Cond", "PI", "DOC", "pH"]
target = "CH4_evasio"
years = range(2000, 2024)
months = range(1, 13)
n_runs = 100
test_fraction = 0.20
n_jobs = 8  # Adjust according to available CPU and memory.
num_boost_round = 800

params = {
    "objective": "reg:squarederror",
    "learning_rate": 0.01,
    "max_depth": 5,
    "alpha": 0.1,
    "lambda": 1,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "seed": 42,
    "nthread": 1,
}

prediction_columns = [f"flux_run_{run_id + 1:03d}" for run_id in range(n_runs)]


def check_columns(df, required_columns, file_name):
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_name}: {missing}")


def train_one_model(run_id, features, target_values):
    # Use a different random training subset for each model.
    train_idx, _ = train_test_split(
        np.arange(len(features)), test_size=test_fraction, random_state=run_id
    )
    dtrain = xgb.DMatrix(features.iloc[train_idx], label=target_values.iloc[train_idx])
    return xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)


def main():
    # Check all expected input files before training.
    input_paths = [
        template.format(year=year, month=month)
        for template in water_files.values()
        for year in years
        for month in months
    ]
    missing_files = [
        path for path in [training_file] + input_paths if not os.path.isfile(path)
    ]
    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} input file(s) were not found:\n"
            + "\n".join(missing_files)
        )

    print(f"Found {len(input_paths)} monthly water-quality files.")
    training_data = pd.read_csv(training_file)
    check_columns(training_data, selected_features + [target], training_file)
    training_data = training_data.dropna(subset=[target]).reset_index(drop=True)
    if len(training_data) < 10:
        raise ValueError("Too few training observations with a valid target value.")

    features = training_data[selected_features]
    target_values = training_data[target]
    print(f"Training observations: {len(training_data)}")
    print(f"Number of model runs: {n_runs}")

    # Train once and reuse the same models for all months and water types.
    models = Parallel(n_jobs=n_jobs)(
        delayed(train_one_model)(run_id, features, target_values)
        for run_id in range(n_runs)
    )
    print(f"Trained {len(models)} models.")

    for water_type, input_template in water_files.items():
        type_output_dir = os.path.join(output_dir, water_type)
        os.makedirs(type_output_dir, exist_ok=True)
        for year in years:
            for month in months:
                input_file = input_template.format(year=year, month=month)
                print(f"Processing {water_type} {year}-{month:02d}: {input_file}")
                water_data = pd.read_csv(input_file)
                check_columns(water_data, selected_features, input_file)
                if water_data.empty:
                    raise ValueError(f"Input file is empty: {input_file}")
                existing = [c for c in prediction_columns if c in water_data.columns]
                if existing:
                    raise ValueError(
                        f"Prediction columns already exist in {input_file}: {existing}"
                    )

                # Preserve all original columns and the order of input rows.
                dwater = xgb.DMatrix(water_data[selected_features])
                predictions = np.column_stack([model.predict(dwater) for model in models])
                predictions_df = pd.DataFrame(
                    predictions, columns=prediction_columns, index=water_data.index
                )
                output_data = pd.concat([water_data, predictions_df], axis=1)
                output_file = os.path.join(
                    type_output_dir, f"flux_{year}{month:02d}_100_runs.csv"
                )
                output_data.to_csv(output_file, index=False)
                print(f"Saved {len(output_data)} rows to {output_file}")

    print("All monthly flux predictions have been saved.")


if __name__ == "__main__":
    main()

