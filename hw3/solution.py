#!/usr/bin/env python3
"""Train the cross-validated CatBoost ranker or generate a Kaggle submission."""

import argparse
from pathlib import Path

import pandas as pd
from catboost import CatBoostRanker, Pool


def train(data_dir: Path, model_file: Path) -> None:
    data = pd.read_csv(data_dir / "train.csv")
    # CatBoost requires documents from each query to occupy consecutive rows.
    data = data.sort_values("QueryId", kind="stable")
    features = data.drop(columns=["QueryId", "Label"])
    pool = Pool(features, label=data["Label"], group_id=data["QueryId"])

    model = CatBoostRanker(
        loss_function="YetiRank",
        iterations=900,
        depth=6,
        learning_rate=0.07,
        l2_leaf_reg=5,
        thread_count=5,
        random_seed=0,
        allow_writing_files=False,
    )
    model.fit(pool, verbose=False)
    model.save_model(str(model_file))


def predict(data_dir: Path, model_file: Path, submission_file: Path) -> None:
    model = CatBoostRanker()
    model.load_model(str(model_file))

    test = pd.read_csv(data_dir / "test.csv")
    scores = model.predict(test[model.feature_names_])
    submission = pd.DataFrame({"ObjectId": test["ObjectId"], "Score": scores})
    submission.to_csv(submission_file, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="train and save the model")
    parser.add_argument("--model_file", required=True, type=Path)
    parser.add_argument("--submission_file", type=Path)
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()

    if args.train:
        train(args.data_dir, args.model_file)
    else:
        if args.submission_file is None:
            parser.error("--submission_file is required for prediction")
        predict(args.data_dir, args.model_file, args.submission_file)


if __name__ == "__main__":
    main()
