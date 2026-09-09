import pandas as pd

def fe_month_dummies(train_df, test_df):
    train_out, test_out = train_df.copy(), test_df.copy()

    train_out["month"] = pd.to_datetime(train_out["date"]).dt.month
    test_out["month"] = pd.to_datetime(test_out["date"]).dt.month

    train_dummies = pd.get_dummies(train_out["month"], prefix="month")
    test_dummies = pd.get_dummies(test_out["month"], prefix="month")

    train_dummies, test_dummies = train_dummies.align(
        test_dummies, join="left", axis=1, fill_value=0
    )

    train_out = pd.concat(
        [train_out.drop(columns=["month"]), train_dummies],
        axis=1
    )
    test_out = pd.concat(
        [test_out.drop(columns=["month"]), test_dummies],
        axis=1
    )

    return train_out, test_out