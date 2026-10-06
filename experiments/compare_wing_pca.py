import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from cs229.files import get_file
from cs229.load_data_wing import X_JOBLIB_NAME, Y_JOBLIB_NAME


def evaluate(components, X_train, X_test, y_train, y_test):
    model = make_pipeline(
        PCA(n_components=components),
        LinearRegression()
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    return mae, rmse, r2


def main():
    X = joblib.load(get_file('output', 'data', X_JOBLIB_NAME))
    y = joblib.load(get_file('output', 'data', Y_JOBLIB_NAME))

    # Same split for both models
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42
    )

    results = {}

    for components in [40, 60]:
        mae, rmse, r2 = evaluate(
            components,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results[components] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }

    baseline = results[40]
    improved = results[60]

    mae_improvement = (
        (baseline["mae"] - improved["mae"])
        / baseline["mae"]
        * 100
    )

    rmse_improvement = (
        (baseline["rmse"] - improved["rmse"])
        / baseline["rmse"]
        * 100
    )

    print("Wing-angle PCA comparison")
    print("=========================")

    for components in [40, 60]:
        r = results[components]

        print(
            "PCA={:>2} | MAE={:.6f} rad ({:.3f} deg) | "
            "RMSE={:.6f} rad ({:.3f} deg) | R2={:.6f}".format(
                components,
                r["mae"],
                np.degrees(r["mae"]),
                r["rmse"],
                np.degrees(r["rmse"]),
                r["r2"]
            )
        )

    print()
    print("Improvement from 40 -> 60 components")
    print("-------------------------------------")
    print("MAE improvement:  {:.2f}%".format(mae_improvement))
    print("RMSE improvement: {:.2f}%".format(rmse_improvement))
    print(
        "R2 change:         {:.6f}".format(
            improved["r2"] - baseline["r2"]
        )
    )


if __name__ == '__main__':
    main()
