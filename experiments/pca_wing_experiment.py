import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from cs229.files import get_file
from cs229.load_data_wing import X_JOBLIB_NAME, Y_JOBLIB_NAME


PCA_COMPONENTS = [10, 20, 40, 60, 80]


def main():
    X = joblib.load(get_file('output', 'data', X_JOBLIB_NAME))
    y = joblib.load(get_file('output', 'data', Y_JOBLIB_NAME))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42
    )

    print("Wing-angle PCA experiment")
    print("=========================")

    for components in PCA_COMPONENTS:
        model = make_pipeline(
            PCA(n_components=components),
            LinearRegression()
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        print(
            "PCA={:>3} | MAE={:.6f} | RMSE={:.6f} | R2={:.6f}".format(
                components, mae, rmse, r2
            )
        )


if __name__ == '__main__':
    main()
