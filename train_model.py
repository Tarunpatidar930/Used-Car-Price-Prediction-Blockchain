import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("data/used_cars.csv")


# ==========================================
# 2. BASIC CLEANING
# ==========================================

df = df.drop_duplicates()

df = df.dropna()


# ==========================================
# 3. FEATURES & TARGET
# ==========================================

X = df.drop("Selling_Price", axis=1)

y = df["Selling_Price"]


# ==========================================
# 4. IDENTIFY FEATURES
# ==========================================

numeric_features = X.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=np.number
).columns.tolist()


print("Numeric Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ==========================================
# 5. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 6. PREPROCESSING
# ==========================================

numeric_pipeline = Pipeline([
    ("scaler", StandardScaler())
])


categorical_pipeline = Pipeline([
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])


preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),

    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])


# ==========================================
# 7. KNN MODEL
# ==========================================

pipeline = Pipeline([
    ("preprocessor", preprocessor),

    (
        "knn",
        KNeighborsRegressor()
    )
])


# ==========================================
# 8. HYPERPARAMETERS
# ==========================================

param_grid = {

    "knn__n_neighbors":
        [3, 5, 7, 9, 11],

    "knn__weights":
        ["uniform", "distance"],

    "knn__p":
        [1, 2]
}


# ==========================================
# 9. GRID SEARCH
# ==========================================

grid = GridSearchCV(
    pipeline,

    param_grid,

    cv=5,

    scoring="neg_root_mean_squared_error",

    n_jobs=-1
)


# ==========================================
# 10. TRAIN
# ==========================================

print("\nTraining model...")

grid.fit(
    X_train,
    y_train
)


# ==========================================
# 11. BEST MODEL
# ==========================================

best_model = grid.best_estimator_


print("\nBest Parameters:")
print(grid.best_params_)


# ==========================================
# 12. TEST PREDICTION
# ==========================================

y_pred = best_model.predict(
    X_test
)


# ==========================================
# 13. EVALUATION
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("MAE :", mae)

print("RMSE:", rmse)

print("R2  :", r2)


# ==========================================
# 14. SAVE MODEL
# ==========================================

joblib.dump(
    best_model,
    "car_price_model.pkl"
)


print("\nModel saved successfully!")