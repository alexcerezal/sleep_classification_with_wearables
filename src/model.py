import pandas as pd

from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Cargar features ya extraídas
df = pd.read_csv(r"C:\Proyectos_compartidos\TFG\sleep_classification_with_wearables\data\synthetic_sleep_features.csv")

# Columnas que no son features
non_feature_cols = ["subject_id", "epoch", "label"] #["subject_id", "epoch_id", "etiqueta"]

X = df.drop(columns=non_feature_cols)
y = df["label"] #df["etiqueta"]
groups = df["subject_id"]

# Split por sujeto
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=100
)

train_idx, test_idx = next(splitter.split(X, y, groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train = groups.iloc[train_idx]

param_grid = {
    "rf__n_estimators": [100, 300, 500],
    "rf__max_depth": [10, 20, None],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
}

cv = GroupKFold(n_splits=5)

# Modelo
model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("rf", RandomForestClassifier(
        class_weight="balanced",
        random_state=100,
        n_jobs=-1
    ))
])

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=2
)

# Entrenamiento
grid.fit(X_train, y_train, groups=groups_train)

# Evaluación
y_pred = grid.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

importances = pd.Series(
    grid.best_estimator_.named_steps["rf"].feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print(importances.head(20))

