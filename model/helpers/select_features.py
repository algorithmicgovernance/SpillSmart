import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from lightgbm import LGBMRegressor


def select_kept_features(
    feature_df: pd.DataFrame,
    *,
    time_col: str = "time",
    target_col: str = "fill_ratio_target",
    n_splits: int = 3,
    sample_size: int = 20000,
    importance_frac: float = 0.05,
    corr_threshold: float = 0.90,
    random_state: int = 42,
):
    """
    Returns:
        kept_features: list[str]
    """

    # --- Prepare data ---
    model_df = (
        feature_df.dropna()
        .sort_values(time_col)
        .reset_index(drop=True)
    )

    X = model_df.drop(columns=[time_col, target_col])
    y = model_df[target_col]

    # --- Temporal split ---
    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_idx, val_idx = list(tscv.split(X))[-1]

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # --- Train model ---
    model = LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # --- Permutation importance ---
    X_val_sample = X_val.sample(min(sample_size, len(X_val)), random_state=random_state)
    y_val_sample = y_val.loc[X_val_sample.index]

    result = permutation_importance(
        model,
        X_val_sample,
        y_val_sample,
        n_repeats=1,
        random_state=random_state,
        n_jobs=-1,
    )

    importance = pd.Series(result.importances_mean, index=X_val_sample.columns)

    # --- Step 1: filter by importance ---
    importance_threshold = importance_frac * importance.max()
    selected = importance[importance > importance_threshold]

    if len(selected) == 0:
        return []

    # --- Step 2: correlation matrix ---
    corr = X[selected.index].corr().abs()

    # --- Step 3: union-find grouping ---
    features = list(corr.columns)
    parent = {f: f for f in features}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, f1 in enumerate(features):
        for f2 in features[i + 1:]:
            if corr.loc[f1, f2] >= corr_threshold:
                union(f1, f2)

    groups = {}
    for f in features:
        root = find(f)
        groups.setdefault(root, []).append(f)

    # --- Step 4: best per group ---
    kept = []
    for group in groups.values():
        group_imp = importance[group].sort_values(ascending=False)
        kept.append(group_imp.index[0])

    # --- Step 5: rank final features ---
    kept_features = (
        pd.Series(importance[kept])
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    return kept_features


#### EXAMPLE USAGE

# kept_features = select_kept_features(feature_df)