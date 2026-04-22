import numpy as np
import pandas as pd

from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)

def build_cso_model(
    model_df: pd.DataFrame,
    kept_features: list[str],
    *,
    target_col: str = "fill_ratio_target",
    time_col: str = "time",
    threshold: float = 0.95,
    test_frac: float = 0.2,
    n_splits: int = 5,
    random_state: int = 42,
    alpha: float = 0.10,
    n_estimators: int = 500,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    n_jobs: int = -1,
):
    """
    Build the CSO exceedance model using temporally blocked CV and conformal calibration.

    Returns a dict containing:
      - continuous_metrics
      - exceedance_metrics
      - fold_models
      - features
      - conf_threshold
      - alpha
      - threshold
      - conformal_scores_cso
      - conformal_scores_safe
      - oof_prob
      - clf_prob_exceed
      - test rows and labels used for evaluation
    """
    if "fill_ratio" not in model_df.columns:
        raise KeyError("model_df does not contain 'fill_ratio'.")

    features = list(kept_features) + ["fill_ratio"]

    missing_features = [c for c in features if c not in model_df.columns]
    if missing_features:
        raise KeyError(f"model_df is missing required feature columns: {missing_features}")

    data = (
        model_df[[time_col] + features + [target_col]]
        .sort_values(time_col)
        .dropna()
        .reset_index(drop=True)
    )

    X_all = data[features]
    y_all = data[target_col].astype(float)
    fill_all = data["fill_ratio"].astype(float)
    y_all_bin = (y_all > threshold).astype(int)

    split_idx = int(len(data) * (1 - test_frac))

    X_train_full = X_all.iloc[:split_idx].copy()
    y_train_full = y_all.iloc[:split_idx].copy()
    y_train_full_bin = y_all_bin.iloc[:split_idx].copy()
    fill_train_full = fill_all.iloc[:split_idx].copy()

    X_test = X_all.iloc[split_idx:].copy()
    y_test = y_all.iloc[split_idx:].copy()
    y_test_bin = y_all_bin.iloc[split_idx:].copy()
    fill_test = fill_all.iloc[split_idx:].copy()

    # Continuous baseline
    pers_cont_pred = fill_test.to_numpy()
    pers_cont_mae = mean_absolute_error(y_test, pers_cont_pred)
    pers_cont_mse = mean_squared_error(y_test, pers_cont_pred)

    continuous_metrics = pd.DataFrame(
    {
        "Persistence": {
            "MAE": pers_cont_mae,
            "MSE": pers_cont_mse,
        }
    }
    )

    # Temporally blocked CV
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_test_probs = []
    fold_models = []
    oof_prob = np.full(len(X_train_full), np.nan)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_full), start=1):
        X_train = X_train_full.iloc[train_idx]
        y_train = y_train_full_bin.iloc[train_idx]

        if y_train.nunique() < 2:
            continue

        n_pos = max(int(y_train.sum()), 1)
        n_neg = max(int((1 - y_train).sum()), 1)
        scale_pos_weight = n_neg / n_pos

        clf = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=-1,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state + fold,
            n_jobs=n_jobs,
        )

        clf.fit(X_train, y_train)

        val_prob = clf.predict_proba(X_train_full.iloc[val_idx])[:, 1]
        oof_prob[val_idx] = val_prob

        fold_test_probs.append(clf.predict_proba(X_test)[:, 1])
        fold_models.append(clf)

    if len(fold_test_probs) == 0:
        if y_train_full_bin.nunique() < 2:
            clf_prob_exceed = np.clip(fill_test.to_numpy(), 0, 1)
            fold_models = []
            conf_threshold = np.nan
            conformal_scores_cso = np.array([])
            conformal_scores_safe = np.array([])
        else:
            n_pos = max(int(y_train_full_bin.sum()), 1)
            n_neg = max(int((1 - y_train_full_bin).sum()), 1)
            scale_pos_weight = n_neg / n_pos

            clf = LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=-1,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            clf.fit(X_train_full, y_train_full_bin)
            clf_prob_exceed = clf.predict_proba(X_test)[:, 1]
            fold_models = [clf]
            conf_threshold = np.nan
            conformal_scores_cso = np.array([])
            conformal_scores_safe = np.array([])
    else:
        clf_prob_exceed = np.mean(np.vstack(fold_test_probs), axis=0)

    mask = ~np.isnan(oof_prob)
    oof_prob_clean = oof_prob[mask]
    y_oof_clean = y_train_full_bin.iloc[mask].to_numpy()

    # Default fallback if no valid fold models exist
    if len(fold_test_probs) == 0:
        clf_prob_exceed = np.clip(fill_test.to_numpy(), 0, 1)
        clf_pred_bin = (clf_prob_exceed >= 0.5).astype(int)
        conf_threshold = np.nan
        conf_pred_bin = clf_pred_bin.copy()
        conformal_scores_cso = np.array([])
        conformal_scores_safe = np.array([])
    else:
        clf_prob_exceed = np.mean(np.vstack(fold_test_probs), axis=0)
        clf_pred_bin = (clf_prob_exceed >= 0.5).astype(int)

        # Default fallback if conformal cannot be built
        conf_threshold = np.nan
        conf_pred_bin = clf_pred_bin.copy()
        conformal_scores_cso = np.array([])
        conformal_scores_safe = np.array([])

        # Only build conformal calibration if both classes are present in OOF data
        if (
            len(oof_prob_clean) > 0
            and np.any(y_oof_clean == 1)
            and np.any(y_oof_clean == 0)
        ):
            conformal_scores_cso = 1.0 - oof_prob_clean[y_oof_clean == 1]
            conformal_scores_safe = oof_prob_clean[y_oof_clean == 0]

            nonconformity = np.where(y_oof_clean == 1, 1 - oof_prob_clean, oof_prob_clean)
            n = len(nonconformity)
            q_level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
            q_hat = np.quantile(nonconformity, q_level, method="higher")
            conf_threshold = 1 - q_hat
            conf_pred_bin = (clf_prob_exceed >= conf_threshold).astype(int)

    clf_mae = mean_absolute_error(y_test_bin, clf_prob_exceed)
    clf_mse = mean_squared_error(y_test_bin, clf_prob_exceed)
    clf_brier = brier_score_loss(y_test_bin, clf_prob_exceed)
    clf_ap = average_precision_score(y_test_bin, clf_prob_exceed)

    clf_prec = precision_score(y_test_bin, clf_pred_bin, zero_division=0)
    clf_rec = recall_score(y_test_bin, clf_pred_bin, zero_division=0)
    clf_f1 = f1_score(y_test_bin, clf_pred_bin, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test_bin, clf_pred_bin, labels=[0, 1]).ravel()
    clf_spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    try:
        clf_auc = roc_auc_score(y_test_bin, clf_prob_exceed)
    except ValueError:
        clf_auc = np.nan

    conf_mae = mean_absolute_error(y_test_bin, clf_prob_exceed)
    conf_mse = mean_squared_error(y_test_bin, clf_prob_exceed)
    conf_brier = brier_score_loss(y_test_bin, clf_prob_exceed)
    conf_ap = average_precision_score(y_test_bin, clf_prob_exceed)

    conf_prec = precision_score(y_test_bin, conf_pred_bin, zero_division=0)
    conf_rec = recall_score(y_test_bin, conf_pred_bin, zero_division=0)
    conf_f1 = f1_score(y_test_bin, conf_pred_bin, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test_bin, conf_pred_bin, labels=[0, 1]).ravel()
    conf_spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    try:
        conf_auc = roc_auc_score(y_test_bin, clf_prob_exceed)
    except ValueError:
        conf_auc = np.nan

    # Persistence baseline for exceedance
    pers_prob_exceed = np.clip(fill_test.to_numpy(), 0, 1)
    pers_pred_bin = (pers_prob_exceed >= threshold).astype(int)

    pers_mae = mean_absolute_error(y_test_bin, pers_prob_exceed)
    pers_mse = mean_squared_error(y_test_bin, pers_prob_exceed)
    pers_brier = brier_score_loss(y_test_bin, pers_prob_exceed)
    pers_ap = average_precision_score(y_test_bin, pers_prob_exceed)

    pers_prec = precision_score(y_test_bin, pers_pred_bin, zero_division=0)
    pers_rec = recall_score(y_test_bin, pers_pred_bin, zero_division=0)
    pers_f1 = f1_score(y_test_bin, pers_pred_bin, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test_bin, pers_pred_bin, labels=[0, 1]).ravel()
    pers_spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    try:
        pers_auc = roc_auc_score(y_test_bin, pers_prob_exceed)
    except ValueError:
        pers_auc = np.nan

    exceedance_metrics = pd.DataFrame(
        {
            "EnsembleClassifier": {
                "MAE": clf_mae,
                "MSE": clf_mse,
                "Brier": clf_brier,
                "PR_AUC": clf_ap,
                "ROC_AUC": clf_auc,
                "Precision": clf_prec,
                "Recall": clf_rec,
                "F1": clf_f1,
                "Specificity": clf_spec,
            },
            "ConformalEnsemble": {
                "MAE": conf_mae,
                "MSE": conf_mse,
                "Brier": conf_brier,
                "PR_AUC": conf_ap,
                "ROC_AUC": conf_auc,
                "Precision": conf_prec,
                "Recall": conf_rec,
                "F1": conf_f1,
                "Specificity": conf_spec,
            },
            "Persistence": {
                "MAE": pers_mae,
                "MSE": pers_mse,
                "Brier": pers_brier,
                "PR_AUC": pers_ap,
                "ROC_AUC": pers_auc,
                "Precision": pers_prec,
                "Recall": pers_rec,
                "F1": pers_f1,
                "Specificity": pers_spec,
            },
        }
    )

    return {
        "continuous_metrics": continuous_metrics,
        "exceedance_metrics": exceedance_metrics,
        "fold_models": fold_models,
        "features": features,
        "conf_threshold": conf_threshold,
        "alpha": alpha,
        "threshold": threshold,
        "conformal_scores_cso": conformal_scores_cso,
        "conformal_scores_safe": conformal_scores_safe,
        "oof_prob": oof_prob,
        "clf_prob_exceed": clf_prob_exceed,
        "y_test": y_test,
        "y_test_bin": y_test_bin,
        "fill_test": fill_test,
        "X_test": X_test,
        "X_train_full": X_train_full,
        "y_train_full": y_train_full,
        "y_train_full_bin": y_train_full_bin,
    }


#### EXAMPLE USAGE

# out = build_cso_model(
#     model_df=model_df,
#     kept_features=kept_features,
#     threshold=0.95,
#     test_frac=0.2,
#     n_splits=5,
#     random_state=42,
#     alpha=0.10,
# )

# continuous_metrics = out["continuous_metrics"]
# exceedance_metrics = out["exceedance_metrics"]
# fold_models = out["fold_models"]
# conf_threshold = out["conf_threshold"]
# conformal_scores_cso = out["conformal_scores_cso"]
# conformal_scores_safe = out["conformal_scores_safe"]