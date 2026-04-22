import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def save_cso_model_artifacts(
    *,
    station_info,
    horizon_hours: int,
    fold_models,
    features,
    continuous_metrics: pd.DataFrame,
    exceedance_metrics: pd.DataFrame,
    conf_threshold: float,
    alpha: float,
    threshold: float,
    conformal_scores_cso,
    conformal_scores_safe,
    base_dir: str = "model/model_weights",
):
    if isinstance(station_info, pd.DataFrame):
        if len(station_info) != 1:
            raise ValueError("station_info DataFrame must contain exactly one row.")
        station_info = station_info.iloc[0]

    if isinstance(station_info, pd.Series):
        station_info_dict = station_info.to_dict()
    elif isinstance(station_info, dict):
        station_info_dict = dict(station_info)
    else:
        raise TypeError("station_info must be a pandas Series, one-row DataFrame, or dict.")

    station_id = str(station_info_dict["id"])
    horizon_dir = Path(base_dir) / station_id / f"{int(horizon_hours)}hr"
    station_dir = Path(base_dir) / station_id

    horizon_dir.mkdir(parents=True, exist_ok=True)
    station_dir.mkdir(parents=True, exist_ok=True)

    with open(station_dir / "station_info.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(station_info_dict), f, indent=2)

    with open(horizon_dir / "features.json", "w", encoding="utf-8") as f:
        json.dump(list(features), f, indent=2)

    metadata = {
        "station_id": station_id,
        "horizon_hours": int(horizon_hours),
        "threshold": float(threshold),
        "alpha": float(alpha),
        "conformal_threshold": float(conf_threshold),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_models": len(fold_models),
        "performance_sentence_source": (
            "ConformalEnsemble" if "ConformalEnsemble" in exceedance_metrics.columns else "EnsembleClassifier"
        ),
    }

    with open(horizon_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(metadata), f, indent=2)

    pd.concat(
        {
            "continuous_metrics": continuous_metrics,
            "exceedance_metrics": exceedance_metrics,
        },
        axis=0,
    ).to_csv(horizon_dir / "metrics.csv")

    bundle = {
        "station_id": station_id,
        "horizon_hours": int(horizon_hours),
        "threshold": float(threshold),
        "alpha": float(alpha),
        "conformal_threshold": float(conf_threshold),
        "features": list(features),
        "fold_models": fold_models,
        "continuous_metrics": continuous_metrics,
        "exceedance_metrics": exceedance_metrics,
        "conformal_scores_cso": list(np.asarray(conformal_scores_cso, dtype=float)),
        "conformal_scores_safe": list(np.asarray(conformal_scores_safe, dtype=float)),
        "metadata": metadata,
    }

    joblib.dump(bundle, horizon_dir / "model_bundle.joblib", compress=3)
    return horizon_dir


#### EXAMPLE USAGE

# save_cso_model_artifacts(
#     station_info=station_info,
#     horizon_hours=6,
#     fold_models=fold_models,
#     features=features,
#     continuous_metrics=continuous_metrics,
#     exceedance_metrics=exceedance_metrics,
#     conf_threshold=conf_threshold,
#     alpha=alpha,
#     threshold=threshold,
#     conformal_scores_cso=conformal_scores_cso,
#     conformal_scores_safe=conformal_scores_safe,
#     base_dir="model",
# )