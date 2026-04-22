from pathlib import Path
import yaml
import pandas as pd

from model.helpers.prepare_data import prepare_station_data
from model.helpers.engineer_features import engineer_station_features
from model.helpers.select_features import select_kept_features
from model.helpers.build_models import build_cso_model
from model.helpers.save_models import save_cso_model_artifacts
from model.helpers.evaluate import load_model_metrics

import argparse
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


def main(config_path: str = "config.yaml"):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    paths = config["paths"]
    model_cfg = config["modeling"]

    # Load data
    df_gefs = pd.read_csv(paths["gefs_analysis_csv"])
    df_rain = pd.read_csv(Path(paths["hde_root"]) / "graylingwell/chilgrove_house_rainfall_15min.csv")
    df_grnd = pd.read_csv(Path(paths["hde_root"]) / "graylingwell/graylingwell_groundwaterLevel_15min.csv")
    df_flow = pd.read_csv(paths["southern_water_flow_csv"])
    df_monitors = pd.read_csv(paths["southern_water_monitors_csv"])

    # Config values
    horizons = model_cfg["horizons_hours"]
    threshold = model_cfg["threshold"]
    alpha = model_cfg["alpha"]
    test_frac = model_cfg["test_frac"]
    n_splits = model_cfg["n_splits"]
    random_state = model_cfg["random_state"]

    freq_minutes = model_cfg["freq_minutes"]
    baseline_hours = model_cfg["baseline_hours"]
    percentile_window_hours = model_cfg["percentile_window_hours"]
    surcharge_threshold = model_cfg["surcharge_threshold"]
    high_flow_quantile = model_cfg["high_flow_quantile"]

    model_root = Path(paths["model_root"])
    model_root.mkdir(parents=True, exist_ok=True)

    # Main pipeline
    for _, station_info in df_monitors.loc[
        df_monitors["recently_active"].eq("yes")
        & df_monitors["type"].eq("flow_monitor")
    ].iterrows():

        station_id = str(station_info["id"])

        df_station = prepare_station_data(
            station_info=station_info,
            df_gefs=df_gefs,
            df_rain=df_rain,
            df_grnd=df_grnd,
            df_flow=df_flow,
        )

        for horizon_hours in horizons:
            model_path = model_root / station_id / f"{horizon_hours}hr" / "model_bundle.joblib"
            if model_path.exists():
                continue

            feature_df = engineer_station_features(
                df_station,
                horizon_hours=horizon_hours,
                freq_minutes=freq_minutes,
                baseline_hours=baseline_hours,
                percentile_window_hours=percentile_window_hours,
                surcharge_threshold=surcharge_threshold,
                high_flow_quantile=high_flow_quantile,
            )

            kept_features = select_kept_features(feature_df)
            model_df = feature_df.dropna().reset_index(drop=True)

            out = build_cso_model(
                model_df=model_df,
                kept_features=kept_features,
                threshold=threshold,
                test_frac=test_frac,
                n_splits=n_splits,
                random_state=random_state,
                alpha=alpha,
            )

            save_cso_model_artifacts(
                station_info=station_info,
                horizon_hours=horizon_hours,
                fold_models=out["fold_models"],
                features=out["features"],
                continuous_metrics=out["continuous_metrics"],
                exceedance_metrics=out["exceedance_metrics"],
                conf_threshold=out["conf_threshold"],
                alpha=out["alpha"],
                threshold=out["threshold"],
                conformal_scores_cso=out["conformal_scores_cso"],
                conformal_scores_safe=out["conformal_scores_safe"],
                base_dir=str(model_root),
            )

    # Evaluate once at end
    results = load_model_metrics(str(model_root))
    results["best_model_table"].to_csv(model_root / "best_model_table.csv", index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
    
    
#### EXAMPLE USAGE

# python model_pipeline.py --config config.yaml