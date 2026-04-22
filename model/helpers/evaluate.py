from pathlib import Path
import numpy as np
import pandas as pd

LOWER_BETTER = {"MAE", "MSE", "Brier"}
HIGHER_BETTER = {"PR_AUC", "ROC_AUC", "Precision", "Recall", "F1", "Specificity"}
ALL_COMPARE_METRICS = sorted(LOWER_BETTER | HIGHER_BETTER)


def load_model_metrics(base_dir: str = "model/model_weights"):
    """
    Load model/model_weights/{station_id}/{horizon}hr/metrics.csv files and build:
      1) average % improvement vs Persistence by horizon (mean, std)
      2) % of times each model beat Persistence by horizon
      3) best model table by station_id x horizon

    Best model is chosen by average rank across available exceedance metrics
    (lower is better for MAE/MSE/Brier; higher is better for the rest).
    """
    base_dir = Path(base_dir)

    long_parts = []
    files = sorted(base_dir.glob("*/*hr/metrics.csv"))

    for metrics_path in files:
        station_id = metrics_path.parents[1].name
        horizon_name = metrics_path.parent.name
        try:
            horizon_hours = int(horizon_name.removesuffix("hr"))
        except ValueError:
            continue

        df = pd.read_csv(metrics_path, index_col=[0, 1])
        df.index = pd.MultiIndex.from_tuples(df.index, names=["section", "metric"])
        df = df.apply(pd.to_numeric, errors="coerce")

        long_df = (
            df.stack(dropna=False, future_stack=True)
            .rename("value")
            .reset_index()
            .rename(columns={"level_2": "model"})
        )
        long_df["station_id"] = station_id
        long_df["horizon_hours"] = horizon_hours
        long_parts.append(long_df)

    combined_long = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()

    if combined_long.empty:
        return {
            "combined_long": combined_long,
            "comparison_df": pd.DataFrame(),
            "avg_diff_summary": pd.DataFrame(),
            "beat_rate_overall": pd.DataFrame(),
            "beat_rate_by_metric": pd.DataFrame(),
            "best_model_table": pd.DataFrame(),
        }

    exceed = combined_long.loc[combined_long["section"].eq("exceedance_metrics")].copy()

    # ------------------------------------------------------------------
    # 1) % difference vs Persistence, by horizon, with std
    #    Positive = better than Persistence
    # ------------------------------------------------------------------
    comparison_rows = []

    for (station_id, horizon_hours, metric), g in exceed.groupby(["station_id", "horizon_hours", "metric"]):
        if metric not in ALL_COMPARE_METRICS:
            continue

        vals = g.set_index("model")["value"]
        if "Persistence" not in vals:
            continue

        base = vals.get("Persistence", np.nan)
        if pd.isna(base) or base == 0:
            continue

        for model in ["EnsembleClassifier", "ConformalEnsemble"]:
            mv = vals.get(model, np.nan)
            if pd.isna(mv):
                continue

            if metric in LOWER_BETTER:
                pct_diff = 100.0 * (base - mv) / abs(base)
                beat = mv < base
            else:
                pct_diff = 100.0 * (mv - base) / abs(base)
                beat = mv > base

            comparison_rows.append(
                {
                    "station_id": station_id,
                    "horizon_hours": horizon_hours,
                    "metric": metric,
                    "model": model,
                    "persistence": base,
                    "model_value": mv,
                    "pct_diff": pct_diff,
                    "beat_persistence": bool(beat),
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)

    avg_diff_summary = (
        comparison_df.groupby(["horizon_hours", "model", "metric"], as_index=False)
        .agg(
            mean_pct_diff=("pct_diff", "mean"),
            std_pct_diff=("pct_diff", "std"),
            n=("pct_diff", "count"),
        )
        .sort_values(["horizon_hours", "model", "metric"])
    )

    beat_rate_overall = (
        comparison_df.groupby(["horizon_hours", "model"], as_index=False)
        .agg(beat_rate_pct=("beat_persistence", lambda s: 100.0 * s.mean()), n=("beat_persistence", "count"))
        .sort_values(["horizon_hours", "model"])
    )

    beat_rate_by_metric = (
        comparison_df.groupby(["horizon_hours", "model", "metric"], as_index=False)
        .agg(beat_rate_pct=("beat_persistence", lambda s: 100.0 * s.mean()), n=("beat_persistence", "count"))
        .sort_values(["horizon_hours", "model", "metric"])
    )

    # ------------------------------------------------------------------
    # 4) Best model table by station x horizon
    #    Best = lowest average rank across available exceedance metrics
    # ------------------------------------------------------------------
    def best_model_for_group(g: pd.DataFrame):
        score_lists = {
            "Persistence": [],
            "EnsembleClassifier": [],
            "ConformalEnsemble": [],
        }

        for metric in ALL_COMPARE_METRICS:
            if metric not in g["metric"].unique():
                continue

            vals = g.loc[g["metric"].eq(metric)].set_index("model")["value"]
            if not {"Persistence", "EnsembleClassifier", "ConformalEnsemble"}.issubset(vals.index):
                continue

            tri = vals[["Persistence", "EnsembleClassifier", "ConformalEnsemble"]].astype(float)
            if tri.isna().any():
                continue

            ascending = metric in LOWER_BETTER
            ranks = tri.rank(ascending=ascending, method="average")

            for model in score_lists:
                score_lists[model].append(float(ranks[model]))

        if all(len(v) == 0 for v in score_lists.values()):
            return np.nan

        avg_rank = {
            model: (np.mean(vals) if len(vals) else np.inf)
            for model, vals in score_lists.items()
        }
        return min(avg_rank, key=avg_rank.get)

    best_model_table = (
        exceed.groupby(["station_id", "horizon_hours"])
        .apply(best_model_for_group)
        .rename("best_model")
        .reset_index()
        .pivot(index="station_id", columns="horizon_hours", values="best_model")
    )

    return {
        "combined_long": combined_long,
        "comparison_df": comparison_df,
        "avg_diff_summary": avg_diff_summary,
        "beat_rate_overall": beat_rate_overall,
        "beat_rate_by_metric": beat_rate_by_metric,
        "best_model_table": best_model_table,
    }


#### EXAMPLE USAGE

# results = load_model_metrics("model/model_weights")

# combined_long = results["combined_long"]
# comparison_df = results["comparison_df"]
# avg_diff_summary = results["avg_diff_summary"]
# beat_rate_overall = results["beat_rate_overall"]
# beat_rate_by_metric = results["beat_rate_by_metric"]
# best_model_table = results["best_model_table"]

# print(avg_diff_summary)
# print(beat_rate_overall)
# print(best_model_table)

# best_model_table.to_csv('model/model_weights/best_model_table.csv', index=True)