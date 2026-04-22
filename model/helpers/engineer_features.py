import numpy as np
import pandas as pd

def engineer_station_features(
    df: pd.DataFrame,
    horizon_hours: float = 6,
    freq_minutes: int = 2,
    baseline_hours: float = 24,
    percentile_window_hours: float = 24,
    surcharge_threshold: float = 0.9,
    high_flow_quantile: float = 0.9,
) -> pd.DataFrame:
    """
    Build forecast-time features for a fixed 2 minute grid.

    Assumptions:
    - df is already filtered to one station
    - df has one row per 2 minute interval
    - precipitation_surface is a forecast series that can be used into the future
    - precipitation_surface_cum is a cumulative rainfall series
    """

    out = df.copy()
    out["time"] = pd.to_datetime(out["time"])
    out = out.sort_values("time").reset_index(drop=True)

    step_hours = freq_minutes / 60.0

    def rows(hours: float) -> int:
        return int(round(hours * 60 / freq_minutes))

    def trailing_roll(s: pd.Series, window_rows: int, how: str):
        r = s.rolling(window_rows, min_periods=window_rows)
        return getattr(r, how)()

    def forward_roll(s: pd.Series, window_rows: int, how: str):
        r = s.iloc[::-1].rolling(window_rows, min_periods=window_rows)
        return getattr(r, how)().iloc[::-1]

    def forward_max_excluding_current(s: pd.Series, window_rows: int):
        future = s.shift(-1)
        return future.iloc[::-1].rolling(window_rows, min_periods=window_rows).max().iloc[::-1]

    def rolling_percentile_of_current(s: pd.Series, window_rows: int):
        def pct(x):
            x = np.asarray(x)
            return np.mean(x <= x[-1])
        return s.rolling(window_rows, min_periods=window_rows).apply(pct, raw=True)

    def slope_from_lag(s: pd.Series, lag_rows: int, lag_hours: float):
        return (s - s.shift(lag_rows)) / lag_hours

    def time_since_last_true(mask: pd.Series):
        idx = np.arange(len(mask), dtype=float)
        last_true = np.where(mask.to_numpy(), idx, np.nan)
        last_true = pd.Series(last_true).ffill().to_numpy()
        out_vals = (idx - last_true) * step_hours
        out_vals[np.isnan(last_true)] = np.nan
        return pd.Series(out_vals, index=mask.index)

    def time_above(mask: pd.Series, window_rows: int):
        return mask.astype(float).rolling(window_rows, min_periods=window_rows).sum() * step_hours

    # Core time windows
    n_h = rows(horizon_hours)
    n_10m = rows(10 / 60)
    n_30m = rows(30 / 60)
    n_1h = rows(1)
    n_3h = rows(3)
    n_6h = rows(6)
    n_24h = rows(24)
    n_72h = rows(72)
    n_7d = rows(24 * 7)
    n_30d = rows(24 * 30)

    # Target: max fill_ratio over the next horizon, excluding the current row
    out["fill_ratio_target"] = forward_max_excluding_current(out["fill_ratio"], n_h)

    # Core state
    out["normalized_flow"] = out["flow_l_s"] / trailing_roll(out["flow_l_s"], rows(baseline_hours), "median")
    out["depth_percentile"] = rolling_percentile_of_current(out["depth_mm"], rows(percentile_window_hours))

    # Short-term dynamics
    out["d_depth_dt"] = out["depth_mm"].diff() / step_hours
    out["d_flow_dt"] = out["flow_l_s"].diff() / step_hours
    out["d_velocity_dt"] = out["velocity_m_s"].diff() / step_hours

    out["depth_slope_10min"] = slope_from_lag(out["depth_mm"], n_10m, 10 / 60)
    out["depth_slope_30min"] = slope_from_lag(out["depth_mm"], n_30m, 30 / 60)
    out["depth_slope_1h"] = slope_from_lag(out["depth_mm"], n_1h, 1)
    out["flow_slope_10min"] = slope_from_lag(out["flow_l_s"], n_10m, 10 / 60)

    out["depth_acceleration"] = (out["d_depth_dt"] - out["d_depth_dt"].shift(n_10m)) / (10 / 60)

    # Lag features
    out["fill_ratio_lag_2min"] = out["fill_ratio"].shift(1)
    out["fill_ratio_lag_10min"] = out["fill_ratio"].shift(n_10m)
    out["fill_ratio_lag_30min"] = out["fill_ratio"].shift(n_30m)
    out["fill_ratio_lag_1h"] = out["fill_ratio"].shift(n_1h)
    out["fill_ratio_lag_3h"] = out["fill_ratio"].shift(n_3h)
    out["depth_lag_10min"] = out["depth_mm"].shift(n_10m)
    out["flow_lag_10min"] = out["flow_l_s"].shift(n_10m)
    out["velocity_lag_10min"] = out["velocity_m_s"].shift(n_10m)

    # Rolling summaries, short
    out["fill_ratio_max_10min"] = trailing_roll(out["fill_ratio"], n_10m, "max")
    out["fill_ratio_max_30min"] = trailing_roll(out["fill_ratio"], n_30m, "max")
    out["fill_ratio_max_1h"] = trailing_roll(out["fill_ratio"], n_1h, "max")
    out["fill_ratio_mean_1h"] = trailing_roll(out["fill_ratio"], n_1h, "mean")
    out["fill_ratio_std_1h"] = trailing_roll(out["fill_ratio"], n_1h, "std")
    out["depth_range_1h"] = trailing_roll(out["depth_mm"], n_1h, "max") - trailing_roll(out["depth_mm"], n_1h, "min")
    out["flow_mean_1h"] = trailing_roll(out["flow_l_s"], n_1h, "mean")

    # Rolling summaries, medium/long
    out["fill_ratio_max_3h"] = trailing_roll(out["fill_ratio"], n_3h, "max")
    out["fill_ratio_max_6h"] = trailing_roll(out["fill_ratio"], n_6h, "max")
    out["fill_ratio_max_24h"] = trailing_roll(out["fill_ratio"], n_24h, "max")
    out["fill_ratio_max_24h"] = trailing_roll(out["fill_ratio"], n_24h, "max")
    out["fill_ratio_mean_24h"] = trailing_roll(out["fill_ratio"], n_24h, "mean")
    out["fill_ratio_max_72h"] = trailing_roll(out["fill_ratio"], n_72h, "max")
    out["fill_ratio_max_7d"] = trailing_roll(out["fill_ratio"], n_7d, "max")
    out["flow_sum_6h"] = trailing_roll(out["flow_l_s"], n_6h, "sum")
    out["flow_sum_24h"] = trailing_roll(out["flow_l_s"], n_24h, "sum")

    # Threshold and persistence
    gt_07 = out["fill_ratio"] > 0.7
    gt_08 = out["fill_ratio"] > 0.8

    out["time_since_fill_ratio_gt_0.7"] = time_since_last_true(gt_07)
    out["time_since_fill_ratio_gt_0.8"] = time_since_last_true(gt_08)
    out["time_above_0.7_last_1h"] = time_above(gt_07, n_1h)
    out["time_above_0.8_last_6h"] = time_above(gt_08, n_6h)
    out["count_exceedances_0.8_last_6h"] = gt_08.astype(int).rolling(n_6h, min_periods=n_6h).sum()
    out["max_fill_ratio_last_1h"] = trailing_roll(out["fill_ratio"], n_1h, "max")

    # Rainfall, short term
    out["rainfall_sum_10min"] = forward_roll(out["precipitation_surface"], n_10m, "sum")
    out["rainfall_sum_30min"] = forward_roll(out["precipitation_surface"], n_30m, "sum")
    out["rainfall_sum_1h"] = forward_roll(out["precipitation_surface"], n_1h, "sum")
    out["rainfall_sum_3h"] = forward_roll(out["precipitation_surface"], n_3h, "sum")
    out["rainfall_sum_6h"] = forward_roll(out["precipitation_surface"], n_6h, "sum")
    out["rainfall_intensity_max_1h"] = forward_roll(out["precipitation_surface"], n_1h, "max")
    out["rainfall_variance_1h"] = forward_roll(out["precipitation_surface"], n_1h, "var")
    out["rainfall_sum_horizon"] = forward_roll(out["precipitation_surface"], n_h, "sum")
    out["rainfall_max_horizon"] = forward_roll(out["precipitation_surface"], n_h, "max")

    # Rainfall, antecedent
    out["rainfall_sum_24h"] = out["precipitation_surface_cum"] - out["precipitation_surface_cum"].shift(n_24h)
    out["rainfall_sum_72h"] = out["precipitation_surface_cum"] - out["precipitation_surface_cum"].shift(n_72h)
    out["rainfall_sum_7d"] = out["precipitation_surface_cum"] - out["precipitation_surface_cum"].shift(n_7d)
    out["rainfall_sum_30d"] = out["precipitation_surface_cum"] - out["precipitation_surface_cum"].shift(n_30d)

    dry_mask = out["precipitation_surface"].fillna(0).eq(0)
    out["antecedent_dry_period"] = time_since_last_true(~dry_mask)

    # Hydrological indices
    out["rain_hde_lag_1h"] = out["rain_hde"].shift(n_1h)
    out["rain_hde_smooth"] = trailing_roll(out["rain_hde"], n_1h, "mean")

    out["grnd_hde_lag_6h"] = out["grnd_hde"].shift(n_6h)
    out["grnd_hde_smooth"] = trailing_roll(out["grnd_hde"], n_6h, "mean")

    # Groundwater trends
    out["d_grnd_hde_dt"] = out["grnd_hde"].diff() / step_hours
    out["grnd_hde_slope_1h"] = (out["grnd_hde"] - out["grnd_hde"].shift(n_1h)) / 1
    out["grnd_hde_slope_6h"] = (out["grnd_hde"] - out["grnd_hde"].shift(n_6h)) / 6
    out["grnd_hde_slope_24h"] = (out["grnd_hde"] - out["grnd_hde"].shift(n_24h)) / 24
    out["grnd_hde_slope_7d"] = (out["grnd_hde"] - out["grnd_hde"].shift(n_7d)) / (24 * 7)
    out["grnd_hde_slope_30d"] = (out["grnd_hde"] - out["grnd_hde"].shift(n_30d)) / (24 * 30)
    out["grnd_hde_acceleration"] = (
    out["d_grnd_hde_dt"] - out["d_grnd_hde_dt"].shift(n_1h)) / 1

    # Interactions
    out["fill_ratio_x_rainfall_1h"] = out["fill_ratio"] * out["rainfall_sum_1h"]
    out["depth_x_rainfall_1h"] = out["depth_mm"] * out["rainfall_sum_1h"]
    out["flow_x_rainfall"] = out["flow_l_s"] * out["precipitation_surface"]
    out["grnd_hde_x_rainfall"] = out["grnd_hde"] * out["precipitation_surface"]
    out["velocity_x_depth"] = out["velocity_m_s"] * out["depth_mm"]

    # Temporal features at the forecast horizon, not the current row
    horizon_time = out["time"] + pd.to_timedelta(horizon_hours, unit="h")
    hour = horizon_time.dt.hour + horizon_time.dt.minute / 60.0
    day_of_year = horizon_time.dt.dayofyear

    out["hour_of_day_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_of_day_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["day_of_week"] = horizon_time.dt.dayofweek
    out["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    out["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Regime indicators
    # Use future rainfall over the full prediction horizon
    out["is_raining"] = (forward_roll(out["precipitation_surface"], n_h, "sum") > 0).astype(int)

    flow_threshold = out["flow_l_s"].rolling(n_24h, min_periods=n_24h).quantile(high_flow_quantile)
    out["is_high_flow"] = (out["flow_l_s"] > flow_threshold).astype(int)

    out["is_rising_depth"] = (out["depth_slope_1h"] > 0).astype(int)
    out["is_surcharged"] = (out["fill_ratio"] > surcharge_threshold).astype(int)

    return out


#### EXAMPLE USAGE

# feature_df = engineer_station_features(
#     df_station,
#     horizon_hours=6,
#     freq_minutes=2,
#     baseline_hours=24,
#     percentile_window_hours=24,
#     surcharge_threshold=0.9,
#     high_flow_quantile=0.9,
# )