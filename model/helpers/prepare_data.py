import pandas as pd


def prepare_station_data(
    station_info: pd.Series,
    df_gefs: pd.DataFrame,
    df_rain: pd.DataFrame,
    df_grnd: pd.DataFrame,
    df_flow: pd.DataFrame,
    *,
    freq: str = "2min",
    spline_limit: int = 5040,
) -> pd.DataFrame:
    """
    Prepare one station-level time series dataframe.

    This is station-specific, but horizon-independent.
    It is meant to be called inside a future loop over station_id, then horizon.
    """
    station_id = station_info["id"]
    alt_station_id = station_info.get("alt_id")

    station_ids = [station_id]
    if pd.notna(alt_station_id):
        station_ids.append(alt_station_id)

    # Base flow data
    df_station = df_flow.loc[df_flow["station_id"].isin(station_ids)].copy()
    df_station = df_station.drop(
        columns=["source_file", "station_id_original", "station_id"],
        errors="ignore",
    )

    df_station["time"] = pd.to_datetime(df_station["time"])

    # Remove duplicate times, keep only non negative depth, sort by time
    df_station = (
        df_station.loc[~df_station["time"].duplicated(keep=False)]
        .loc[df_station["depth_mm"].ge(0)]
        .sort_values("time")
        .reset_index(drop=True)
    )

    # Build a complete regular time grid
    full_time = pd.date_range(
        start=df_station["time"].min(),
        end=df_station["time"].max(),
        freq=freq,
    )

    df_station = (
        df_station.set_index("time")
        .reindex(full_time)
        .rename_axis("time")
        .reset_index()
    )

    # GEFS join
    df_gefs = df_gefs.copy()
    df_gefs["time"] = pd.to_datetime(df_gefs["time"])
    df_gefs = df_gefs.drop_duplicates("time")

    df_station = df_station.merge(
        df_gefs[["time", "precipitation_surface", "precipitation_surface_cum"]],
        on="time",
        how="left",
    )

    # Rain
    rain_hde = df_rain.rename(columns={"dateTime": "time", "value": "rain_hde"}).copy()
    rain_hde["time"] = pd.to_datetime(rain_hde["time"])
    rain_hde = rain_hde.drop_duplicates("time")

    # Groundwater
    grnd_hde = df_grnd.rename(columns={"dateTime": "time", "value": "grnd_hde"}).copy()
    grnd_hde["time"] = pd.to_datetime(grnd_hde["time"])
    grnd_hde = grnd_hde.drop_duplicates("time")

    df_station = (
        df_station.merge(rain_hde, on="time", how="left")
        .merge(grnd_hde, on="time", how="left")
        .sort_values("time")
        .reset_index(drop=True)
    )

    # Short-gap interpolation only
    rain_cols = [
        "precipitation_surface",
        "precipitation_surface_cum",
        "rain_hde",
        "grnd_hde",
    ]

    df_station[rain_cols] = df_station[rain_cols].interpolate(
        method="spline",
        order=3,
        limit=spline_limit,
        limit_area="inside",
    )

    # Fill ratio
    df_station["fill_ratio"] = df_station["depth_mm"] / station_info["pipe_diameter_mm"]

    return df_station


#### EXAMPLE USAGE

# df_station = prepare_station_data(
#     station_info=station_info,
#     df_gefs=df_gefs,
#     df_rain=df_rain,
#     df_grnd=df_grnd,
#     df_flow=df_flow,
# )