import xarray as xr
import pandas as pd
import yaml
from pathlib import Path


def process_gefs_analysis(config_path="config.yaml"):
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    gefs_cfg = config["gefs"]

    lat, lon = gefs_cfg["coordinates"][0]
    output_dir = Path(gefs_cfg["analysis_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "gefs_analysis.csv"

    # Load dataset
    ds = xr.open_zarr("https://data.dynamical.org/noaa/gefs/analysis/latest.zarr")

    # Select location
    da = ds["precipitation_surface"].sel(
        latitude=lat,
        longitude=lon,
        method="nearest"
    )

    # Convert to DataFrame
    df = da.to_dataframe(name="precipitation_surface").reset_index()

    # Compute statistics
    mean = df["precipitation_surface"].mean()
    std = df["precipitation_surface"].std()

    # Derived columns
    df["precipitation_surface_z"] = (df["precipitation_surface"] - mean) / std
    df["precipitation_surface_cum"] = df["precipitation_surface_z"].cumsum()

    # Save to CSV
    df[["time", "precipitation_surface", "precipitation_surface_cum"]].to_csv(
        output_path,
        index=False
    )

    return df