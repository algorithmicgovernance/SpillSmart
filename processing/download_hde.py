import logging
import requests
import numpy as np
import pandas as pd
import re
import os
import glob
from typing import List, Dict

logger = logging.getLogger(__name__)

def station_has_15min_flow(station_guid, session=None, timeout=10):
    base = "https://environment.data.gov.uk/hydrology/id"
    url = f"{base}/stations/{station_guid}/measures?observedProperty=waterFlow"
    get = session.get if session is not None else requests.get

    try:
        r = get(url, timeout=timeout)
        if r.status_code == 404:
            return False
        r.raise_for_status()
    except Exception:
        return False

    measures = r.json().get("items", []) or []
    return any(
        str(m.get("period")) == "900" or "-t-900-" in (m.get("@id", "") or "")
        for m in measures
    )


def _download_hde_measure_data(measure_id, label, output_prefix, file_prefix, observed_property, period_label, session=None):

    # build target folder + filename up-front so we can skip if file exists
    folder = os.path.join("data", "hde", output_prefix)
    fname = os.path.join(
        folder,
        f"{file_prefix}_{observed_property}_{period_label}.csv"
    )
    
    existing_df = None
    last_timestamp = None
    
    if os.path.exists(fname):
        try:
            existing_df = pd.read_csv(fname, parse_dates=["dateTime"], index_col="dateTime")
            if not existing_df.empty:
                last_timestamp = existing_df.index.max()
                logger.debug(f"Updating existing file from {last_timestamp}: {fname}")
        except Exception:
            logger.warning(f"Failed to read existing file, will overwrite: {fname}")
            
    all_readings = []
    offset = 0
    limit = 10000
    get = session.get if session is not None else requests.get
    
    while True:
        if last_timestamp is not None:
            since = last_timestamp.isoformat()
            url = f"{measure_id}/readings?since={since}&_limit={limit}"
            logger.debug(f"Fetching new data since {last_timestamp}")
        else:
            url = f"{measure_id}/readings?_limit={limit}&_offset={offset}"
            if offset % (limit * 5) == 0:
                logger.debug(f"Fetching historical data (offset={offset})")
    
        resp = get(url, timeout=30)
    
        if resp.status_code == 404:
            logger.warning(f"No readings found for {label}")
            break
    
        resp.raise_for_status()
        items = resp.json().get("items", [])
    
        if not items:
            break
    
        all_readings.extend(items)
    
        logger.debug(f"Fetched {len(items)} rows (total so far: {len(all_readings)})")
    
        if last_timestamp is not None:
            break
    
        if len(items) < limit:
            break
    
        offset += limit

    if not all_readings:
        logger.debug(f"No new data returned for {label}")
        return

    df = pd.DataFrame(all_readings)
    if "dateTime" in df.columns:
        df["dateTime"] = pd.to_datetime(df["dateTime"])
        df = df.set_index("dateTime").sort_index()

    df = df[[c for c in df.columns if c in ["value", "valueQualifier"]]]
    
    if last_timestamp is not None:
        df = df[df.index > last_timestamp]
    
        if df.empty:
            logger.debug(f"No new rows to append for {fname}")
            return
    
    os.makedirs(folder, exist_ok=True)
    
    if existing_df is not None and not existing_df.empty:
        df = pd.concat([existing_df, df])
        df = df[~df.index.duplicated(keep="last")].sort_index()
    
    df.to_csv(fname)
    logger.debug(f"File now has {len(df)} rows: {fname}")
    
    
def download_hde_river_data_15min(station_guid, observed_property="waterFlow", output_prefix="", session=None):
    base = "https://environment.data.gov.uk/hydrology/id"
    measures_url = f"{base}/stations/{station_guid}/measures?observedProperty={observed_property}"
    get = session.get if session is not None else requests.get
    r = get(measures_url)

    if r.status_code == 404:
        logger.warning(f"Warning: Station {station_guid} not found for {observed_property}")
        return

    r.raise_for_status()
    measures = r.json().get("items", [])

    if not measures:
        logger.warning(f"Warning: No {observed_property} measures found for station {station_guid}")
        return

    quarter_hour_measures = [m for m in measures if str(m.get("period")) == "900" or "-t-900-" in m.get("@id", "")]
    if not quarter_hour_measures:
        logger.warning(f"Warning: No 15 min {observed_property} measures for station {station_guid}")
        return

    logger.debug(f"Found {len(quarter_hour_measures)} {observed_property} measure(s) at 15 min")

    for m in quarter_hour_measures:
        measure_id = m["@id"].replace("http://", "https://")
        label = m.get("label", measure_id.split("/")[-1])

        station_name = (
            m.get("station", {})
             .get("label", "station")
             .lower()
             .replace(" ", "_")
        )

        _download_hde_measure_data(
            measure_id,
            label,
            output_prefix,        # folder stays chalk-based
            station_name,         # filename prefix
            observed_property,
            "15min",
            session=session
        )


def download_hde_rainfall_15min(station_guid, output_prefix="", session=None):
    base = "https://environment.data.gov.uk/hydrology/id"
    observed_property = "rainfall"
    measures_url = f"{base}/stations/{station_guid}/measures?observedProperty=rainfall"
    get = session.get if session is not None else requests.get
    r = get(measures_url)

    if r.status_code == 404:
        logger.warning(f"Warning: Station {station_guid} not found for rainfall")
        return

    r.raise_for_status()
    measures = r.json().get("items", [])

    if not measures:
        logger.warning(f"Warning: No rainfall measures found for station {station_guid}")
        return

    quarter_hour_measures = [m for m in measures if str(m.get("period")) == "900" or "-t-900-" in m.get("@id", "")]
    if not quarter_hour_measures:
        logger.warning(f"Warning: No 15 min rainfall measures for station {station_guid}")
        return

    logger.debug(f"Found {len(quarter_hour_measures)} rainfall measure(s) at 15 min")

    for m in quarter_hour_measures:
        measure_id = m["@id"].replace("http://", "https://")
        label = m.get("label", measure_id.split("/")[-1])

        station_name = (
            m.get("station", {})
             .get("label", "station")
             .lower()
             .replace(" ", "_")
        )

        _download_hde_measure_data(
            measure_id,
            label,
            output_prefix,        # folder stays chalk-based
            station_name,         # filename prefix
            observed_property,
            "15min",
            session=session
        )


def download_hde_groundwater_15min(station_guid, output_prefix="", session=None):
    base = "https://environment.data.gov.uk/hydrology/id"
    observed_property = "groundwaterLevel"
    measures_url = f"{base}/stations/{station_guid}/measures?observedProperty={observed_property}"
    get = session.get if session is not None else requests.get
    r = get(measures_url)

    if r.status_code == 404:
        logger.warning(f"Warning: Station {station_guid} not found for groundwater")
        return

    r.raise_for_status()
    measures = r.json().get("items", [])

    if not measures:
        logger.warning(f"Warning: No groundwater measures found for station {station_guid}")
        return

    # Try 15 min first
    measures_15 = [m for m in measures if str(m.get("period")) == "900" or "-t-900-" in m.get("@id", "")]
    
    if measures_15:
        selected_measures = measures_15
        period_label = "15min"
        logger.debug(f"Using 15 min groundwater data ({len(measures_15)} measure(s))")
    
    else:
        # fallback to hourly
        measures_60 = [m for m in measures if str(m.get("period")) == "3600"]
    
        if measures_60:
            selected_measures = measures_60
            period_label = "60min"
            logger.debug(f"No 15 min data — using hourly groundwater ({len(measures_60)} measure(s))")
        else:
            logger.warning(f"No suitable groundwater measures (15min or hourly) for station {station_guid}")
            return

    for m in selected_measures:
        measure_id = m["@id"].replace("http://", "https://")
        label = m.get("label", measure_id.split("/")[-1])

        station_name = (
            m.get("station", {})
             .get("label", "station")
             .lower()
             .replace(" ", "_")
        )

        _download_hde_measure_data(
            measure_id,
            label,
            output_prefix,        # folder stays chalk-based
            station_name,         # filename prefix
            observed_property,
            period_label,
            session=session
        )


def run_hde_downloads(
    rainfall_station=np.nan,
    river_station=np.nan,
    groundwater_station=np.nan,
    output_prefix="",
    session=None
):
    """Run all three 15 min downloads only when station ids are provided."""
    if not pd.isna(rainfall_station):
        logger.debug("Downloading Rainfall (15 min)...")
        download_hde_rainfall_15min(rainfall_station, output_prefix=output_prefix, session=session)

    if not pd.isna(river_station):
        logger.debug("Downloading River Data (15 min)...")
        download_hde_river_data_15min(
            river_station,
            observed_property="waterFlow",
            output_prefix=output_prefix,
            session=session
        )

    if not pd.isna(groundwater_station):
        logger.debug("Downloading Groundwater Data (15 min)...")
        download_hde_groundwater_15min(groundwater_station, output_prefix=output_prefix, session=session)


# -----------------------
# Processing code added as functions below
# -----------------------

from concurrent.futures import ThreadPoolExecutor, as_completed


def build_chalk_rows(
    chalk_results_csv_path: str = 'analysis/results/chalk_rivers_with_catchment.csv',
    measurement: str = 'groundwater',
    chalk_df: pd.DataFrame = None,
) -> List[Dict]:
    """
    Read the chalk_rivers_with_catchment CSV or accept a DataFrame and build the rows structure
    expected by the worker tasks.

    If chalk_df is provided it is used directly; otherwise chalk_results_csv_path is read.
    """
    measurement = measurement.lower()
    if measurement not in ('groundwater', 'rainfall', 'flow'):
        raise ValueError("measurement must be 'groundwater', 'rainfall' or 'flow'")

    # select the appropriate boolean column based on measurement argument
    if measurement == 'groundwater':
        flag_col = 'is_groundwater'
    elif measurement == 'rainfall':
        flag_col = 'is_rainfall'
    else:  # flow
        flag_col = 'is_flow'

    if chalk_df is None:
        if not os.path.exists(chalk_results_csv_path):
            raise FileNotFoundError(f"Chalk results CSV not found: {chalk_results_csv_path}")
        df = pd.read_csv(chalk_results_csv_path)
    else:
        # use the provided DataFrame copy to avoid mutating caller's object
        df = chalk_df.copy()

    # chalk_ids: flow stations that are chalk and have distance 0
    chalk_ids = df[df['is_flow'] & (df['distance_m'] == 0)]['chalk_station_index']

    # select the appropriate boolean column based on measurement argument
    candidates_df = df[df[flag_col]]

    # only consider rows that relate to chalk stations of interest
    candidates_df = candidates_df[candidates_df['chalk_station_index'].isin(chalk_ids)]

    rows = []
    for idx, group in candidates_df.groupby('chalk_station_index'):
        ordered = group.assign(_up=group['upstream'].astype(bool)).sort_values(by=['_up', 'distance_m'], ascending=[False, True]).drop(columns=['_up'])
        candidate_dicts = ordered.to_dict(orient='records')
        # create a filesystem-friendly prefix from the chalk station name
        name = str(candidate_dicts[0].get('chalk_station_name', 'station')).lower().replace(' ', '_')
        rows.append({
            'chalk_station_index': idx,
            'chalk_station_name': candidate_dicts[0].get('chalk_station_name', ''),
            'candidates': candidate_dicts,
            'output_prefix': name,     # base name used as folder name
            'measurement': measurement,
        })
    return rows


def _task_for_row(row):
    """
    Worker task to try candidate stations for a chalk row.

    Behaviour depends on row['measurement'] which must be 'groundwater' or 'rainfall'.
    Returns output_prefix (string) at completion (whether or not data found).
    """
    output_prefix = row['output_prefix']
    measurement = row.get('measurement', 'groundwater')

    chalk_flow_station = (
        row['candidates'][0].get('chalk_station_id')
        if row.get('candidates') else None
    )

    if chalk_flow_station:
        with requests.Session() as sess:
            if not station_has_15min_flow(chalk_flow_station, session=sess):
                logging.getLogger(__name__).info(
                    "Skipping chalk %s entirely: no 15-min flow data exists",
                    row['chalk_station_index']
                )
                return output_prefix

    # determine observed_property and the folder where files will be written
    folder = os.path.join("data", "hde", output_prefix)
    if measurement == 'groundwater':
        observed_property = 'groundwaterLevel'
    elif measurement == 'flow':
        observed_property = 'waterFlow'
    else:  # rainfall
        observed_property = 'rainfall'

    # pattern to match any file for this measurement inside the folder
    pattern = os.path.join(folder, f"*_{observed_property}_*.csv")

    # helper to find a matching file (returns first match or None)
    def find_matching():
        matches = glob.glob(pattern)
        return matches[0] if matches else None

    # early skip if expected file already exists (avoid any network calls)
    overwrite = bool(row.get('overwrite', False))
    existing = find_matching()
    if overwrite and existing:
        try:
            for f in glob.glob(pattern):
                os.remove(f)
            logging.getLogger(__name__).info(
                "Overwriting existing file(s) for %s (%s): %s",
                row['chalk_station_index'], measurement, pattern
            )
        except Exception:
            logging.getLogger(__name__).exception("Failed removing existing file(s): %s", pattern)
        existing = None

    if existing and not overwrite:
        logging.getLogger(__name__).info(
            "Updating existing file for %s (%s): %s",
            row['chalk_station_index'], measurement, existing
        )

    with requests.Session() as sess:
        for cand in row['candidates']:
            # station id for the candidate (both rainfall and groundwater use the same 'station_id' column)
            station_id = cand.get('station_id')
            # chalk flow id is the chalk station (used for river_station argument)
            flow_id = cand.get('chalk_station_id') or np.nan

            # attempt download for this candidate depending on measurement
            try:
                if measurement == 'groundwater':
                    run_hde_downloads(
                        rainfall_station=np.nan,
                        river_station=flow_id,
                        groundwater_station=station_id,
                        output_prefix=output_prefix,
                        session=sess
                    )
                elif measurement == 'flow':
                    # try station_id as the river (flow) station
                    run_hde_downloads(
                        rainfall_station=np.nan,
                        river_station=station_id,
                        groundwater_station=np.nan,
                        output_prefix=output_prefix,
                        session=sess
                    )
                else:  # rainfall
                    run_hde_downloads(
                        rainfall_station=station_id,
                        river_station=np.nan,
                        groundwater_station=np.nan,
                        output_prefix=output_prefix,
                        session=sess
                    )

            except Exception as e:
                logging.getLogger(__name__).exception(
                    "Error downloading for %s station %s (chalk %s): %s",
                    measurement, station_id, row['chalk_station_index'], e
                )

            # check if a matching file (for this measurement) was created
            found = find_matching()
            if found:
                logger.info(f"Updated {measurement} data: {found}")
                return output_prefix
            else:
                logging.getLogger(__name__).info(
                    "No 15min %s file from station %s, trying next candidate (if any).",
                    measurement, station_id
                )

    # if we get here, none of the candidates produced data
    logging.getLogger(__name__).warning(
        "No 15min %s data found for chalk station %s after trying %d candidates.",
        measurement, row['chalk_station_index'], len(row['candidates'])
    )
    return output_prefix


def run_hde_downloads_pipeline(
    max_workers=1,
    chalk_results_csv_path='processing/analysis/results/chalk_rivers_with_elevation.csv',
    chalk_results_df: pd.DataFrame = None,
    measurements=('flow', 'groundwater', 'rainfall'),
    overwrite: bool = False,
    target_chalk_id=None,
):
    """
    Process chalk catchments using candidate stations for selected measurements.

    Pass either chalk_results_csv_path or chalk_results_df. If chalk_results_df is provided it is used.
    """

    # normalise to lowercase set
    measurements = {m.lower() for m in measurements}  
    valid = {'groundwater', 'rainfall', 'flow'}
    invalid = measurements - valid
    if invalid:
        raise ValueError(f"Invalid measurement(s): {invalid}. Valid options: {valid}")

    # configure root logger (unchanged) ...
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # build rows only for selected measurements
    rows = []
    for measurement in measurements:
        try:
            measurement_rows = build_chalk_rows(
                chalk_results_csv_path=chalk_results_csv_path,
                measurement=measurement,
                chalk_df=chalk_results_df,
            )
        except Exception:
            logging.getLogger(__name__).exception("Failed building rows for %s", measurement)
            measurement_rows = []
    
        # Apply filter here instead
        if target_chalk_id is not None:
            measurement_rows = [
                r for r in measurement_rows
                if r['chalk_station_index'] == target_chalk_id
            ]
    
        for r in measurement_rows:
            r['measurement'] = measurement
            r['overwrite'] = overwrite 
    
        rows.extend(measurement_rows)

    if not rows:
        logging.getLogger(__name__).warning("No chalk rows to process (found 0).")
        return []
        
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(_task_for_row, r): r for r in rows}
        
        for fut in as_completed(futures):
            row = futures[fut]
        
            try:
                name = fut.result()
            except Exception as e:
                logger.exception(f"Task failed: {e}")


#### EXAMPLE USE ####

# from processing.download_hde import run_hde_downloads_pipeline
# run_hde_downloads_pipeline(measurements=['rainfall'])
# run_hde_downloads_pipeline(target_chalk_id=291731) # Graylingwell