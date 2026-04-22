from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import yaml
from tqdm import tqdm


CONFIG_PATH = Path("config.yaml")


CHICHESTER_INTERIM_RE = re.compile(
    r"^chichester\s+weekly\s+report\s*-\s*interim\s+(?P<spec>[\d,\s&-]+)$",
    re.IGNORECASE,
)
P_FOLDER_RE = re.compile(r"^p(?P<num>\d+)$", re.IGNORECASE)
P_RANGE_CONTAINER_RE = re.compile(r"^p(?P<start>\d+)-(?P<end>\d+)$", re.IGNORECASE)

IDENTIFIER_RE = re.compile(r"^\*\*IDENTIFIER:\s*\d+,(.+)$", re.MULTILINE)

RANGE_12_RE = re.compile(r"(?P<start>\d{12})\s+(?P<end>\d{12})\s+(?P<interval>\d+)")
RANGE_10_RE = re.compile(r"(?P<start>\d{10})\s+(?P<end>\d{10})\s+(?P<interval>\d+)")


def load_config(config_path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_southern_water_paths(config: dict[str, Any]) -> dict[str, Path]:
    sw = config["southern-water"]
    return {
        "raw_downloads_dir": Path(sw["raw_downloads_dir"]),
        "unzipped_dir": Path(sw["unzipped_dir"]),
        "combined_dir": Path(sw["combined_dir"]),
        "processed_dir": Path(sw["processed_dir"]),
        "monitors_path": Path(sw["monitors_path"]),
    }


def unzip_archives(downloads_dir: Path, output_dir: Path) -> None:
    if not downloads_dir.exists():
        raise FileNotFoundError(f"Downloads directory does not exist: {downloads_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(downloads_dir.glob("*.zip"))
    if not zip_files:
        logging.warning("No zip files found in %s", downloads_dir)
        return

    for zip_path in zip_files:
        target_dir = output_dir / zip_path.stem
        target_dir.mkdir(parents=True, exist_ok=True)

        logging.info("Extracting %s -> %s", zip_path.name, target_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)


def _read_text(path: Path) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin1"):
        try:
            return path.read_text(encoding=enc, errors="strict")
        except UnicodeDecodeError:
            pass
    return path.read_text(encoding="utf-8", errors="replace")


def _expand_int_spec(spec: str) -> list[int]:
    parts = re.findall(r"\d+\s*-\s*\d+|\d+", spec)
    values: list[int] = []

    for part in parts:
        part = part.replace(" ", "")
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start_i = int(start_s)
            end_i = int(end_s)
            if end_i < start_i:
                raise ValueError(f"Invalid range: {part}")
            values.extend(range(start_i, end_i + 1))
        else:
            values.append(int(part))

    return list(dict.fromkeys(values))


def _extract_output_ids(folder_path: Path) -> list[str]:
    name = folder_path.name

    m = CHICHESTER_INTERIM_RE.fullmatch(name)
    if m:
        return [str(i) for i in _expand_int_spec(m.group("spec"))]

    m = P_FOLDER_RE.fullmatch(name)
    if m:
        return [str(int(m.group("num")))]

    raise ValueError(f"Could not extract output ids from folder name: {name}")


def _discover_processing_folders(raw_root: Path) -> list[Path]:
    roots: list[Path] = []

    for top in sorted(raw_root.iterdir()):
        if not top.is_dir():
            continue

        if P_RANGE_CONTAINER_RE.fullmatch(top.name):
            for child in sorted(top.iterdir()):
                if child.is_dir() and P_FOLDER_RE.fullmatch(child.name):
                    roots.append(child)
            continue

        if CHICHESTER_INTERIM_RE.fullmatch(top.name) or P_FOLDER_RE.fullmatch(top.name):
            roots.append(top)

    return roots


def _parse_identifier(text: str) -> str | None:
    m = IDENTIFIER_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()


def _parse_datetime_range(text: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None, int | None]:
    for regex, fmt in (
        (RANGE_12_RE, "%Y%m%d%H%M"),
        (RANGE_10_RE, "%y%m%d%H%M"),
    ):
        m = regex.search(text)
        if m:
            start = pd.to_datetime(m.group("start"), format=fmt, errors="coerce")
            end = pd.to_datetime(m.group("end"), format=fmt, errors="coerce")
            interval = int(m.group("interval"))
            return start, end, interval

    return None, None, None


def _split_sections(text: str, source: str | None = None) -> tuple[list[str], list[str], list[str]]:
    lines = text.splitlines()
    try:
        cstart_idx = next(i for i, line in enumerate(lines) if line.startswith("*CSTART"))
        cend_idx = next(i for i, line in enumerate(lines) if line.startswith("*CEND"))
    except StopIteration as exc:
        raise ValueError(f"Could not find *CSTART and *CEND markers in {source}") from exc

    header_lines = lines[:cstart_idx]
    cstart_lines = lines[cstart_idx + 1 : cend_idx]
    data_lines = lines[cend_idx + 1 :]
    return header_lines, cstart_lines, data_lines


def _build_timestamps(start: pd.Timestamp | None, interval_min: int | None, n: int) -> pd.DatetimeIndex:
    if start is None or pd.isna(start) or interval_min is None:
        return pd.DatetimeIndex([pd.NaT] * n)
    return pd.date_range(start=start, periods=n, freq=f"{interval_min}min")


def _parse_fdv_text(text: str, source_name: str, report_id: str) -> pd.DataFrame:
    _, _, data_lines = _split_sections(text, source_name)

    identifier = _parse_identifier(text)
    start, end, interval = _parse_datetime_range(text)

    records: list[dict[str, Any]] = []
    record_idx = 0

    for line in data_lines:
        if not line.strip():
            continue

        for i in range(0, len(line), 15):
            chunk = line[i : i + 15]
            if not chunk.strip():
                continue

            m = re.match(
                r"^\s*(?P<flow>-?\d+)\s+(?P<depth>-?\d+)\s+(?P<velocity>-?\d+(?:\.\d+)?)\s*$",
                chunk,
            )
            if not m:
                continue

            record_idx += 1
            records.append(
                {
                    "record_idx": record_idx,
                    "flow_l_s": int(m.group("flow")),
                    "depth_mm": int(m.group("depth")),
                    "velocity_m_s": float(m.group("velocity")),
                }
            )

    df = pd.DataFrame(records)
    df.insert(0, "timestamp", _build_timestamps(start, interval, len(df)))
    df.insert(0, "report_id", report_id)
    df.insert(1, "station_id", identifier)
    df.insert(2, "source_file", source_name)
    df.insert(3, "start_time", start)
    df.insert(4, "end_time", end)
    df.insert(5, "interval_min", interval)
    return df


def _parse_rain_text(text: str, source_name: str, report_id: str) -> pd.DataFrame:
    _, _, data_lines = _split_sections(text, source_name)

    identifier = _parse_identifier(text)
    start, end, interval = _parse_datetime_range(text)

    values: list[float] = []
    for line in data_lines:
        if not line.strip():
            continue
        values.extend(float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", line))

    df = pd.DataFrame(
        {
            "record_idx": range(1, len(values) + 1),
            "rain_mm_hr": values,
        }
    )
    df.insert(0, "timestamp", _build_timestamps(start, interval, len(df)))
    df.insert(0, "report_id", report_id)
    df.insert(1, "gauge_id", identifier)
    df.insert(2, "source_file", source_name)
    df.insert(3, "start_time", start)
    df.insert(4, "end_time", end)
    df.insert(5, "interval_min", interval)
    return df


def _iter_data_files(folder_path: Path) -> Iterator[tuple[str, str, str]]:
    for file_path in sorted(folder_path.rglob("*")):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        if suffix in {".fdv", ".r"}:
            yield suffix, file_path.name, _read_text(file_path)
            continue

        if suffix == ".zip":
            try:
                with zipfile.ZipFile(file_path) as zf:
                    for member in zf.infolist():
                        if member.is_dir():
                            continue
                        member_suffix = Path(member.filename).suffix.lower()
                        if member_suffix not in {".fdv", ".r"}:
                            continue

                        source_name = f"{file_path.name}::{member.filename}"
                        text = zf.read(member).decode("utf-8", errors="replace")
                        yield member_suffix, source_name, text
            except zipfile.BadZipFile as exc:
                print(f"Skipping bad zip: {file_path}")
                print(f"Reason: {exc}")


def extract_report_folder(
    folder_path: str | Path,
    output_root: str | Path,
    skip_existing: bool = True,
    overwrite: bool = False,
) -> list[tuple[Path, Path]]:
    folder_path = Path(folder_path)
    output_root = Path(output_root)

    output_ids = _extract_output_ids(folder_path)

    fdv_frames: list[pd.DataFrame] = []
    rain_frames: list[pd.DataFrame] = []

    for suffix, source_name, text in _iter_data_files(folder_path):
        try:
            if suffix == ".fdv":
                fdv_frames.append(_parse_fdv_text(text, source_name, output_ids[0]))
            elif suffix == ".r":
                rain_frames.append(_parse_rain_text(text, source_name, output_ids[0]))
        except Exception as exc:
            print(f"Skipping bad file: {folder_path} :: {source_name}")
            print(f"Reason: {exc}")
            continue

    fdv_df = pd.concat(fdv_frames, ignore_index=True) if fdv_frames else pd.DataFrame()
    rain_df = pd.concat(rain_frames, ignore_index=True) if rain_frames else pd.DataFrame()

    outputs: list[tuple[Path, Path]] = []

    for report_id in output_ids:
        out_dir = output_root / report_id

        if out_dir.exists():
            if overwrite:
                for f in out_dir.iterdir():
                    if f.is_file():
                        f.unlink()
            elif skip_existing and any(out_dir.iterdir()):
                continue

        out_dir.mkdir(parents=True, exist_ok=True)

        fdv_out = out_dir / "fdv.csv"
        rain_out = out_dir / "rain.csv"

        if not fdv_df.empty:
            out_fdv = fdv_df.copy()
            out_fdv["report_id"] = report_id
            out_fdv.to_csv(fdv_out, index=False)
        else:
            pd.DataFrame().to_csv(fdv_out, index=False)

        if not rain_df.empty:
            out_rain = rain_df.copy()
            out_rain["report_id"] = report_id
            out_rain.to_csv(rain_out, index=False)
        else:
            pd.DataFrame().to_csv(rain_out, index=False)

        outputs.append((fdv_out, rain_out))

    return outputs


def extract_all_source_folders(
    raw_root: str | Path,
    output_root: str | Path,
    skip_existing: bool = True,
    overwrite: bool = False,
) -> list[tuple[Path, Path]]:
    raw_root = Path(raw_root)
    output_root = Path(output_root)

    folders = _discover_processing_folders(raw_root)

    outputs: list[tuple[Path, Path]] = []
    for folder_path in tqdm(folders, desc="Extracting Southern Water data"):
        outputs.extend(
            extract_report_folder(
                folder_path,
                output_root=output_root,
                skip_existing=skip_existing,
                overwrite=overwrite,
            )
        )

    return outputs


def build_processed_rain_and_fdv(
    combined_root: str | Path,
    processed_root: str | Path,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    combined_root = Path(combined_root)
    processed_root = Path(processed_root)
    processed_root.mkdir(parents=True, exist_ok=True)

    fdv_frames: list[pd.DataFrame] = []
    rain_frames: list[pd.DataFrame] = []

    for folder in sorted(p for p in combined_root.iterdir() if p.is_dir()):
        fdv_file = folder / "fdv.csv"
        rain_file = folder / "rain.csv"

        if fdv_file.exists():
            try:
                df = pd.read_csv(fdv_file)
                if not df.empty:
                    if "station_id" in df.columns:
                        df["station_id"] = df["station_id"].astype(str).str.strip()
                    fdv_frames.append(df)
            except pd.errors.EmptyDataError:
                pass

        if rain_file.exists():
            try:
                df = pd.read_csv(rain_file)
                if not df.empty:
                    if "gauge_id" in df.columns:
                        df["gauge_id"] = df["gauge_id"].astype(str).str.strip()
                    rain_frames.append(df)
            except pd.errors.EmptyDataError:
                pass

    fdv_out = processed_root / "fdv.csv"
    rain_out = processed_root / "rain.csv"

    if overwrite:
        if fdv_out.exists():
            fdv_out.unlink()
        if rain_out.exists():
            rain_out.unlink()

    fdv_df = pd.concat(fdv_frames, ignore_index=True) if fdv_frames else pd.DataFrame()
    rain_df = pd.concat(rain_frames, ignore_index=True) if rain_frames else pd.DataFrame()
    
    fdv_df = fdv_df.rename(columns={"timestamp": "time"})
    rain_df = rain_df.rename(columns={"timestamp": "time"})
    
    cols_fdv = [
    "station_id",
    "start_time",
    "end_time",
    "interval_min",
    "time",
    "record_idx",
    "flow_l_s",
    "depth_mm",
    "velocity_m_s",
    ]
    
    cols_rain = [
    "gauge_id",
    "start_time",
    "end_time",
    "interval_min",
    "time",
    "record_idx",
    "rain_mm_hr",

    
    fdv_df = fdv_df.drop_duplicates(subset=cols_fdv)
    fdv_df[['station_id', 'time', 'flow_l_s', 'depth_mm', 'velocity_m_s']].to_csv(fdv_out, index=False)
    
    rain_df = rain_df.drop_duplicates(subset=cols_rain)
    rain_df[['gauge_id', 'time', 'rain_mm_hr']].to_csv(rain_out, index=False)

    return fdv_out, rain_out


def build_station_lookup(monitors_df: pd.DataFrame) -> dict[str, str]:
    monitors = monitors_df.copy()

    monitors["id"] = monitors["id"].astype("string").str.strip()
    monitors["alt_id"] = monitors["alt_id"].astype("string").str.strip().str.upper()

    site_code = (
        monitors["site_name"]
        .astype("string")
        .str.upper()
        .str.extract(r"\b(FM\d+|DM\d+)\b", expand=False)
    )

    mapping_parts = [
        monitors[["id"]].rename(columns={"id": "key"}).assign(value=monitors["id"]),
        monitors.loc[monitors["alt_id"].notna(), ["alt_id", "id"]].rename(
            columns={"alt_id": "key", "id": "value"}
        ),
        monitors.loc[site_code.notna(), ["id"]].assign(key=site_code[site_code.notna()]).rename(
            columns={"id": "value"}
        ),
    ]

    mapping = pd.concat(mapping_parts, ignore_index=True).dropna()
    mapping["key"] = mapping["key"].astype("string").str.strip().str.upper()
    mapping["value"] = mapping["value"].astype("string").str.strip()

    mapping = mapping.drop_duplicates(subset=["key"], keep="first")

    return dict(zip(mapping["key"], mapping["value"]))


def clean_station_ids(fdv_df: pd.DataFrame, monitors_path: str | Path) -> pd.DataFrame:
    monitors = pd.read_csv(monitors_path, dtype="string")
    lookup = build_station_lookup(monitors)

    df = fdv_df.copy()

    station_norm = df["station_id"].astype("string").str.strip().str.upper()
    mapped = station_norm.map(lookup)

    fallback_id = df["source_file"].astype("string").str.extract(r"(\d{4,5})", expand=False)

    final = mapped.fillna(fallback_id).fillna(station_norm)
    df["station_id"] = final
    return df


def build_rain_gauge_lookup(monitors_df: pd.DataFrame) -> dict[str, str]:
    monitors = monitors_df.copy()
    monitors = monitors[monitors["type"].astype("string").str.lower().eq("rain_gauge")].copy()

    monitors["id"] = monitors["id"].astype("string").str.strip()
    monitors["alt_id"] = monitors["alt_id"].astype("string").str.strip().str.upper()

    site_code = (
        monitors["site_name"]
        .astype("string")
        .str.upper()
        .str.extract(r"\b(RG\d+)\b", expand=False)
    )

    mapping_parts = [
        monitors[["id"]].rename(columns={"id": "key"}).assign(value=monitors["id"]),
        monitors.loc[monitors["alt_id"].notna(), ["alt_id", "id"]].rename(
            columns={"alt_id": "key", "id": "value"}
        ),
        monitors.loc[site_code.notna(), ["id"]].assign(key=site_code[site_code.notna()]).rename(
            columns={"id": "value"}
        ),
    ]

    mapping = pd.concat(mapping_parts, ignore_index=True).dropna()
    mapping["key"] = mapping["key"].astype("string").str.strip().str.upper()
    mapping["value"] = mapping["value"].astype("string").str.strip()

    mapping = mapping.drop_duplicates(subset=["key"], keep="first")

    return dict(zip(mapping["key"], mapping["value"]))


def clean_rain_gauge_ids(rain_df: pd.DataFrame, monitors_path: str | Path) -> pd.DataFrame:
    monitors = pd.read_csv(monitors_path, dtype="string")
    lookup = build_rain_gauge_lookup(monitors)

    df = rain_df.copy()

    gauge_norm = df["gauge_id"].astype("string").str.strip().str.upper()
    mapped = gauge_norm.map(lookup)

    source_id = df["source_file"].astype("string").str.extract(r"(\d+)\.(?:r|fdv)$", expand=False)
    starts_with_2 = gauge_norm.str.match(r"^2\d+$", na=False)

    final = mapped.copy()
    final.loc[~starts_with_2] = source_id.loc[~starts_with_2]
    final.loc[starts_with_2] = mapped.loc[starts_with_2].fillna(source_id.loc[starts_with_2])

    df["gauge_id"] = final.fillna(gauge_norm)
    return df


def clean_processed_files(
    processed_root: str | Path,
    monitors_path: str | Path,
) -> tuple[Path, Path]:
    processed_root = Path(processed_root)
    
    fdv_path = processed_root / "fdv.csv"
    rain_path = processed_root / "rain.csv"

    fdv = pd.read_csv(fdv_path, dtype={"station_id": "string"})
    fdv_clean = clean_station_ids(fdv, monitors_path)
    fdv_clean.to_csv(fdv_path, index=False)

    rain = pd.read_csv(rain_path, dtype={"gauge_id": "string"})
    rain_clean = clean_rain_gauge_ids(rain, monitors_path)
    rain_clean.to_csv(rain_path, index=False)

    return fdv_path, rain_path


def process_southern_water(config_path: str | Path = CONFIG_PATH) -> tuple[Path, Path]:
    config = load_config(config_path)
    paths = get_southern_water_paths(config)

    unzip_archives(paths["raw_downloads_dir"], paths["unzipped_dir"])
    extract_all_source_folders(paths["unzipped_dir"], paths["combined_dir"])
    build_processed_rain_and_fdv(paths["combined_dir"], paths["processed_dir"])
    return clean_processed_files(paths["processed_dir"], paths["monitors_path"])


if __name__ == "__main__":
    fdv_out, rain_out = process_southern_water()
    print(f"Saved: {fdv_out}")
    print(f"Saved: {rain_out}")