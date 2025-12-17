import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import re


# Shared utilities dumped here
def period_from_hour(hour):
    if 7 <= hour < 10: return "AM"
    if 11 <= hour < 14: return "MD"
    if 16 <= hour < 20: return "PM"
    return "OFF"


def spatial_join_ped_bike(
    ped_df,
    bike_df,
    ped_loc_col,
    bike_loc_col,
    ped_time_cols,
    bike_time_cols,
    ped_value_col,
    bike_value_col,
    dist_m=400
):
    """
    Shared spatial + temporal join used by BOTH NYC and DC.
    """
    deg_thresh = dist_m / 111_000 * 1.25

    merged_rows = []
    matched = set()

    ped_locs = ped_df[[ped_loc_col, "lat", "lon"]].drop_duplicates()
    bike_locs = bike_df[[bike_loc_col, "lat", "lon"]].drop_duplicates()

    for _, p in ped_locs.iterrows():
        nearby = bike_locs[
            (np.abs(bike_locs["lat"] - p["lat"]) < deg_thresh) &
            (np.abs(bike_locs["lon"] - p["lon"]) < deg_thresh)
        ]

        if nearby.empty:
            continue

        matched.add(p[ped_loc_col])

        bike_near = bike_df[bike_df[bike_loc_col].isin(nearby[bike_loc_col])]
        bike_sum = bike_near.groupby(bike_time_cols)[bike_value_col].sum().reset_index()

        ped_loc = ped_df[ped_df[ped_loc_col] == p[ped_loc_col]]
        merged = ped_loc.merge(bike_sum, on=ped_time_cols, how="inner")

        merged_rows.append(merged)

    merged = pd.concat(merged_rows, ignore_index=True) if merged_rows else pd.DataFrame()
    unmatched = ped_df[~ped_df[ped_loc_col].isin(matched)]

    return merged, unmatched


def preprocess_dc(ped_raw, bike_raw, ped_coords):
    ped = ped_raw.copy()
    ped["datetime"] = pd.to_datetime(ped["Datetime"])
    ped["date"] = ped["datetime"].dt.date
    ped["hour"] = ped["datetime"].dt.hour
    ped["period"] = ped["hour"].apply(period_from_hour)

    ped["lat"] = ped["Station"].map(lambda x: ped_coords.get(x, (np.nan, np.nan))[0])
    ped["lon"] = ped["Station"].map(lambda x: ped_coords.get(x, (np.nan, np.nan))[1])

    ped = ped.dropna(subset=["Counts","lat","lon"])
    ped["ped"] = ped["Counts"]
    ped["location"] = ped["Station"]
    ped["time_key"] = ped["date"].astype(str) + "-H" + ped["hour"].astype(str)

    bike = bike_raw.copy()
    bike["started_at"] = pd.to_datetime(bike["started_at"], format="mixed")
    bike["date"] = bike["started_at"].dt.date
    bike["hour"] = bike["started_at"].dt.hour
    bike["period"] = bike["hour"].apply(period_from_hour)

    starts = bike.groupby(["start_station_name","start_lat","start_lng","date","hour","period"]).size().reset_index(name="s")
    ends = bike.groupby(["end_station_name","end_lat","end_lng","date","hour","period"]).size().reset_index(name="e")

    starts = starts.rename(columns={"start_station_name":"station","start_lat":"lat","start_lng":"lon"})
    ends = ends.rename(columns={"end_station_name":"station","end_lat":"lat","end_lng":"lon"})

    total = starts.merge(ends, on=["station","lat","lon","date","hour","period"], how="outer").fillna(0)
    total["bike"] = total["s"] + total["e"]
    bike_agg = total.groupby(["station","date","hour","period"]).agg({"lat":"mean","lon":"mean","bike":"sum"}).reset_index()
    bike_agg["time_key"] = bike_agg["date"].astype(str) + "-H" + bike_agg["hour"].astype(str)

    merged, unmatched = spatial_join_ped_bike(
        ped, bike_agg,
        ped_loc_col="location",
        bike_loc_col="station",
        ped_time_cols=["time_key","period"],
        bike_time_cols=["time_key","period"],
        ped_value_col="ped",
        bike_value_col="bike",
        dist_m=800
    )

    return merged, unmatched


def preprocess_nyc(ped_raw, bike_raw, dist_m=400):
    """
    NYC pipeline:
    - Wide bi-annual pedestrian counts
    - Peak-only bikeshare aggregation
    - Period-level spatial join

    Returns:
        merged_df, unmatched_ped_df, ped_long_df, bike_agg_df
    """

    df = ped_raw.copy()

    count_cols = [c for c in df.columns if re.match(r'^[A-Za-z]+\d{2}_[A-Za-z]+$', c)]
    id_vars = [c for c in df.columns if c not in count_cols]

    ped_long = df.melt(
        id_vars=id_vars,
        value_vars=count_cols,
        var_name="period_code",
        value_name="ped"
    )

    def parse_code(code):
        m = re.match(r'([A-Za-z]+)(\d{2})_([A-Za-z]+)', code)
        if m:
            month, year, period = m.groups()
            return month, int("20" + year), period.upper()
        return None, None, None

    parsed = ped_long["period_code"].apply(lambda x: pd.Series(parse_code(x)))
    ped_long[["month_name","year","period"]] = parsed

    month_map = {"May":5, "June":6, "Jun":6, "Sept":9, "Oct":10}
    ped_long["month"] = ped_long["month_name"].map(month_map)

    # Geometry extraction (robust ver.)
    if "the_geom" in ped_long.columns and "lat" not in ped_long.columns:
        ped_long["lon"] = ped_long["the_geom"].str.extract(r'POINT \(([-0-9.]+)') .astype(float)
        ped_long["lat"] = ped_long["the_geom"].str.extract(r'POINT \([-0-9.]+ ([-0-9.]+)') .astype(float)

    ped_long = ped_long.dropna(subset=["ped","lat","lon"])
    ped_long["location"] = ped_long["Loc"].astype(str)
    ped_long["time_key"] = ped_long["year"].astype(str) + "-" + ped_long["month"].astype(str)

    bike = bike_raw.copy()
    bike["started_at"] = pd.to_datetime(bike["started_at"])
    bike["hour"] = bike["started_at"].dt.hour
    bike["year"] = bike["started_at"].dt.year
    bike["month"] = bike["started_at"].dt.month

    def period_from_hour(h):
        if 7 <= h < 10: return "AM"
        if 11 <= h < 14: return "MD"
        if 16 <= h < 20: return "PM"
        return "OFF"

    bike["period"] = bike["hour"].apply(period_from_hour)
    bike = bike[bike["period"] != "OFF"]

    starts = bike.groupby(
        ["start_station_name","start_lat","start_lng","year","month","period"]
    ).size().reset_index(name="s")

    ends = bike.groupby(
        ["end_station_name","end_lat","end_lng","year","month","period"]
    ).size().reset_index(name="e")

    starts = starts.rename(columns={
        "start_station_name":"station",
        "start_lat":"lat",
        "start_lng":"lon"
    })
    ends = ends.rename(columns={
        "end_station_name":"station",
        "end_lat":"lat",
        "end_lng":"lon"
    })

    total = starts.merge(
        ends,
        on=["station","lat","lon","year","month","period"],
        how="outer"
    ).fillna(0)

    total["bike"] = total["s"] + total["e"]

    bike_agg = total.groupby(
        ["station","year","month","period"]
    ).agg({"lat":"mean","lon":"mean","bike":"sum"}).reset_index()

    bike_agg["time_key"] = bike_agg["year"].astype(str) + "-" + bike_agg["month"].astype(str)

    # joins
    merged, unmatched = spatial_join_ped_bike(
        ped_long, bike_agg,
        ped_loc_col="location",
        bike_loc_col="station",
        ped_time_cols=["time_key","period"],
        bike_time_cols=["time_key","period"],
        ped_value_col="ped",
        bike_value_col="bike",
        dist_m=dist_m
    )

    return merged, unmatched, ped_long, bike_agg

