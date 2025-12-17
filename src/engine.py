import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class RepositioningMasterEngine:
    """
    Data-adaptive repositioning engine for bikeshare systems.

    Supports:
      • NYC-style sparse snapshot datasets
      • DC-style continuous hourly datasets
      • Arbitrary future cities with similar fields
    """

    # INITIALIZATION & STANDARDIZATION
    def __init__(
        self,
        df_enriched: pd.DataFrame,
        city_name: str | None = None,
        unmatched_ped: pd.DataFrame | None = None,
        accessibility: pd.DataFrame | None = None,
        ped_col: str | None = None,
        bike_col: str | None = None,
        loc_col: str | None = None,
        period_col: str = "period",
    ):
        self.city_name = city_name or "UnknownCity"
        self.raw = df_enriched.copy()

        self.unmatched_ped_raw = unmatched_ped.copy() if unmatched_ped is not None else None
        self.access_raw = accessibility.copy() if accessibility is not None else None

        self.ped_col = ped_col or self._detect_column(["ped_count", "Counts"])
        self.bike_col = bike_col or self._detect_column(["bike_activity"])
        self.loc_col = loc_col or self._detect_column(["Loc", "ped_station", "Station"])
        self.period_col = period_col if period_col in self.raw.columns else None

        self.df = self._standardize_df()

        self.access_df = self._standardize_accessibility(self.access_raw)
        self.unmatched_ped_df = self._standardize_unmatched(self.unmatched_ped_raw)

        self.profiles = None

    def _detect_column(self, candidates):
        for c in candidates:
            if c in self.raw.columns:
                return c
        raise ValueError(f"None of {candidates} found among columns: {self.raw.columns.tolist()}")
    
    @staticmethod
    def format_table(df, cols=None, top_n=15, round_decimals=2):
        if df is None or df.empty:
            return pd.DataFrame()
        if cols is None:
            cols = [
                "location", "period_std",
                "ped_mean", "bike_mean", "conv_mean",
                "gap_mean", "frac_gap_hi", "frac_gap_lo",
                "undersupply_score", "oversupply_score"
            ]
        cols = [c for c in cols if c in df.columns]
        out = df[cols].head(top_n).copy()
        return out.round(round_decimals)

    def _standardize_df(self):
        """Creates consistent columns: ped, bike, location, period_std, z-scores, labels."""
        df = self.raw.copy()

        df["ped"] = df[self.ped_col].astype(float)
        df["bike"] = df[self.bike_col].astype(float)
        df["location"] = df[self.loc_col].astype(str)

        # Standard period
        df["period_std"] = (
            df[self.period_col].astype(str) if self.period_col else "ALL"
        )

        df["conversion_rate"] = np.where(df["ped"] > 0, df["bike"] / df["ped"], np.nan)

        if "Street_Nam" in df.columns:
            df["location_label"] = df["location"].astype(str) + " - " + df["Street_Nam"].astype(str)
        else:
            df["location_label"] = df["location"]

        # Z-scores per period
        def zt(x):
            std = x.std(ddof=0)
            return (x - x.mean()) / (std if std > 0 else 1)

        df["z_ped"] = df.groupby("period_std")["ped"].transform(zt)
        df["z_bike"] = df.groupby("period_std")["bike"].transform(zt)

        df["gap_score"] = df["z_ped"] - df["z_bike"]

        return df
    
    def recommended_reallocation(self, df_original=None, total_shift_frac=0.10):
        """
        Compute a reallocation vector indicating bikes to add/remove per row,
        properly distributed across multiple observations per location-period.
        """
        if df_original is None:
            df = self.df.copy()
        else:
            df = df_original.copy()

        if self.profiles is None:
            self.build_location_profiles()

        prof = self.profiles.reset_index()

        # Total bikes and pool
        total_bikes = df["bike"].sum()
        pool = total_bikes * total_shift_frac

        # Normalize weights
        prof["under_w"] = prof["undersupply_score"].clip(lower=0)
        prof["over_w"] = prof["oversupply_score"].clip(lower=0)

        sum_under = prof["under_w"].sum()
        sum_over = prof["over_w"].sum()

        if sum_under == 0 or sum_over == 0:
            return pd.Series(0, index=df.index)

        # NET change per location-period (total bikes to add/remove)
        prof["net_change"] = (
            pool * (prof["under_w"] / sum_under) -
            pool * (prof["over_w"] / sum_over)
        )

        # Merge net_change back to df
        df = df.merge(
            prof[["location", "period_std", "net_change"]],
            on=["location", "period_std"],
            how="left"
        )
        df["net_change"] = df["net_change"].fillna(0)

        # Count rows per (location, period)
        df["n_rows"] = df.groupby(["location", "period_std"])["location"].transform("size")
        df["reallocation"] = df["net_change"] / df["n_rows"]

        return df["reallocation"]




    #  Some random Optional Tables
    def _standardize_accessibility(self, access_df):
        if access_df is None:
            return None
        df = access_df.copy()

        loc_col = next((c for c in ["Loc", "ped_station", "Station", "location"] if c in df.columns), None)
        if loc_col is None:
            return None

        df["location"] = df[loc_col].astype(str)

        if "dist_m" not in df.columns:
            if "dist_nearest_station_deg" in df.columns:
                df["dist_m"] = df["dist_nearest_station_deg"] * 111000
            else:
                df["dist_m"] = np.nan

        if "num_within_threshold" not in df.columns:
            df["num_within_threshold"] = np.nan

        return df[["location", "dist_m", "num_within_threshold"]].drop_duplicates()

    def _standardize_unmatched(self, unmatched_df):
        if unmatched_df is None:
            return None
        df = unmatched_df.copy()

        loc_col = next((c for c in ["Loc", "ped_station", "Station"] if c in df.columns), None)
        ped_col = next((c for c in ["ped_count", "Counts"] if c in df.columns), None)
        if loc_col is None or ped_col is None:
            return None

        df["location"] = df[loc_col].astype(str)
        df["ped"] = df[ped_col].astype(float)

        cols = ["location", "ped"]
        for c in ["lat", "lon"]:
            if c in df.columns:
                cols.append(c)

        return df[cols]

    # ------------------------------------------------------------------
    # BUILD LOCATION PROFILES
    # ------------------------------------------------------------------
    def build_location_profiles(self, gap_hi=1.0, gap_lo=-1.0, min_obs=3):
        df = self.df.copy()

        agg_dict = {
            "ped": "mean",
            "bike": "mean",
            "conversion_rate": "mean",
            "z_ped": "mean",
            "z_bike": "mean",
            "gap_score": "mean",
        }
        if "lat" in df.columns:
            agg_dict["lat"] = "mean"
        if "lon" in df.columns:
            agg_dict["lon"] = "mean"

        prof = df.groupby(["location", "period_std"]).agg(agg_dict).rename(
            columns={
                "ped": "ped_mean",
                "bike": "bike_mean",
                "conversion_rate": "conv_mean",
                "z_ped": "z_ped_mean",
                "z_bike": "z_bike_mean",
                "gap_score": "gap_mean",
            }
        )

        prof["n_obs"] = df.groupby(["location", "period_std"]).size()

        tmp = df.copy()
        tmp["is_hi"] = tmp["gap_score"] >= gap_hi
        tmp["is_lo"] = tmp["gap_score"] <= gap_lo

        prof["frac_gap_hi"] = tmp.groupby(["location", "period_std"])["is_hi"].mean()
        prof["frac_gap_lo"] = tmp.groupby(["location", "period_std"])["is_lo"].mean()

        # Scoring
        prof["undersupply_score"] = (
            np.maximum(prof["gap_mean"], 0)
            * prof["frac_gap_hi"].clip(lower=0)
            * np.log1p(prof["ped_mean"].clip(lower=0))
        )
        prof["oversupply_score"] = (
            np.maximum(-prof["gap_mean"], 0)
            * prof["frac_gap_lo"].clip(lower=0)
            * np.log1p(prof["bike_mean"].clip(lower=0))
        )

        # Minimum sample count
        prof = prof[prof["n_obs"] >= min_obs].copy()

        # Attach labels (this was the missing piece!)
        labels = df[["location", "location_label"]].drop_duplicates("location")
        prof = prof.reset_index().merge(labels, on="location", how="left")

        # Accessibility correction
        if self.access_df is not None:
            prof = prof.merge(self.access_df, on="location", how="left")

            dist_term = 1 + (prof["dist_m"].fillna(0) / 500)
            neigh_term = 1 + 1 / (1 + prof["num_within_threshold"].fillna(0))
            access_mult = dist_term * neigh_term

            prof["undersupply_score"] *= access_mult

        self.profiles = prof.set_index(["location", "period_std"])
        return self.profiles

    def rank_undersupplied(self, top_n=20, min_score=0.0):
        if self.profiles is None:
            self.build_location_profiles()
        prof = self.profiles.reset_index()
        return prof[prof["undersupply_score"] > min_score].sort_values(
            "undersupply_score", ascending=False
        ).head(top_n)

    def rank_oversupplied(self, top_n=20, min_score=0.0):
        if self.profiles is None:
            self.build_location_profiles()
        prof = self.profiles.reset_index()
        return prof[prof["oversupply_score"] > min_score].sort_values(
            "oversupply_score", ascending=False
        ).head(top_n)


    #(if unmatched ped counters exist)
    def rank_new_station_candidates(self, ped_quantile=0.9, min_obs=3):
        if self.unmatched_ped_df is None:
            print("No unmatched pedestrian data supplied.")
            return pd.DataFrame()

        df = self.unmatched_ped_df.copy()
        agg = df.groupby("location").agg(
            mean_ped=("ped", "mean"),
            n_obs=("ped", "size"),
            lat=("lat", "mean") if "lat" in df.columns else ("ped", "size"),
            lon=("lon", "mean") if "lon" in df.columns else ("ped", "size")
        )

        agg = agg[agg["n_obs"] >= min_obs]
        thr = agg["mean_ped"].quantile(ped_quantile)

        cand = agg[agg["mean_ped"] >= thr]
        cand["new_station_score"] = cand["mean_ped"] * np.log1p(cand["n_obs"])

        return cand.sort_values("new_station_score", ascending=False)

    # plots
    def plot_top_locations(self, mode="undersupplied", top_n=10):
        if self.profiles is None:
            self.build_location_profiles()

        col = "undersupply_score" if mode == "undersupplied" else "oversupply_score"
        prof = self.profiles.reset_index()

        df = prof.sort_values(col, ascending=False).head(top_n)
        if df.empty:
            print("No entries to plot.")
            return

        plt.figure(figsize=(10, 5))
        sns.barplot(data=df, x=col, y="location_label", hue="period_std", dodge=False)
        plt.title(f"{self.city_name}: Top {top_n} {mode.capitalize()} Locations")
        plt.xlabel("Score")
        plt.ylabel("Location")
        plt.tight_layout()
        plt.show()

    def plot_map(self, mode="undersupplied", top_n=50):
        if self.profiles is None:
            self.build_location_profiles()

        col = "undersupply_score" if mode == "undersupplied" else "oversupply_score"
        df = self.profiles.reset_index()

        if "lat" not in df.columns or "lon" not in df.columns:
            print("Missing lat/lon — cannot plot map.")
            return

        cand = df.sort_values(col, ascending=False).head(top_n)

        plt.figure(figsize=(8, 8))
        plt.scatter(df["lon"], df["lat"], s=5, alpha=0.2, label="All")

        plt.scatter(cand["lon"], cand["lat"], s=50, alpha=0.9, label=f"Top {mode}")
        for _, row in cand.iterrows():
            plt.annotate(
                f"{row['location_label']} ({row['period_std']})",
                (row["lon"], row["lat"]),
                fontsize=7, xytext=(5, 2), textcoords="offset points"
            )

        plt.title(f"{self.city_name}: Top {mode.capitalize()} Spatial Distribution")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def plot_before_after_map(self, transformed_df, top_n=50):
        if self.profiles is None:
            self.build_location_profiles()

        before = self.df.groupby("location").agg({"bike": "mean", "lat": "mean", "lon": "mean"}).reset_index()
        after = transformed_df.groupby("location").agg({"bike": "mean", "lat": "mean", "lon": "mean"}).reset_index()

        merged = before.merge(after, on="location", suffixes=("_before", "_after"))
        merged["delta_bike"] = merged["bike_after"] - merged["bike_before"]

        selected = merged.sort_values("delta_bike", ascending=False).head(top_n)

        plt.figure(figsize=(14, 7))

        # BEFORE
        plt.subplot(1, 2, 1)
        plt.scatter(before["lon"], before["lat"], s=5, alpha=0.3)
        plt.scatter(
            selected["lon_before"], selected["lat_before"],
            s=40, c=selected["bike_before"], cmap="Blues"
        )
        plt.title("BEFORE Repositioning")
        plt.grid(alpha=0.3)

        # AFTER
        plt.subplot(1, 2, 2)
        plt.scatter(after["lon"], after["lat"], s=5, alpha=0.3)
        plt.scatter(
            selected["lon_after"], selected["lat_after"],
            s=40, c=selected["bike_after"], cmap="Greens"
        )
        plt.title("AFTER Repositioning")
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
        

    def coverage_improvement_panel(self, df_after, beta=1.0, top_n=15):
        """
        Highlights improvements between BEFORE vs AFTER repositioning.
        Uses raw (non-z-score) gap to show real imbalance.
        """

        before = self.df.copy()
        after = df_after.copy()

        # Compute RAW GAP
        before["raw_gap"] = before["ped"] - beta * before["bike"]
        after["raw_gap"]  = after["ped"]  - beta * after["bike"]

        # Station-level averages
        b = before.groupby("location")["raw_gap"].mean()
        a = after.groupby("location")["raw_gap"].mean()

        improvement = (b - a).sort_values(ascending=False)
        improvement_df = improvement.to_frame("raw_gap_reduction")
        improvement_df["gap_before"] = b
        improvement_df["gap_after"] = a

        # Compute network metrics
        # Compute thresholds for severe mismatch
        quant = before["raw_gap"].quantile(0.9)
        quant_neg = before["raw_gap"].quantile(0.1)

        # Severe undersupply: raw_gap well above typical
        severe_under_before = (before["raw_gap"] > quant).mean()
        severe_under_after  = (after["raw_gap"]  > quant).mean()

        # Severe oversupply: raw_gap well below typical
        severe_over_before  = (before["raw_gap"] < quant_neg).mean()
        severe_over_after   = (after["raw_gap"]  < quant_neg).mean()


        # Misallocation Index
        nmi_before = (before["raw_gap"].abs()).mean()
        nmi_after  = (after["raw_gap"].abs()).mean()

        # Gini improvement (bike supply)
        gini_before = self._gini(before["bike"])
        gini_after  = self._gini(after["bike"])

        # ---- PLOTS ----
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,5))

        plt.subplot(1,2,1)
        sns.kdeplot(before["raw_gap"], label="BEFORE", fill=True, alpha=0.3)
        sns.kdeplot(after["raw_gap"], label="AFTER", fill=True, alpha=0.3)
        plt.title("Raw Gap Distribution Shift")
        plt.xlabel("ped - β·bike")
        plt.legend()

        plt.subplot(1,2,2)
        sns.scatterplot(
            x=b, y=a, alpha=0.5
        )
        lim = [min(b.min(), a.min()), max(b.max(), a.max())]
        plt.plot(lim, lim, '--', color='black')
        plt.xlabel("Raw Gap Before")
        plt.ylabel("Raw Gap After")
        plt.title("Station-Level Improvement (Below Line = Better)")

        plt.tight_layout()
        plt.show()

        # ---- SUMMARY REPORT ----
        summary = {
            "Severe Undersupply Before": severe_under_before,
            "Severe Undersupply After": severe_under_after,
            "Severe Oversupply Before": severe_over_before,
            "Severe Oversupply After": severe_over_after,
            "Misallocation Index Before": nmi_before,
            "Misallocation Index After": nmi_after,
            "Bike Gini Before": gini_before,
            "Bike Gini After": gini_after,
        }

        return summary, improvement_df.sort_values("raw_gap_reduction", ascending=False).head(top_n)


    @staticmethod
    def _gini(x):
        # Safe Gini, should work
        x = np.asarray(x).flatten()
        if np.amin(x) < 0:
            x -= np.amin(x)
        x += 1e-9
        x_sorted = np.sort(x)
        n = x.shape[0]
        return (2 * np.sum((np.arange(1, n+1) * x_sorted)) / (n * np.sum(x_sorted))) - (n + 1) / n

