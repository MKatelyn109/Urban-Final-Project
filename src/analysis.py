# A very aggressively simplified analysis file
from scipy.spatial.distance import cdist

# nit picking pearson for RQ1
def analyze_rq1_correlation(df):
    from scipy.stats import pearsonr

    r, p = pearsonr(df["ped"], df["bike"])
    return r, p


# latent demand for rq2
def analyze_rq2_latent_demand(df, ped_q=0.9, bike_q=0.3):
    ped_thr = df["ped"].quantile(ped_q)
    bike_thr = df["bike"].quantile(bike_q)

    latent = df[(df["ped"] >= ped_thr) & (df["bike"] <= bike_thr)].copy()
    return latent.sort_values("ped", ascending=False)


# accessibility for rq3
def analyze_rq3_accessibility(ped_locs, bike_locs):
    coords_p = ped_locs[["lat","lon"]].values
    coords_b = bike_locs[["lat","lon"]].values

    d = cdist(coords_p, coords_b)
    ped_locs["dist_m"] = d.min(axis=1) * 111_000
    ped_locs["num_near"] = (d < 0.008).sum(axis=1)
    return ped_locs


def plot_distributions(df, ped_col="ped", bike_col="bike", period_col="period"):
    import seaborn as sns
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.histplot(df[ped_col], bins=50, log_scale=True, ax=axes[0])
    axes[0].set_title("Pedestrian Volume Distribution")

    sns.boxplot(data=df[df[period_col].isin(["AM","MD","PM"])],
                x=period_col, y=ped_col, ax=axes[1])
    axes[1].set_title("Pedestrian Intensity by Period")

    sns.scatterplot(
        data=df[df[period_col].isin(["AM","MD","PM"])],
        x=ped_col, y=bike_col, hue=period_col, alpha=0.3, ax=axes[2]
    )
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_title("Pedestrian vs Bikeshare (Raw)")

    plt.tight_layout()
    plt.show()

def segmented_correlations(df, ped_col="ped", bike_col="bike", period_col="period"):
    from scipy.stats import pearsonr

    print(f"{'Period':<8} | {'N':<6} | Pearson r")
    print("-"*30)

    for p in ["AM","MD","PM"]:
        sub = df[df[period_col] == p]
        if len(sub) > 10:
            r, _ = pearsonr(sub[ped_col], sub[bike_col])
            print(f"{p:<8} | {len(sub):<6} | {r:.3f}")


def plot_structural_mismatch(df, period_col="period"):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    df = df[df[period_col].isin(["AM","MD","PM"])].copy()

    df["z_ped"] = df.groupby(period_col)["ped"].transform(lambda x: (x-x.mean())/x.std(ddof=0))
    df["z_bike"] = df.groupby(period_col)["bike"].transform(lambda x: (x-x.mean())/x.std(ddof=0))
    df["gap"] = df["z_ped"] - df["z_bike"]

    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x="z_ped", y="z_bike", hue=period_col, alpha=0.5)

    lims = [df[["z_ped","z_bike"]].min().min(),
            df[["z_ped","z_bike"]].max().max()]
    plt.plot(lims, lims, '--', color="red")

    plt.title("Z-Score Structural Mismatch")
    plt.xlabel("Pedestrian Intensity (Z)")
    plt.ylabel("Bikeshare Intensity (Z)")
    plt.grid(alpha=0.3)
    plt.show()


def dc_time_lag_analysis(df):
    from scipy.stats import pearsonr
    import matplotlib.pyplot as plt

    hourly = df.groupby("datetime")[["ped","bike"]].sum().sort_index()
    lags = [-2,-1,0,1,2]
    rs = []

    for lag in lags:
        shifted = hourly["bike"].shift(-lag)
        valid = shifted.notna()
        r, _ = pearsonr(hourly.loc[valid,"ped"], shifted[valid])
        rs.append(r)

    plt.plot(lags, rs, marker="o")
    plt.axvline(0, linestyle="--", color="gray")
    plt.title("Ped(t) vs Bike(t+lag)")
    plt.xlabel("Lag (hours)")
    plt.ylabel("Correlation")
    plt.grid(alpha=0.3)
    plt.show()

def plot_lorenz_gini(values, title="Lorenz Curve"):
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.sort(values)
    lorenz = np.cumsum(x) / x.sum()
    lorenz = np.insert(lorenz, 0, 0)

    plt.plot(np.linspace(0,1,len(lorenz)), lorenz)
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.title(title)
    plt.xlabel("Fraction of Nodes")
    plt.ylabel("Fraction of Activity")
    plt.show()

    gini = (np.abs(lorenz - np.linspace(0,1,len(lorenz))).sum()) / len(lorenz) / 0.5
    print(f"Gini: {gini:.3f}")

