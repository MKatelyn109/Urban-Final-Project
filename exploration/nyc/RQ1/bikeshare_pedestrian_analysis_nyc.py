"""
NYC Bikeshare and Pedestrian Traffic Correlation Analysis
RQ1: How well does pedestrian traffic intensity correlate with bikeshare usage?

This script performs spatial and temporal alignment of bikeshare and pedestrian data,
then analyzes correlations across time and space.
"""

import re
import pandas as pd
import numpy as np
from datetime import datetime
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Point
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Assuming you have bike and ped dataframes already loaded
# bike = pd.read_csv('your_bikeshare_data.csv')
# ped = pd.read_csv('your_pedestrian_data.csv')


class BikeSharePedestrianAnalyzer:
    """
    Analyzes correlation between bikeshare usage and pedestrian traffic
    """
    
    def __init__(self, bike_df, ped_df, distance_threshold=800):
        """
        Initialize analyzer with bikeshare and pedestrian dataframes
        
        Parameters:
        -----------
        bike_df : DataFrame with bikeshare trip data
        ped_df : DataFrame with pedestrian count data
        distance_threshold : maximum distance (meters) for spatial matching
        """
        self.bike_raw = bike_df.copy()
        self.ped_raw = ped_df.copy()
        self.distance_threshold = distance_threshold
        self.bike_clean = None
        self.ped_clean = None
        self.station_coords = None
        self.ped_coords = None
        self.spatial_matches = None
        self.correlation_results = None
        
    def clean_bikeshare_data(self):
        """
        Clean bikeshare data and prepare for analysis
        """
        print("Cleaning bikeshare data...")
        df = self.bike_raw.copy()
        
        # Keep only necessary columns
        columns_to_keep = [
            'started_at', 'ended_at', 
            'start_station_id', 'start_station_name',
            'end_station_id', 'end_station_name',
            'start_lat', 'start_lng',
            'end_lat', 'end_lng',
            'member_casual'
        ]
        df = df[columns_to_keep]
        
        # Parse dates
        df['start_datetime'] = pd.to_datetime(df['started_at'])
        df['end_datetime'] = pd.to_datetime(df['ended_at'])
        
        # Extract temporal features
        df['start_hour'] = df['start_datetime'].dt.hour
        df['start_date'] = df['start_datetime'].dt.date
        df['start_dow'] = df['start_datetime'].dt.dayofweek  # 0=Monday
        df['start_month'] = df['start_datetime'].dt.month
        df['start_year'] = df['start_datetime'].dt.year
        
        # Similar for end times
        df['end_hour'] = df['end_datetime'].dt.hour
        
        # Remove any rows with missing station info or coordinates
        df = df.dropna(subset=['start_station_name', 'end_station_name', 
                               'start_lat', 'start_lng', 'end_lat', 'end_lng'])
        
        # Remove rows with invalid coordinates (0,0 or outside DC area)
        # NYC roughly: lat 40.5-40.9, lng -74.25 to -73.7
        df = df[
            (df['start_lat'] >= 40.5) & (df['start_lat'] <= 40.9) &
            (df['start_lng'] >= -74.25) & (df['start_lng'] <= -73.7) &
            (df['end_lat'] >= 40.5) & (df['end_lat'] <= 40.9) &
            (df['end_lng'] >= -74.25) & (df['end_lng'] <= -73.7)
        ]
        
        self.bike_clean = df
        print(f"Bikeshare data cleaned: {len(df)} trips")
        return df
    
    def clean_pedestrian_data(self):
        """
        Clean NYC pedestrian data and prepare for analysis.
        Converts wide-format (e.g., May07_AM, Sept18_PM, etc.) into long-format
        and extracts date/time components for correlation with bikeshare data.
        """
        print("Cleaning NYC pedestrian data...")
        df = self.ped_raw.copy()
        df.columns = df.columns.str.strip()
        # Parse geometry
        if 'the_geom' in df.columns:
            from shapely import wkt
            df['geometry'] = df['the_geom'].apply(wkt.loads)
        else:
            raise ValueError("Expected 'the_geom' column in pedestrian data")

        # Identify all pedestrian count columns
        count_cols = [c for c in df.columns if re.match(r'^(May|Sept)\d{2}_(AM|PM|MD)$', c)]

        if not count_cols:
            raise ValueError("No pedestrian count columns found (expected columns like 'May07_AM').")

        # Melt wide columns into long format
        df_long = df.melt(
            id_vars=['OBJECTID', 'Borough', 'Street_Nam', 'From_Stree', 'To_Street', 'geometry'],
            value_vars=count_cols,
            var_name='date_period',
            value_name='ped_count'
        )

        # Extract time components 
        df_long[['month', 'year', 'time_period']] = df_long['date_period'].str.extract(r'([A-Za-z]+)(\d+)_([A-Za-z]+)')

        # Convert year to numeric 
        df_long['year'] = pd.to_numeric(df_long['year'], errors='coerce')
        df_long.loc[df_long['year'] < 100, 'year'] += 2000  # assume all years are 2000s

        # Clean pedestrian counts
        df_long['ped_count'] = pd.to_numeric(df_long['ped_count'], errors='coerce')
        df_long = df_long.dropna(subset=['ped_count'])
        df_long = df_long[df_long['ped_count'] >= 0]

        # --- 6. Optional: normalize month names 
        # month_map = {
        #     'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 'Jul': 7,
        #     'Aug': 8, 'Sept': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
        # }
        # df_long['month_num'] = df_long['month'].map(month_map).fillna(0).astype(int)

        # Save cleaned dataframe
        self.ped_clean = df_long
        print(f"NYC pedestrian data cleaned: {len(df_long):,} rows (long format)")
        
        return df_long

    
    def geocode_stations(self):
        """
        Extract bikeshare and pedestrian station coordinates directly from datasets
        (no external geocoding needed).

        Parameters
        ----------
        use_cache : bool
            Whether to load from cache if available
        cache_file : str
            Path to cache CSV
        """
        import re
        import numpy as np
        import pandas as pd
        import os

        print("\nProcessing station coordinates (using provided lat/lon data)...")

        # ------------------------------------------------------------------------
        # PART 1: Bikeshare stations (already has lat/lon)
        # ------------------------------------------------------------------------

        # Start stations
        start_stations = self.bike_clean[[
            'start_station_id', 'start_station_name', 'start_lat', 'start_lng'
        ]].rename(columns={
            'start_station_id': 'station_id',
            'start_station_name': 'station_name',
            'start_lat': 'latitude',
            'start_lng': 'longitude'
        }).drop_duplicates()

        # End stations
        end_stations = self.bike_clean[[
            'end_station_id', 'end_station_name', 'end_lat', 'end_lng'
        ]].rename(columns={
            'end_station_id': 'station_id',
            'end_station_name': 'station_name',
            'end_lat': 'latitude',
            'end_lng': 'longitude'
        }).drop_duplicates()

        # Combine & average any duplicates
        all_bike_stations = pd.concat([start_stations, end_stations])
        bike_coords = all_bike_stations.groupby(['station_id', 'station_name']).agg({
            'latitude': 'mean',
            'longitude': 'mean'
        }).reset_index()
        bike_coords['station_type'] = 'bikeshare'


        # ------------------------------------------------------------------------
        # PART 2: Pedestrian counters (parse from the_geom)
        # ------------------------------------------------------------------------

        def parse_geom(geom):
            """Handle Shapely Point or WKT string."""
            # If Shapely Point object
            if isinstance(geom, Point):
                return geom.y, geom.x  # (lat, lon)
            # If string like "POINT (-73.905 40.879)"
            elif isinstance(geom, str) and "POINT" in geom.upper():
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", geom)
                if len(nums) == 2:
                    lon, lat = map(float, nums)
                    return lat, lon
            return None, None



        ped = self.ped_clean.copy()
        ped.columns = ped.columns.str.strip().str.replace('\ufeff', '')

        for col in ped.columns:
            if col.lower().strip() in ['geometry', 'the_geom']:
                geom_col = col
                break

        if geom_col is None:
            raise ValueError("No geometry column found in pedestrian dataset")

        ped[['latitude', 'longitude']] = ped[geom_col].apply(
            lambda g: pd.Series(parse_geom(g))
        )
        ped_coords = ped[['Street_Nam', 'latitude', 'longitude']].drop_duplicates()
        ped_coords = ped_coords.rename(columns={'Street_Nam': 'station_name'})
        ped_coords['station_id'] = None
        ped_coords['station_type'] = 'pedestrian'


        # ------------------------------------------------------------------------
        # PART 3: Combine & save
        # ------------------------------------------------------------------------
        all_coords = pd.concat([bike_coords, ped_coords], ignore_index=True)

        self.station_coords = all_coords
        return all_coords

    
    def create_spatial_matches(self):
        """
        Match pedestrian counters to nearby bikeshare stations
        Returns matches within distance_threshold
        """
        print(f"\nCreating spatial matches (threshold: {self.distance_threshold}m)...")
        
        # Separate bikeshare and pedestrian coordinates
        bike_coords = self.station_coords[
            self.station_coords['station_type'] == 'bikeshare'
        ].dropna(subset=['latitude', 'longitude'])
        
        ped_coords = self.station_coords[
            self.station_coords['station_type'] == 'pedestrian'
        ].dropna(subset=['latitude', 'longitude'])
        
        if len(bike_coords) == 0 or len(ped_coords) == 0:
            print("ERROR: No valid coordinates found for matching")
            return None
        
        
        # Calculate pairwise distances using haversine formula for better accuracy
        from math import radians, cos, sin, asin, sqrt
        
        def haversine(lon1, lat1, lon2, lat2):
            """
            Calculate the great circle distance between two points 
            on the earth (specified in decimal degrees)
            Returns distance in meters
            """
            # Convert decimal degrees to radians
            lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371000  # Radius of earth in meters
            return c * r
        
        # Find matches within threshold
        matches = []
        for i, ped_row in ped_coords.iterrows():
            ped_name = ped_row['station_name']
            ped_lat = ped_row['latitude']
            ped_lon = ped_row['longitude']
            
            for j, bike_row in bike_coords.iterrows():
                bike_name = bike_row['station_name']
                bike_id = bike_row['station_id']
                bike_lat = bike_row['latitude']
                bike_lon = bike_row['longitude']
                
                # Calculate distance
                distance = haversine(ped_lon, ped_lat, bike_lon, bike_lat)
                
                if distance <= self.distance_threshold:
                    matches.append({
                        'ped_station': ped_name,
                        'bike_station': bike_name,
                        'bike_station_id': bike_id,
                        'distance_m': distance,
                        'ped_lat': ped_lat,
                        'ped_lon': ped_lon,
                        'bike_lat': bike_lat,
                        'bike_lon': bike_lon
                    })
        
        matches_df = pd.DataFrame(matches)
        
        if len(matches_df) == 0:
            print("WARNING: No matches found!")
            print("Consider increasing distance_threshold")
            return matches_df
        
        print(f"    {matches_df['ped_station'].nunique()} pedestrian stations matched")
        print(f"    {matches_df['bike_station'].nunique()} bikeshare stations matched")
        
        self.spatial_matches = matches_df
        return matches_df
    
    def aggregate_bikeshare_hourly(self):
        """
        Aggregate bikeshare trips to hourly counts by station
        """
        print("\nAggregating bikeshare data to hourly resolution...")
        
        # Aggregate starts
        starts = self.bike_clean.groupby([
            'start_station_name', 'start_year', 'start_month', 
            'start_date', 'start_hour', 'start_dow'
        ]).size().reset_index(name='trips_started')
        
        # Aggregate ends
        ends = self.bike_clean.groupby([
            'end_station_name', 'start_year', 'start_month',
            'start_date', 'end_hour', 'start_dow'
        ]).size().reset_index(name='trips_ended')
        
        # Rename for consistency
        starts.columns = ['station', 'year', 'month', 'date', 'hour', 'dow', 'trips_started']
        ends.columns = ['station', 'year', 'month', 'date', 'hour', 'dow', 'trips_ended']
        
        # Merge starts and ends
        hourly = starts.merge(
            ends, 
            on=['station', 'year', 'month', 'date', 'hour', 'dow'],
            how='outer'
        ).fillna(0)
        
        # Calculate total activity (turnover)
        hourly['total_trips'] = hourly['trips_started'] + hourly['trips_ended']
        
        # Create datetime for easier matching - convert date object to datetime
        hourly['datetime'] = pd.to_datetime(hourly['date']) + pd.to_timedelta(hourly['hour'], unit='h')
        
        print(f"  Created hourly aggregation: {len(hourly)} station-hour combinations")
        
        self.bike_hourly = hourly
        return hourly
    
    def prepare_pedestrian_long(self):
        """
        Prepare pedestrian data in long format.
        Assumes ped_clean is already in long format with columns like:
        'station_name', 'latitude', 'longitude', 'date_period', 'ped_count'
        """
        import re
        import pandas as pd
        from shapely.geometry import Point

        print("\nPreparing pedestrian data in long format...")

        ped = self.ped_clean.copy()

        # Rename station column
        if 'Street_Nam' in ped.columns:
            ped = ped.rename(columns={'Street_Nam': 'station_name'})

        # Parse geometry if latitude/longitude not present
        if 'latitude' not in ped.columns or 'longitude' not in ped.columns:
            if 'geometry' not in ped.columns and 'the_geom' in ped.columns:
                ped = ped.rename(columns={'the_geom': 'geometry'})

            def parse_geom(geom):
                if isinstance(geom, Point):
                    return geom.y, geom.x
                if isinstance(geom, str) and "POINT" in geom:
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", geom)
                    if len(nums) == 2:
                        lon, lat = map(float, nums)
                        return lat, lon
                return None, None

            ped[['latitude', 'longitude']] = ped['geometry'].apply(lambda g: pd.Series(parse_geom(g)))

        ped = ped.dropna(subset=['latitude', 'longitude'])

        # Assign ped_long directly from cleaned long-format columns
        self.ped_long = ped.rename(columns={
            'ped_count': 'pedestrian_count',
            'date_period': 'datetime'
        })[['station_name', 'latitude', 'longitude', 'datetime', 'pedestrian_count']]

        print("ped_long prepared:")
        print(self.ped_long.head())
        print("Total rows:", len(self.ped_long))

        return self.ped_long

    
    def compute_correlations(self):
        """
        Compute correlations between pedestrian traffic and bikeshare usage
        for spatially matched locations
        """
        print("\nComputing correlations...")
        
        if self.spatial_matches is None:
            print("ERROR: Must create spatial matches first")
            return None
        
        # Aggregate bikeshare hourly
        bike_hourly = self.aggregate_bikeshare_hourly()
        bike_hourly['datetime'] = pd.to_datetime(bike_hourly['datetime'])

        # We'll use ped_clean directly instead of ped_long
        ped_df = self.ped_clean.copy()

        # Ensure column names are consistent
        if 'ped_count' not in ped_df.columns or 'date_period' not in ped_df.columns:
            print("ERROR: ped_clean is missing expected columns ('ped_count', 'date_period').")
            print("Available columns:", ped_df.columns.tolist())
            return None

        results = []

        # For each pedestrian station
        for ped_station in self.spatial_matches['ped_station'].unique():
            # Get matched bikeshare stations
            matched_bikes = self.spatial_matches[
                self.spatial_matches['ped_station'] == ped_station
            ]['bike_station'].values

            # Pedestrian data for this station
            ped_data = self.ped_long[self.ped_long['station_name'] == ped_station].copy()
            ped_data = ped_data.rename(columns={'datetime': 'period'})

            # Bikeshare data for matched stations
            bike_data = bike_hourly[bike_hourly['station'].isin(matched_bikes)].copy()

            # Create period labels for bikeshare
            def get_period_label(ts):
                month_day = ts.strftime('%b%d')  # e.g., 'May07'
                hour = ts.hour
                if hour < 12:
                    return f'{month_day}_AM'
                elif hour < 17:
                    return f'{month_day}_MD'
                else:
                    return f'{month_day}_PM'

            bike_data['period'] = bike_data['datetime'].apply(get_period_label)

            # Aggregate bike trips by period
            bike_data = bike_data.groupby('period').agg({
                'trips_started': 'sum',
                'trips_ended': 'sum',
                'total_trips': 'sum'
            }).reset_index()

            # Merge on 'period'
            merged = ped_data.merge(bike_data, on='period', how='inner')

            if len(merged) > 10:  # Need sufficient data points
                try:
                    from scipy.stats import pearsonr, spearmanr
                    pearson_total, p_pearson = pearsonr(merged['pedestrian_count'], merged['total_trips'])
                    spearman_total, p_spearman = spearmanr(merged['pedestrian_count'], merged['total_trips'])

                    results.append({
                        'ped_station': ped_station,
                        'n_matched_bike_stations': len(matched_bikes),
                        'n_observations': len(merged),
                        'pearson_r': pearson_total,
                        'pearson_p': p_pearson,
                        'spearman_r': spearman_total,
                        'spearman_p': p_spearman,
                        'mean_ped_count': merged['pedestrian_count'].mean(),
                        'mean_bike_trips': merged['total_trips'].mean(),
                        'std_ped_count': merged['pedestrian_count'].std(),
                        'std_bike_trips': merged['total_trips'].std()
                    })
                except Exception as e:
                    print(f"Error computing correlation for {ped_station}: {e}")
        
        results_df = pd.DataFrame(results)
        print(f"\nCorrelation analysis complete for {len(results_df)} locations")
        
        self.correlation_results = results_df
        return results_df

    
    def plot_correlation_summary(self, save_path='RQ1_PLOTS/correlation_summary.png'):
        """
        Create visualization of correlation results
        """
        if self.correlation_results is None:
            print("No correlation results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Distribution of correlation coefficients
        ax = axes[0, 0]
        ax.hist(self.correlation_results['pearson_r'], bins=20, alpha=0.6, label='Pearson', edgecolor='black')
        ax.hist(self.correlation_results['spearman_r'], bins=20, alpha=0.6, label='Spearman', edgecolor='black')
        ax.axvline(self.correlation_results['pearson_r'].mean(), color='blue', linestyle='--', linewidth=2)
        ax.axvline(self.correlation_results['spearman_r'].mean(), color='orange', linestyle='--', linewidth=2)
        ax.set_xlabel('Correlation Coefficient')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Correlation Coefficients')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. Correlation strength by number of observations
        ax = axes[0, 1]
        scatter = ax.scatter(
            self.correlation_results['n_observations'],
            self.correlation_results['pearson_r'],
            c=self.correlation_results['n_matched_bike_stations'],
            s=100,
            alpha=0.6,
            cmap='viridis',
            edgecolors='black'
        )
        ax.set_xlabel('Number of Observations')
        ax.set_ylabel('Pearson Correlation')
        ax.set_title('Correlation Strength vs Sample Size')
        plt.colorbar(scatter, ax=ax, label='# Matched Bike Stations')
        ax.grid(alpha=0.3)
        
        # 3. Mean pedestrian count vs mean bike trips
        ax = axes[1, 0]
        ax.scatter(
            self.correlation_results['mean_ped_count'],
            self.correlation_results['mean_bike_trips'],
            s=100,
            alpha=0.6,
            edgecolors='black'
        )
        ax.set_xlabel('Mean Pedestrian Count (per hour)')
        ax.set_ylabel('Mean Bike Trips (per hour)')
        ax.set_title('Average Activity Levels')
        ax.grid(alpha=0.3)
        
        # 4. Statistical significance
        ax = axes[1, 1]
        sig_level = 0.05
        significant = self.correlation_results[self.correlation_results['pearson_p'] < sig_level]
        not_sig = self.correlation_results[self.correlation_results['pearson_p'] >= sig_level]
        
        ax.scatter(significant['pearson_r'], significant['spearman_r'], 
                  label=f'Significant (p<{sig_level})', s=100, alpha=0.6, edgecolors='black')
        ax.scatter(not_sig['pearson_r'], not_sig['spearman_r'], 
                  label=f'Not Significant (p≥{sig_level})', s=100, alpha=0.6, edgecolors='black')
        ax.plot([-1, 1], [-1, 1], 'k--', alpha=0.3)
        ax.set_xlabel('Pearson Correlation')
        ax.set_ylabel('Spearman Correlation')
        ax.set_title('Pearson vs Spearman Correlation')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved correlation summary plot to {save_path}")
        plt.close()
        
        return fig
    
    def generate_summary_report(self):
        """
        Generate text summary of findings
        """
        if self.correlation_results is None:
            print("No correlation results available")
            return
        
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nDataset Overview:")
        print(f"  Bikeshare trips: {len(self.bike_clean):,}")
        print(f"  Pedestrian observations: {len(self.ped_clean):,}")
        print(f"  Spatial matches: {len(self.spatial_matches)}")
        print(f"  Locations analyzed: {len(self.correlation_results)}")
        
        print(f"\nCorrelation Statistics:")
        print(f"  Mean Pearson r: {self.correlation_results['pearson_r'].mean():.3f}")
        print(f"  Median Pearson r: {self.correlation_results['pearson_r'].median():.3f}")
        print(f"  Std Pearson r: {self.correlation_results['pearson_r'].std():.3f}")
        
        sig_count = (self.correlation_results['pearson_p'] < 0.05).sum()
        print(f"\n  Statistically significant (p<0.05): {sig_count}/{len(self.correlation_results)}")
        
        positive = (self.correlation_results['pearson_r'] > 0).sum()
        print(f"  Positive correlations: {positive}/{len(self.correlation_results)}")
        
        strong = (self.correlation_results['pearson_r'].abs() > 0.5).sum()
        print(f"  Strong correlations (|r|>0.5): {strong}/{len(self.correlation_results)}")
        
        print(f"\nTop 5 Strongest Positive Correlations:")
        top_5 = self.correlation_results.nlargest(5, 'pearson_r')
        for idx, row in top_5.iterrows():
            print(f"  {row['ped_station'][:40]:40s} r={row['pearson_r']:.3f} (p={row['pearson_p']:.4f})")
        
        print(f"\nInterpretation:")
        mean_r = self.correlation_results['pearson_r'].mean()
        if mean_r > 0.5:
            print("  Strong positive relationship between pedestrian traffic and bikeshare usage")
        elif mean_r > 0.3:
            print("  Moderate positive relationship between pedestrian traffic and bikeshare usage")
        elif mean_r > 0.1:
            print("  Weak positive relationship between pedestrian traffic and bikeshare usage")
        else:
            print("  Minimal relationship between pedestrian traffic and bikeshare usage")
        
        print("\n" + "="*60)
    
    def plot_spatial_map(self, save_path='RQ1_PLOTS/spatial_correlation_map.png'):
        """
        Create a map showing correlation strength by location
        """
        if self.correlation_results is None or self.station_coords is None:
            print("No results to map")
            return
        
        # Merge correlation results with pedestrian coordinates
        ped_coords = self.station_coords[self.station_coords['station_type'] == 'pedestrian'].copy()
        map_data = ped_coords.merge(
            self.correlation_results[['ped_station', 'pearson_r', 'pearson_p', 'n_observations']],
            left_on='station_name',
            right_on='ped_station',
            how='inner'
        )
        
        # Get bikeshare station coordinates
        bike_coords = self.station_coords[self.station_coords['station_type'] == 'bikeshare']
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Plot bikeshare stations as small gray points
        ax.scatter(bike_coords['longitude'], bike_coords['latitude'],
                  s=20, c='lightgray', alpha=0.3, label='Bikeshare Stations', zorder=1)
        
        # Plot pedestrian counters colored by correlation strength
        scatter = ax.scatter(map_data['longitude'], map_data['latitude'],
                           c=map_data['pearson_r'],
                           s=map_data['n_observations'] / 30,  # Size by sample size
                           cmap='RdYlGn', vmin=-0.2, vmax=1.0,
                           alpha=0.8, edgecolors='black', linewidth=2,
                           label='Pedestrian Counters', zorder=3)
        
        # Add location labels for significant correlations
        for idx, row in map_data.iterrows():
            if row['pearson_p'] < 0.05:  # Only label significant ones
                ax.annotate(row['station_name'].split(' - ')[0][:20],  # Shorten labels
                          (row['longitude'], row['latitude']),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8, alpha=0.7)
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Pearson Correlation (r)', fontsize=12)
        
        # Labels and title
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Spatial Distribution of Pedestrian-Bikeshare Correlations\n(Size = Sample Size, Color = Correlation Strength)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(alpha=0.3)
        
        # Add text box with statistics
        textstr = f'Mean r = {map_data["pearson_r"].mean():.3f}\n'
        textstr += f'Significant: {(map_data["pearson_p"] < 0.05).sum()}/{len(map_data)}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved spatial correlation map to {save_path}")
        plt.close()
        
        return fig
    
    def create_location_type_analysis(self, save_path='RQ1_PLOTS/location_type_analysis.png'):
        """
        Analyze correlations by location type (trails vs streets vs waterfront)
        """
        if self.correlation_results is None:
            print("No correlation results available")
            return
        
        # Categorize locations
        def categorize_location(name):
            name_lower = name.lower()
            if 'trail' in name_lower or 'park' in name_lower:
                return 'Trail/Park'
            elif 'wharf' in name_lower or 'waterfront' in name_lower:
                return 'Waterfront'
            else:
                return 'Street'
        
        results = self.correlation_results.copy()
        results['location_type'] = results['ped_station'].apply(categorize_location)
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Boxplot of correlations by location type
        ax = axes[0, 0]
        location_types = results['location_type'].unique()
        data_by_type = [results[results['location_type'] == lt]['pearson_r'].values 
                       for lt in location_types]
        
        bp = ax.boxplot(data_by_type, labels=location_types, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightsalmon']):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Pearson Correlation (r)', fontsize=11)
        ax.set_title('Correlation Distribution by Location Type', fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # 2. Sample counts by location type
        ax = axes[0, 1]
        type_counts = results['location_type'].value_counts()
        colors = ['lightblue', 'lightgreen', 'lightsalmon']
        ax.bar(type_counts.index, type_counts.values, color=colors[:len(type_counts)])
        ax.set_ylabel('Number of Locations', fontsize=11)
        ax.set_title('Sample Size by Location Type', fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        # Add count labels on bars
        for i, (idx, val) in enumerate(type_counts.items()):
            ax.text(i, val + 0.1, str(val), ha='center', fontweight='bold')
        
        # 3. Mean activity levels by location type
        ax = axes[1, 0]
        type_activity = results.groupby('location_type').agg({
            'mean_ped_count': 'mean',
            'mean_bike_trips': 'mean'
        })
        
        x = np.arange(len(type_activity))
        width = 0.35
        
        ax.bar(x - width/2, type_activity['mean_ped_count'], width, 
              label='Pedestrian Count', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, type_activity['mean_bike_trips'], width,
              label='Bike Trips', color='darkorange', alpha=0.8)
        
        ax.set_ylabel('Mean Hourly Count', fontsize=11)
        ax.set_title('Average Activity Levels by Location Type', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(type_activity.index)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')
        
        # 4. Statistical summary table
        ax = axes[1, 1]
        ax.axis('tight')
        ax.axis('off')
        
        summary_stats = []
        for lt in location_types:
            lt_data = results[results['location_type'] == lt]
            summary_stats.append([
                lt,
                len(lt_data),
                f"{lt_data['pearson_r'].mean():.3f}",
                f"{lt_data['pearson_r'].std():.3f}",
                f"{(lt_data['pearson_p'] < 0.05).sum()}/{len(lt_data)}"
            ])
        
        table = ax.table(cellText=summary_stats,
                        colLabels=['Location Type', 'n', 'Mean r', 'SD', 'Sig.'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.3, 0.1, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(5):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Statistical Summary by Location Type', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved location type analysis to {save_path}")
        plt.close()
        
        # Print statistical test
        from scipy.stats import kruskal
        groups = [results[results['location_type'] == lt]['pearson_r'].values 
                 for lt in location_types]
        if len(groups) > 1 and all(len(g) > 0 for g in groups):
            stat, p = kruskal(*groups)
            print(f"\nKruskal-Wallis Test: H={stat:.3f}, p={p:.4f}")
            if p < 0.05:
                print("  → Significant difference between location types")
            else:
                print("  → No significant difference between location types")
        
        return fig, results
    
    def create_statistical_summary_table(self, save_path='RQ1_PLOTS/statistical_summary_table.csv'):
        """
        Create a detailed statistical summary table for academic paper
        """
        if self.correlation_results is None:
            print("No correlation results available")
            return
        
        # Merge with spatial matches to get number of matched stations
        table_data = self.correlation_results.copy()
        
        # Add significance stars
        def add_stars(row):
            if row['pearson_p'] < 0.001:
                return '***'
            elif row['pearson_p'] < 0.01:
                return '**'
            elif row['pearson_p'] < 0.05:
                return '*'
            else:
                return ''
        
        table_data['significance'] = table_data.apply(add_stars, axis=1)
        
        # Categorize location type
        def categorize_location(name):
            name_lower = name.lower()
            if 'trail' in name_lower or 'park' in name_lower:
                return 'Trail/Park'
            elif 'wharf' in name_lower or 'waterfront' in name_lower:
                return 'Waterfront'
            else:
                return 'Street'
        
        table_data['location_type'] = table_data['ped_station'].apply(categorize_location)
        
        # Select and order columns
        summary_table = table_data[[
            'ped_station', 'location_type', 'n_observations', 
            'mean_ped_count', 'mean_bike_trips', 'pearson_r', 'pearson_p', 
            'significance', 'n_matched_bike_stations'
        ]].copy()
        
        # Rename columns for clarity
        summary_table.columns = [
            'Pedestrian Counter Location', 'Type', 'n_obs', 
            'Mean_Ped_Count', 'Mean_Bike_Trips', 'Pearson_r', 'p_value',
            'Sig', 'n_Bike_Stations'
        ]
        
        # Round numeric columns
        summary_table['Mean_Ped_Count'] = summary_table['Mean_Ped_Count'].round(1)
        summary_table['Mean_Bike_Trips'] = summary_table['Mean_Bike_Trips'].round(1)
        summary_table['Pearson_r'] = summary_table['Pearson_r'].round(3)
        summary_table['p_value'] = summary_table['p_value'].round(4)
        
        # Sort by correlation strength
        summary_table = summary_table.sort_values('Pearson_r', ascending=False)
        
        # Add summary statistics row
        summary_row = pd.DataFrame({
            'Pedestrian Counter Location': ['OVERALL (Mean ± SD)'],
            'Type': ['All'],
            'n_obs': [f"{table_data['n_observations'].mean():.0f} ± {table_data['n_observations'].std():.0f}"],
            'Mean_Ped_Count': [f"{table_data['mean_ped_count'].mean():.1f} ± {table_data['mean_ped_count'].std():.1f}"],
            'Mean_Bike_Trips': [f"{table_data['mean_bike_trips'].mean():.1f} ± {table_data['mean_bike_trips'].std():.1f}"],
            'Pearson_r': [f"{table_data['pearson_r'].mean():.3f} ± {table_data['pearson_r'].std():.3f}"],
            'p_value': ['-'],
            'Sig': [f"{(table_data['pearson_p'] < 0.05).sum()}/{len(table_data)} sig."],
            'n_Bike_Stations': [f"{table_data['n_matched_bike_stations'].mean():.1f} ± {table_data['n_matched_bike_stations'].std():.1f}"]
        })
        
        summary_table = pd.concat([summary_table, summary_row], ignore_index=True)
        
        # Save to CSV
        summary_table.to_csv(save_path, index=False)
        print(f"\nSaved statistical summary table to {save_path}")
        
        # Also print to console in nice format
        print("\n" + "="*100)
        print("STATISTICAL SUMMARY TABLE")
        print("="*100)
        print(summary_table.to_string(index=False))
        print("\nNote: *** p<0.001, ** p<0.01, * p<0.05")
        print("="*100)
        
        return summary_table


def main(bike_df, ped_df, distance_threshold=800):
    """
    Main analysis pipeline
    
    Parameters:
    -----------
    bike_df : DataFrame with bikeshare data
    ped_df : DataFrame with pedestrian data
    distance_threshold : meters for spatial matching (default 800m)
    """
    
    # Initialize analyzer
    analyzer = BikeSharePedestrianAnalyzer(bike_df, ped_df, distance_threshold)
    
    # Step 1: Clean data
    analyzer.clean_bikeshare_data()
    analyzer.clean_pedestrian_data()
    
    # Step 2: Geocode stations
    analyzer.geocode_stations()
    
    # Step 3: Create spatial matches
    analyzer.create_spatial_matches()
    print("\n>>> Calling prepare_pedestrian_long() now...")
    analyzer.prepare_pedestrian_long()

    # Step 4: Compute correlations
    analyzer.compute_correlations()
    
    # Step 5: Visualize results
    analyzer.plot_correlation_summary()
    
    # Step 6: NEW - Create minimum viable academic additions
    print("\n" + "="*70)
    print("GENERATING ACADEMIC ANALYSIS ADDITIONS")
    print("="*70)
    
    # Spatial map
    analyzer.plot_spatial_map()
    
    # Location type analysis
    analyzer.create_location_type_analysis()
    
    # Statistical summary table
    analyzer.create_statistical_summary_table()
    
    # Step 7: Generate summary
    analyzer.generate_summary_report()
    
    return analyzer
