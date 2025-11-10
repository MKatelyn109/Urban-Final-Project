import pandas as pd
import glob
import os
from pathlib import Path

def combine_bikeshare_data(data_folder, output_file='master_bikeshare.csv'):
    """
    Combine all Capital Bikeshare CSV files into a master file
    Handles both quarterly (2017) and monthly (2018+) formats
    """
    
    # Find all CSV files matching the patterns
    # eight_files = glob.glob(os.path.join(data_folder, '2020*-capitalbikeshare-tripdata.csv'), recursive=True)
    nine_files = glob.glob(os.path.join(data_folder, '2025*citibike-tripdata_1.csv'), recursive=True)
    
    all_files = sorted(nine_files)
    
    if not all_files:
        print(f"No CSV files found in {data_folder}")
        print("Make sure the files are extracted from the ZIP files.")
        return
    
    print(f"Found {len(all_files)} CSV files to combine")
    print(f"\nFirst few files:")
    for f in all_files[:5]:
        print(f"  - {Path(f).name}")
    if len(all_files) > 5:
        print(f"  ... and {len(all_files) - 5} more\n")
    
    # Read and combine all files
    dfs = []
    errors = []
    
    for i, file in enumerate(all_files, 1):
        try:
            print(f"Reading {i}/{len(all_files)}: {Path(file).name}", end='\r')
            
            # Read CSV - some files might have encoding issues
            try:
                df = pd.read_csv(file)
            except UnicodeDecodeError:  
                df = pd.read_csv(file, encoding='latin-1')
            
            # Add source file column for tracking
            df['source_file'] = Path(file).name
            
            dfs.append(df)
            
        except Exception as e:
            errors.append((Path(file).name, str(e)))
            print(f"\nError reading {Path(file).name}: {e}")
    
    if not dfs:
        print("\nNo data was successfully read!")
        return
    
    print(f"\n\nCombining {len(dfs)} dataframes...")
    
    # Combine all dataframes
    master_df = pd.concat(dfs, ignore_index=True)
    
    print(f"\nMaster dataset created:")
    print(f"  Total rows: {len(master_df):,}")
    print(f"  Total columns: {len(master_df.columns)}")
    print(f"  Date range: {master_df['source_file'].min()} to {master_df['source_file'].max()}")
    print(f"\nColumns: {', '.join(master_df.columns.tolist())}")
    
    # Save to CSV
    print(f"\nSaving to {output_file}...")
    master_df.to_csv(output_file, index=False)
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"✓ Saved successfully! File size: {file_size_mb:.2f} MB")
    
    if errors:
        print(f"\n⚠ Errors encountered in {len(errors)} files:")
        for filename, error in errors:
            print(f"  - {filename}: {error}")
    
    # Show memory usage
    memory_mb = master_df.memory_usage(deep=True).sum() / (1024 * 1024)
    print(f"\nMemory usage: {memory_mb:.2f} MB")
    
    return master_df

if __name__ == '__main__':
    # Set your data folder path here
    data_folder = 'Data/dc_bikeshare'  # Change this to your extracted files location
    
    # Combine all data
    master_df = combine_bikeshare_data(data_folder, output_file='Data/sample_dc_bikeshare/SAMPLE_DC_BIKE.csv')
    
    if master_df is not None:
        print("\n" + "="*50)
        print("Preview of combined data:")
        print("="*50)
        print(master_df.head())