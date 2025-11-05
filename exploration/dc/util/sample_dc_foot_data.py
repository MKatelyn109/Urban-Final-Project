# Read the CSV
import pandas as pd

# Read with UTF-16 encoding and skip empty lines
df = pd.read_csv('Data/dc_foot/dc_foot_data.csv', 
                 encoding='utf-16',
                 sep='\t',  # Looks like tab-separated based on the spacing
                 skip_blank_lines=True)

# Clean up column names (remove extra spaces)
df.columns = df.columns.str.strip()
    
df['Datetime'] = pd.to_datetime(df['Hour of Datetime'], format='%B %d, %Y at %I %p')
# Filter for 2017-2019 (inclusive)
df_filtered = df[(df['Datetime'] >= '2021-01-01') & (df['Datetime'] < '2021-12-31')]
df_filtered = df_filtered.drop(columns='Hour of Datetime')
# df_filtered = df_filtered.drop(columns=df_filtered.columns[0]) # not working
df_filtered.to_csv('Data/sample_dc_foot/dc_foot_data_20_21.csv')