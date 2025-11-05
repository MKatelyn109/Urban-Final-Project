import requests
import xml.etree.ElementTree as ET
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def download_file(url, dest_folder):
    """Download a single file with progress tracking"""
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_folder, filename)
    
    # Skip if already downloaded
    if os.path.exists(filepath):
        print(f"Skipping {filename} (already exists)")
        return filename, True
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        return filename, True
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return filename, False

def get_zip_urls_from_s3(bucket_name):
    """Extract all ZIP file URLs from S3 bucket using XML API"""
    base_url = f'https://{bucket_name}.s3.amazonaws.com'
    zip_urls = []
    marker = None
    
    while True:
        # Build URL with marker for pagination
        if marker:
            list_url = f'{base_url}/?marker={marker}'
        else:
            list_url = f'{base_url}/'
        
        try:
            response = requests.get(list_url)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Define namespace
            ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
            
            # Find all Contents elements (files)
            contents = root.findall('s3:Contents', ns)
            
            if not contents:
                # Try without namespace
                contents = root.findall('Contents')
            
            for content in contents:
                key_elem = content.find('s3:Key', ns)
                if key_elem is None:
                    key_elem = content.find('Key')
                
                if key_elem is not None:
                    key = key_elem.text
                    if key and key.endswith('.zip'):
                        file_url = f'{base_url}/{key}'
                        zip_urls.append(file_url)
                        marker = key  # Update marker for pagination
            
            # Check if there are more results
            is_truncated = root.find('s3:IsTruncated', ns)
            if is_truncated is None:
                is_truncated = root.find('IsTruncated')
            
            if is_truncated is None or is_truncated.text != 'true':
                break
                
        except Exception as e:
            print(f"Error fetching bucket contents: {e}")
            break
    
    return zip_urls

def main():
    bucket_name = 'capitalbikeshare-data'
    dest_folder = 'capitalbikeshare_data'
    
    # Create destination folder
    os.makedirs(dest_folder, exist_ok=True)
    
    print("Fetching list of ZIP files from S3 bucket...")
    zip_urls = get_zip_urls_from_s3(bucket_name)
    
    if not zip_urls:
        print("No ZIP files found. The bucket might be private or the structure is different.")
        print("Trying alternative method...")
        return
    
    print(f"Found {len(zip_urls)} ZIP files")
    
    # Show first few files
    print("\nFirst few files:")
    for url in zip_urls[:5]:
        print(f"  - {url.split('/')[-1]}")
    if len(zip_urls) > 5:
        print(f"  ... and {len(zip_urls) - 5} more")
    
    # Download files with parallel threads
    max_workers = 5  # Adjust based on your connection
    
    print("\nStarting downloads...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, url, dest_folder): url 
                   for url in zip_urls}
        
        with tqdm(total=len(zip_urls), desc="Downloading") as pbar:
            for future in as_completed(futures):
                filename, success = future.result()
                pbar.update(1)
    
    print(f"\nDownload complete! Files saved to '{dest_folder}' folder")

if __name__ == '__main__':
    main()