

import os
import functools
from tcia_utils import nbia


# MIDI-B PHI Detection test / validation data sets.
collection_names = [
    "MIDI-B-Curated-Test",
    "MIDI-B-Curated-Validation",
    "MIDI-B-Synthetic-Test",
    "MIDI-B-Synthetic-Validation"
]

@functools.lru_cache(maxsize=1)
def get_all_series():
    """
    Return list of all series from all MIDI-B collections, caching the result.
    
    Returns:
        List of series dictionaries, each containing SeriesInstanceUID and other metadata
    """
    all_series = []
    for collection_name in collection_names:
        series = nbia.getSeries(collection=collection_name)
        if series is None:
            print(f"{collection_name}: No series found")
            continue
        all_series.extend(series)
    return all_series

def get_series_by_uid(series_instance_uid):
    """
    Find a specific series by SeriesInstanceUID.
    
    Args:
        series_instance_uid: The SeriesInstanceUID to find
        
    Returns:
        Series dictionary or None if not found
    """
    all_series = get_all_series()
    for series in all_series:
        if series.get('SeriesInstanceUID') == series_instance_uid:
            return series
    return None

def download_series(series):
    """
    Download a series as a zip file.
    
    Args:
        series: Either a series dictionary or a SeriesInstanceUID string
        
    Returns:
        Path to downloaded zip file, or None if download failed
    """
    # If series is a string, look it up
    if isinstance(series, str):
        series_uid = series
        series = get_series_by_uid(series_uid)
        if series is None:
            print(f"Series {series_uid} not found in any collection")
            return None
    
    # Download the series as a zip file
    out_dir = os.path.join(os.path.dirname(__file__), ".midi_b_data")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    try:
        nbia.downloadSeries([series], as_zip=True, path=out_dir)
        # Return path to the downloaded file (nbia creates zip with SeriesInstanceUID in name)
        return out_dir
    except Exception as e:
        print(f"Error downloading series: {e}")
        return None

def download_all():
    """Download all available series from MIDI-B collections."""
    for series in get_all_series():
        download_series(series)

if __name__ == "__main__":
    download_all()