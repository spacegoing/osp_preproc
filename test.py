import json
import h5py
import decord
from decord import VideoReader
from decord import cpu
import numpy as np

# Initialize Decord to use CPU
decord.bridge.set_bridge("native")

# Define function to extract video information
def extract_video_info(video_path):
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        nframes = len(vr)
        fps = float(vr.get_avg_fps())
        duration = nframes / fps
        return nframes, fps, duration
    except Exception as e:
        print(f"Error reading video {video_path}: {e}")
        return None, None, None

# Path to your JSON file
json_file = '/workspace/Open-Sora-Plan/mydata/sub_cluster_hier2_5_6_7.json'

# Path to the output HDF5 file
hdf5_file = 'test.hdf5'

# Open the JSON file and load its content
with open(json_file, 'r') as f:
    data = json.load(f)

# Create HDF5 file with chunking enabled
with h5py.File(hdf5_file, 'w') as h5f:
    # Define datasets in HDF5 file
    paths_ds = h5f.create_dataset("path", (len(data),), dtype=h5py.string_dtype(), chunks=True)
    filenames_ds = h5f.create_dataset("filename", (len(data),), dtype=h5py.string_dtype(), chunks=True)
    resolutions_ds = h5f.create_dataset("resolution", (len(data), 2), dtype='i', chunks=True)
    fps_ds = h5f.create_dataset("fps", (len(data),), dtype='f', chunks=True)
    duration_ds = h5f.create_dataset("duration", (len(data),), dtype='f', chunks=True)
    nframes_ds = h5f.create_dataset("nframes", (len(data),), dtype='i', chunks=True)
    
    # To store lists of captions, we use a variable-length string dataset with `dtype=h5py.special_dtype(vlen=str)`
    captions_ds = h5f.create_dataset("cap", (len(data),), dtype=h5py.string_dtype(encoding='utf-8'), chunks=True)

    # Iterate through each entry in JSON and process
    for i, entry in enumerate(data):
        video_path = entry["path"]
        filename = video_path.split("/")[-1]
        
        # Extract video info using decord
        nframes, fps, duration = extract_video_info(video_path)

        # Store the processed information into HDF5 datasets
        paths_ds[i] = video_path
        filenames_ds[i] = filename
        captions_ds[i] = json.dumps(entry["cap"])  # Store the list of captions directly
        resolutions_ds[i] = [entry["resolution"]["width"], entry["resolution"]["height"]]
        fps_ds[i] = fps if fps is not None else entry.get("fps", 0.0)
        duration_ds[i] = duration if duration is not None else entry.get("duration", 0.0)
        nframes_ds[i] = nframes if nframes is not None else 0

print("Conversion to HDF5 with captions stored as lists completed.")
