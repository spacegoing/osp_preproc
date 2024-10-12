import json
import h5py
import decord
import multiprocessing
from multiprocessing import Pool
from decord import VideoReader
from decord import cpu

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

# Function to process a chunk of JSON data
def process_chunk(data_chunk):
    results = []

    for entry in data_chunk:
        video_path = entry["path"]
        filename = video_path.split("/")[-1]

        # Extract video info using decord
        nframes, fps, duration = extract_video_info(video_path)

        # Collect processed information
        processed_entry = {
            'path': video_path,
            'filename': filename,
            'cap': json.dumps(entry["cap"]),  # Store the list of captions as a JSON string
            'resolution': [entry["resolution"]["width"], entry["resolution"]["height"]],
            'fps': fps if fps is not None else entry.get("fps", 0.0),
            'duration': duration if duration is not None else entry.get("duration", 0.0),
            'nframes': nframes if nframes is not None else 0
        }
        results.append(processed_entry)

    return results

# Function to write the processed chunk back to HDF5
def write_results_to_hdf5(results, hdf5_file, start_idx):
    with h5py.File(hdf5_file, 'a') as h5f:
        paths_ds = h5f["path"]
        filenames_ds = h5f["filename"]
        captions_ds = h5f["cap"]
        resolutions_ds = h5f["resolution"]
        fps_ds = h5f["fps"]
        duration_ds = h5f["duration"]
        nframes_ds = h5f["nframes"]

        for i, entry in enumerate(results):
            idx = start_idx + i
            paths_ds[idx] = entry['path']
            filenames_ds[idx] = entry['filename']
            captions_ds[idx] = entry['cap']
            resolutions_ds[idx] = entry['resolution']
            fps_ds[idx] = entry['fps']
            duration_ds[idx] = entry['duration']
            nframes_ds[idx] = entry['nframes']


def process_json_to_hdf5(json_file, hdf5_file, ncpus=4):
    # Open the JSON file and load its content
    with open(json_file, 'r') as f:
        data = json.load(f)

    total_entries = len(data)
    chunk_size = total_entries // ncpus
    remainder = total_entries % ncpus

    # Create HDF5 file and initialize datasets with chunking
    with h5py.File(hdf5_file, 'w') as h5f:
        h5f.create_dataset("path", (total_entries,), dtype=h5py.string_dtype(), chunks=True)
        h5f.create_dataset("filename", (total_entries,), dtype=h5py.string_dtype(), chunks=True)
        h5f.create_dataset("resolution", (total_entries, 2), dtype='i', chunks=True)
        h5f.create_dataset("fps", (total_entries,), dtype='f', chunks=True)
        h5f.create_dataset("duration", (total_entries,), dtype='f', chunks=True)
        h5f.create_dataset("nframes", (total_entries,), dtype='i', chunks=True)
        h5f.create_dataset("cap", (total_entries,), dtype=h5py.string_dtype(encoding='utf-8'), chunks=True)

    # Create a list of chunks
    chunks = [(data[i:i + chunk_size], i) for i in range(0, total_entries, chunk_size)]
    if remainder > 0:
        chunks[-1] = (data[-chunk_size - remainder:], total_entries - chunk_size - remainder)

    # Use multiprocessing to process chunks in parallel
    with Pool(processes=ncpus) as pool:
        results = pool.map(process_chunk, [chunk[0] for chunk in chunks])

    # Write the results back to the HDF5 file in sequence
    for i, result in enumerate(results):
        start_idx = chunks[i][1]
        write_results_to_hdf5(result, hdf5_file, start_idx)

# Example usage
if __name__ == '__main__':
    json_file = '/workspace/host_folder/data_analysis/proc_osp/aes_le6.json'
    hdf5_file = '/workspace/host_folder/data_analysis/proc_osp/aes_le6.hdf5'
    ncpus = 110  # Set number of CPUs to use
    process_json_to_hdf5(json_file, hdf5_file, ncpus)
