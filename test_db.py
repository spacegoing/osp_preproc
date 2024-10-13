import json
import multiprocessing
from multiprocessing import Pool
import decord
from decord import VideoReader, cpu
import duckdb
import os

# Initialize Decord to use CPU
decord.bridge.set_bridge("native")


# Find closest `y` based on the provided nframes
def find_closest_y(x, vae_stride_t=4, model_ds_t=4):
  if x < 29:
    return -1

  y = x - (x - 1) % vae_stride_t
  remainder = ((y - 1) // vae_stride_t + 1) % model_ds_t
  if remainder != 0:
    y -= remainder * vae_stride_t

  return y


def infer_duckdb_type(value):
  """
  Infers the DuckDB type based on the Python type of the value.
  """
  if isinstance(value, str):
    return "VARCHAR"
  elif isinstance(value, float):
    return "DOUBLE"
  elif isinstance(value, int):
    return "INTEGER"
  elif isinstance(value, list):
    if all(isinstance(item, str) for item in value):
      return "VARCHAR[]"  # 'cap' is list of strings [caption1, caption2, etc]
    else:
      raise ValueError(
        f"Unsupported list data type in DuckDB for: {value}"
      )
  else:
    raise ValueError(
      f"Unsupported data type: {type(value)} for value {value}"
    )


# %% Define function to extract video information
def extract_video_info(video_path):
  try:
    vr = VideoReader(video_path, ctx=cpu(0))
    nframes = len(vr)
    fps = float(vr.get_avg_fps())
    duration = nframes / fps
    width, height = vr[0].shape[1], vr[0].shape[0]
    aspect_ratio = height / width
    return nframes, fps, duration, (width, height), aspect_ratio
  except Exception as e:
    print(f"Error reading video {video_path}: {e}")
    return None, None, None, None, None


# Function to process a chunk of JSON data
def process_chunk(
  data_chunk, vae_stride_t, model_ds_t, log_file
):
  results = []

  for entry in data_chunk:
    video_path = entry["path"]
    filename = video_path.split("/")[-1]

    # Extract video info using decord
    nframes, fps, duration, resolution, aspect_ratio = (
      extract_video_info(video_path)
    )

    # If any None values are returned, log the failure and continue to the next entry
    if None in (
      nframes,
      fps,
      duration,
      resolution,
      aspect_ratio,
    ):
      with open(log_file, "a") as log:
        log.write(f"Failed to process video: {video_path}\n")
      continue  # Skip this entry

    # Find the closest `y` for the `nframes`
    closest_y = find_closest_y(nframes, vae_stride_t, model_ds_t)

    # Construct the field name dynamically
    bucket_frame_field = (
      f"bucket_frame_vst_{vae_stride_t}_mst_{model_ds_t}"
    )

    # Collect processed information
    processed_entry = {
      **entry,  # Copy all keys from the original entry
      "path": video_path,
      "filename": filename,
      "nframes": nframes,
      "fps": fps,
      "duration": duration,
      "resolution_width": resolution[0],
      "resolution_height": resolution[1],
      "aspect_ratio": aspect_ratio,
      bucket_frame_field: closest_y,
    }

    results.append(processed_entry)

  return results


def write_results_to_duckdb(results, db_file):
  conn = duckdb.connect(db_file)

  # Convert entries to the format DuckDB can ingest (list of tuples)
  field_names = list(
    results[0].keys()
  )  # Extract the field names dynamically from the first result
  field_values = [tuple(entry.values()) for entry in results]

  # Infer the types based on the first entry's values
  first_entry = results[0]
  field_types = [
    f"{field} {infer_duckdb_type(first_entry[field])}"
    for field in field_names
  ]

  # Create table if it doesn't exist
  conn.execute(f"""
        CREATE TABLE IF NOT EXISTS videos (
            {', '.join(field_types)}
        );
    """)

  # Insert data into the table
  placeholders = ", ".join(["?" for _ in field_names])
  conn.executemany(
    f"INSERT INTO videos VALUES ({placeholders});", field_values
  )

  conn.close()


def process_json_to_duckdb(
  json_file,
  db_file,
  ncpus=4,
  vae_stride_t=4,
  model_ds_t=4,
  log_file="error_log.txt",
):
  # Open the JSON file and load its content
  with open(json_file, "r") as f:
    data = json.load(f)

  # Remove existing log file if it exists
  if os.path.exists(log_file):
    os.remove(log_file)

  total_entries = len(data)
  # Create a list of chunks
  chunk_size = (total_entries + ncpus - 1) // ncpus  # Properly handle remainder
  # Create a list of chunks
  chunks = [data[i:i + chunk_size] for i in range(0, total_entries, chunk_size)]

  # Use multiprocessing to process chunks in parallel
  with Pool(processes=ncpus) as pool:
    results_list = pool.starmap(
      process_chunk,
      [
        (chunk, vae_stride_t, model_ds_t, log_file)
        for chunk in chunks
      ],
    )

  # Write the results to the DuckDB database
  for results in results_list:
    if results:
      write_results_to_duckdb(results, db_file)

  # Final step: Sort the DuckDB table by `bucket_frame_field`, `aspect_ratio`, and `filename`
  bucket_frame_field = (
    f"cut_frame_vst_{vae_stride_t}_mst_{model_ds_t}"
  )
  conn = duckdb.connect(db_file)

  # Create a sorted table and add row numbers
  conn.execute(f"""
      CREATE TABLE sorted_videos AS
      SELECT *, ROW_NUMBER() OVER (ORDER BY {bucket_frame_field}, aspect_ratio, filename) AS row_no
      FROM videos;
  """)

  # Drop the original `videos` table after sorting
  conn.execute("DROP TABLE videos;")
  conn.close()


# Example usage
if __name__ == "__main__":
  base_path = "/workspace/host_folder/data_analysis/proc_osp/"
  json_file = base_path + "aes_le6.json"
  # base_path = "./"
  # json_file = "/workspace/Open-Sora-Plan/mydata/sub_cluster_hier2_5_6_7.json"
  db_file = base_path + "pre_proc.duckdb"
  log_file = base_path + "error_video.log"
  ncpus = 110  # Adjust the number of CPUs as needed
  vae_stride_t = 4  # Example value for vae_stride_t
  model_ds_t = 4  # Example value for model_ds_t
  process_json_to_duckdb(
    json_file, db_file, ncpus, vae_stride_t, model_ds_t, log_file
  )
