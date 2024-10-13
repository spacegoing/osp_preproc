import duckdb


class DuckDBReader:
  def __init__(self, db_file, table_name="sorted_videos"):
    """
    Initialize the DuckDBReader with the path to the DuckDB file.
    """
    self.conn = duckdb.connect(db_file)
    self.table_name = table_name

  def get_all_data(self):
    """
    Retrieve all data from the 'sorted_videos' table.
    Returns a list of dictionaries where each dictionary corresponds to a row.
    """
    query = f"SELECT * FROM {self.table_name};"
    result = self.conn.execute(query).fetchall()
    columns = self.conn.execute(
      f"PRAGMA table_info({self.table_name})"
    ).fetchall()

    # Extract column names
    column_names = [col[1] for col in columns]

    # Convert result to list of dictionaries
    return [dict(zip(column_names, row)) for row in result]

  def get(self, idx, key=[]):
    """
    Retrieve a specific row by its row number.
    If key is empty, retrieve all columns.
    If key is a string, retrieve only that column.
    If key is a list of strings, retrieve those columns.
    Always returns a dictionary for consistency.
    """
    if not key:
      # Get all columns
      query = (
        f"SELECT * FROM {self.table_name} WHERE row_no = ?;"
      )
      columns = self.get_column_names()
    elif isinstance(key, str):
      # Get a single column
      query = (
        f"SELECT {key} FROM {self.table_name} WHERE row_no = ?;"
      )
      columns = [key]
    elif isinstance(key, list):
      # Get multiple columns
      key_str = ", ".join(key)
      query = f"SELECT {key_str} FROM {self.table_name} WHERE row_no = ?;"
      columns = key
    else:
      raise ValueError(
        "Invalid type for key. Must be string, list, or empty."
      )

    # Execute the query and fetch the result
    result = self.conn.execute(query, [idx]).fetchone()

    # If no result is found, return None
    if result is None:
      return None

    # Return the result as a dictionary
    return dict(zip(columns, result))

  def get_column_names(self):
    """
    Helper function to retrieve all column names from the table.
    """
    columns = self.conn.execute(
      f"PRAGMA table_info({self.table_name})"
    ).fetchall()
    return [col[1] for col in columns]

  def get_kv(self, key, value):
    """
    Retrieve all rows where the specified key matches the given value.
    Returns a list of dictionaries for the matching rows.
    """
    query = f"SELECT * FROM {self.table_name} WHERE {key} = ?;"
    result = self.conn.execute(query, [value]).fetchall()
    if not result:
      return None

    columns = self.get_column_names()
    return [dict(zip(columns, row)) for row in result]

  def __len__(self):
    """
    Return the number of rows in the table.
    """
    query = f"SELECT COUNT(*) FROM {self.table_name};"
    result = self.conn.execute(query).fetchone()
    return result[0]

  def close(self):
    """
    Close the DuckDB connection.
    """
    self.conn.close()


if __name__ == "__main__":
  # Initialize the reader
  db_file = "/workspace/host_folder/data_analysis/hdf5_proc/pre_proc.duckdb"
  reader = DuckDBReader(db_file)

  # Get a specific row by its row number
  row_data = reader.get(1)
  print(row_data)

  # Get a specific key from a row
  fps_data = reader.get(1, key="fps")
  print(fps_data)

  # Get multiple keys from a row
  specific_data = reader.get(
    1, key=["filename", "fps", "nframes"]
  )
  print(specific_data)

  # Get the number of rows in the table
  print(f"Number of rows: {len(reader)}")

  # Close the reader
  reader.close()
