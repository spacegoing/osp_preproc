import duckdb


class DuckDBReader:
  def __init__(self, db_file, table_name='sorted_videos'):
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

  def get_row_by_number(self, row_no):
    """
    Retrieve a specific row by its row number.
    Returns a dictionary corresponding to the row.
    """
    query = "SELECT * FROM sorted_videos WHERE row_no = ?;"
    result = self.conn.execute(query, [row_no]).fetchone()
    columns = self.conn.execute(
      "PRAGMA table_info('sorted_videos')"
    ).fetchall()

    # Extract column names
    column_names = [col[1] for col in columns]

    # Convert the result to a dictionary
    if result:
      return dict(zip(column_names, result))
    else:
      return None

  def close(self):
    """
    Close the DuckDB connection.
    """
    self.conn.close()


if __name__ == "__main__":
  # Initialize the reader
  db_file = "/workspace/host_folder/data_analysis/proc_osp/pre_proc.duckdb"
  reader = DuckDBReader(db_file)

  # Get a specific row by its row number
  row_data = reader.get_row_by_number(1)
  print(row_data)

  # Get all data
  all_data = reader.get_all_data()
  print(all_data)

  # Close the reader
  reader.close()
