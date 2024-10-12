import h5py
import json


import h5py
import json

class HDF5Entity:
    """Represents a lazily loaded entity from the HDF5 file."""
    def __init__(self, hdf5_file, idx):
        self.hdf5_file = hdf5_file
        self.idx = idx
        self._cache = {}


    def __repr__(self):
        filename = self.hdf5_file.filename  # Extract just the file name
        return f"<HDF5Entity: file='{filename}', index={self.idx}>"


    def __getitem__(self, key):
        # Lazy load the key only when it's accessed
        if key not in self._cache:
            value = self.hdf5_file[key][self.idx]
            if isinstance(value, bytes):
                # Automatically decode any byte string to a regular string
                self._cache[key] = value.decode('utf-8')
            elif key == 'cap':
                # Parse JSON string to list for 'cap'
                self._cache[key] = json.loads(value)
            else:
                # Store the value directly for other types (int, float, etc.)
                self._cache[key] = value
        return self._cache[key]


    def get_entity(self):
        """Load and return all keys and their values as a dictionary."""
        entity_data = {}
        for key in self.hdf5_file.keys():
            entity_data[key] = self[key]  # Access the data using lazy loading
        return entity_data


class HDF5Reader:
    """Reader class for lazily accessing entries in an HDF5 file."""
    def __init__(self, hdf5_file_path):
        # Open the HDF5 file in read mode
        self.hdf5_file = h5py.File(hdf5_file_path, 'r')

    def __getitem__(self, idx):
        # Return a lazy-loaded HDF5Entity object
        return HDF5Entity(self.hdf5_file, idx)

    def get_slice(self, start, end):
        """Return a list of dictionaries with fully loaded data between `start` and `end`."""
        entities = []
        for i in range(start, end):
            entity = self[i].get_entity()  # Load the full entity as a dict
            entities.append(entity)
        return entities

    def keys(self):
        """Return all keys (dataset names) and their types in the HDF5 file."""
        return {key: str(self.hdf5_file[key].dtype) for key in self.hdf5_file.keys()}

    def close(self):
        """Close the HDF5 file."""
        self.hdf5_file.close()


if __name__ == '__main__':
  # %% Example usage
  hdf5_file_path = 'test.hdf5'
  hdf5_file_path = '/workspace/host_folder/data_analysis/proc_osp/aes_le6.hdf5'
  reader = HDF5Reader(hdf5_file_path)

  # Access the 10th entry lazily
  entry = reader[10]
  print(entry['path'])      # This will load the 'path' lazily
  print(entry['nframes'])   # This will load 'nframes' lazily

  # Get fully loaded data for the 10th entry
  fully_loaded_entity = entry.get_entity()
  print(fully_loaded_entity)

  # Get a slice of entries from 5th to 15th
  slice_entries = reader.get_slice(180, 185)
  print(slice_entries)  # Fully loaded paths from the slice
  for entity in slice_entries:
      print(entity['path'])  # Fully loaded paths from the slice

  # List all keys and their types in the HDF5 file
  keys = reader.keys()
  print(keys)

  # # Close the reader when done
  # reader.close()
