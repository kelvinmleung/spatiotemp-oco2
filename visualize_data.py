import numpy as np

filepath = '/Users/Camila/Desktop/OCO-2 UROP/spatiotemp-oco2/example_oco2_linear_model_201612_wollongong.npy'
data = np.load(filepath)


print(data)

import pandas as pd
file_path = '/Users/Camila/Desktop/OCO-2 UROP/spatiotemp-oco2/linear_oco_model/lnd_nadir_201510_lamont_reference_locations.csv'
df = pd.read_csv(file_path)
max_lat = df['Latitude'].max()
min_lat = df['Latitude'].min()
max_long = df['Longitude'].max()
min_long = df['Longitude'].min()
# Print the results
print(f"Max Latitude: {max_lat}")
print(f"Min Latitude: {min_lat}")
print(f"Max Longitude: {max_long}")
print(f"Min Longitude: {min_long}")