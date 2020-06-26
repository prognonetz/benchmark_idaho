# -----------------------------------------
# observational data only
from prognonetz_idaho.observational_only import PrognonetzDataSet

dataset = PrognonetzDataSet('data/', small=True, download=True)

print(dataset[0][0].shape, dataset[0][1].shape)  # (48, 32) (48, 6)

# -----------------------------------------
# nwp data only
from prognonetz_idaho.nwp_only import PrognonetzDataSet

dataset = PrognonetzDataSet('data/', small=True, download=True)

print(dataset[0][0].shape, dataset[0][1].shape)  # (48, 16, 14, 3) (48, 6)

# -----------------------------------------
# all features and labels with no processing
from prognonetz_idaho.full_dataset import PrognonetzDataSet

dataset = PrognonetzDataSet('data/', small=True, download=True)

print(dataset[0][0][0].shape, dataset[0][0][1].shape, dataset[0][1].shape)  # (48, 16, 14, 3) (48, 32) (48, 6)

# -----------------------------------------
# an example of feature and label selection
from prognonetz_idaho.full_dataset import PrognonetzDataSet

# select temperature and humidity observation at station 690
features_obs = ['690_temperature_15m', '690_humidity']
# select temperature nwp only
features_nwp = ['TMP']

dataset = PrognonetzDataSet('data/', input_shape_obs=features_obs, input_shape_nwp=features_nwp, small=True, download=True)

print(dataset[0][0][0].shape, dataset[0][0][1].shape, dataset[0][1].shape)  # (48, 16, 14, 1) (48, 2) (48, 6)

# -----------------------------------------
# an example of feature and label selection including vectorization of wind components
from prognonetz_idaho.full_dataset import PrognonetzDataSet
from prognonetz_idaho import processing

# select wind as x/y-components
features_obs = [processing.Vector(magnitude='690_wind_speed', direction='690_wind_direction')]
# select wind as x/y-components
features_nwp = [processing.Vector(magnitude='WSPD', direction='WDIR')]

dataset = PrognonetzDataSet('data/', input_shape_obs=features_obs, input_shape_nwp=features_nwp, small=True, download=True)

print(dataset[0][0][0].shape, dataset[0][0][1].shape, dataset[0][1].shape)  # (48, 16, 14, 2) (48, 2) (48, 6)


# -----------------------------------------
# an example of using nwp only of nearest neighbour
from prognonetz_idaho.full_dataset import PrognonetzDataSet
from prognonetz_idaho import processing

# select wind as x/y-components
features_nwp = [processing.Vector(magnitude='WSPD', direction='WDIR')]

dataset = PrognonetzDataSet('data/', obs=False, nearestneighbour=(4,5), input_shape_nwp=features_nwp, small=True, download=True)

print(dataset[0][0].shape, dataset[0][1].shape)  # (48, 2) (48, 6)

# -----------------------------------------
# an example of using processing applied to features and labels
from prognonetz_idaho.full_dataset import PrognonetzDataSet
from prognonetz_idaho import processing
import torch.utils.data as data

dataset = PrognonetzDataSet('./data',
split='train',
small=True,
download=True,
output_shape=['690_CIT'],
# feature selection
input_shape_obs=['690_temperature_15m', processing.Vector(magnitude='690_wind_speed', direction='690_wind_direction')],
input_shape_nwp=['TMP', processing.Vector(magnitude='WSPD', direction='WDIR')],
# Processing labels
output_transform=processing.Normalize(minimum_value=0, maximum_value=3775.0972336), 
# Processing features
input_transform_nwp=processing.Standardize(mean=[280.39971907, -0.6956181, -1.69519702], standard_deviation=[11.47811145, 3.22773494, 2.57422153]),
# Processing features
input_transform_obs=processing.Standardize(mean=[8.10344942, -0.83212702, -1.57366791], standard_deviation=[11.46412673, 3.38725304, 3.48667773]),
)

print(dataset[0][0][0].shape, dataset[0][0][1].shape, dataset[0][1].shape)  # (48, 16, 14, 3) (48, 3) (48, 1)
