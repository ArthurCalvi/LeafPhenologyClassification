import numpy as np
from datetime import datetime
from tqdm import tqdm
import os
import rasterio
from skimage.morphology import dilation, disk
from skimage.filters import rank
import pandas as pd 

def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

def load_folder(folder, func=None, func_args=None, disable=True):
    files = os.listdir(folder)
    files = [f for f in files if (f.endswith('.tif') or f.endswith('.tiff') or f.endswith('.png'))]
    files.sort()
    data = []
    for file in tqdm(files, disable=disable):
        with rasterio.open(os.path.join(folder, file)) as src:
            mask = src.read().squeeze()
            if func is not None:
                mask = func(mask, **func_args)
            data.append(mask)
    return np.array(data)

def postprocess_cloud_mask(cloud_mask : np.array, n : int = 5, nm : int = 20) -> np.array:
    dilated = dilation(cloud_mask, disk(n))
    dilated = dilated.astype(float)
    mean = rank.mean(dilated, disk(nm)) / 255
    return mean.astype('float32')

from scipy.ndimage import gaussian_filter, sobel

def get_aspect(dem):
    dem_smoothed = gaussian_filter(dem, sigma=1)

    # Calculate the gradient
    dx = sobel(dem_smoothed, axis=1)  # Gradient in x direction
    dy = sobel(dem_smoothed, axis=0)  # Gradient in y direction

    # Compute the aspect
    aspect = np.arctan2(dy, -dx)  # Aspect in radians
    aspect = np.degrees(aspect)  # Convert to degrees
    aspect = np.where(aspect < 0, 360 + aspect, aspect)  # Normalize to 0-360 degrees
    aspect[ dem_smoothed == 0 ] = np.nan

    return aspect

#slope 

def datetime_to_ordinal(dates):
    """Convert datetime objects to days since the first date in the list."""
    base_date = dates[0]
    return np.array([(date - base_date).days for date in dates]) 


def calculate_slope_with_dates(tree_cover_timeseries, dates, n, K):
    """
    Calculate the slope of linear regression over tree cover data using actual dates for each time step.

    Parameters:
    - tree_cover_timeseries: 3D numpy array [time, height, width]
    - dates: list of datetime objects corresponding to each time step
    - n: int, central time index
    - K: int, window size around the central index

    Returns:
    - slope_map: 2D numpy array [height, width] representing the slope of the tree cover trend
    """
    # Define the time window
    start_index = max(n - K, 0)
    end_index = min(n + K + 1, tree_cover_timeseries.shape[0])

    # Convert dates to ordinal days
    date_nums = datetime_to_ordinal(dates[start_index:end_index])

    # Create the time indices matrix
    t = date_nums.reshape(-1, 1)  # Column vector
    X = np.hstack([t, np.ones((t.shape[0], 1))])  # Add a column of ones for the intercept

    # Reshape the data so each pixel's time series is a column
    Y = tree_cover_timeseries[start_index:end_index].reshape(t.shape[0], -1)

    # Perform matrix multiplication for the linear regression coefficients
    XT_X = X.T @ X
    XT_X_inv = np.linalg.inv(XT_X)
    XT_Y = X.T @ Y
    beta = XT_X_inv @ XT_Y

    # Extract the slope (first row of beta)
    slopes = beta[0, :].reshape(tree_cover_timeseries.shape[1], tree_cover_timeseries.shape[2])

    return slopes # %/day

import numpy as np
from datetime import datetime

def fit_periodic_function(time_series, qa, dates):
    """
    Fits a periodic function of the form A*cos(2*pi*t) + B*sin(2*pi*t) + C to a 3D time series data
    weighted by a quality assessment array using the provided dates.

    Parameters:
    - time_series: np.ndarray, shape (time, width, height), the RGB channel data.
    - qa: np.ndarray, shape (time, width, height), quality assessment data with values between 0 and 1.
    - dates: list of datetime.datetime, dates corresponding to the time series data points.

    Returns:
    - amplitude_map: np.ndarray, amplitude of the fitted function.
    - phase_map: np.ndarray, phase of the fitted function.
    - offset_map: np.ndarray, constant offset of the fitted function.
    """
    # Convert dates to 'datetime64' and compute normalized time as fraction of year
    times_datetime64 = np.array(dates, dtype='datetime64[D]')
    start_date = times_datetime64[0]
    days_since_start = (times_datetime64 - start_date).astype(int)
    t_normalized = days_since_start / 365.25  # Normalize to fraction of year

    # Convert normalized time to radians
    t_radians = 2 * np.pi * t_normalized

    # Create the design matrix
    A = np.stack([np.cos(t_radians), np.sin(t_radians), np.ones_like(t_radians)], axis=-1)

    # Reshape time_series and qa for vectorized operations
    pixels = time_series.reshape(time_series.shape[0], -1)
    weights = qa.reshape(qa.shape[0], -1)

    # Broadcasting for weighted design matrix
    A_expanded = np.expand_dims(A, 2)
    weights_expanded = np.expand_dims(weights, 1)
    A_weighted = A_expanded * weights_expanded

    # Compute the normal equation components
    ATA = np.einsum('ijk,ilk->jlk', A_weighted, A_expanded)
    ATb = np.einsum('ijk,ik->jk', A_weighted, pixels)

    # Solve for parameters
    ATA_reshaped = ATA.transpose(2, 0, 1)
    ATb_reshaped = ATb.T
    params = np.array([solve_params(ATA_reshaped[i], ATb_reshaped[i]) for i in range(ATA_reshaped.shape[0])])
    params_reshaped = params.reshape(time_series.shape[1], time_series.shape[2], 3).transpose(2, 0, 1)
    A_params, B_params, C_params = params_reshaped

    # Calculate amplitude and phase
    M = np.sqrt(A_params**2 + B_params**2)
    phi = np.arctan2(B_params, A_params)

    # Calculate day offset in radians (assuming 30 days per month)
    first_day_offset = 2 * np.pi * (dates[0].day - 1) / 30
    
    # Adjust and normalize phase
    phi_adjusted = (phi - (2 * np.pi * t_normalized[0])) % (2 * np.pi)
    phi_normalized = np.where(phi_adjusted > np.pi, phi_adjusted - 2 * np.pi, phi_adjusted)


    # Reshape M and phi to match original spatial dimensions
    amplitude_map = M.reshape(time_series.shape[1], time_series.shape[2])
    phase_map = phi_normalized.reshape(time_series.shape[1], time_series.shape[2])
    offset_map = C_params.reshape(time_series.shape[1], time_series.shape[2])

    return amplitude_map, phase_map, offset_map

def solve_params(ATA, ATb):
    """ Solve linear equations with error handling for non-invertible cases. """
    try:
        return np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        return np.full(ATb.shape, np.nan)  # Return NaN for non-invertible matrices

#LOADING 

# Load the data and preprocess it
def load_data_from_tile(path: str) -> dict:
    tile_id = os.path.basename(path).split('_')[1]
    dates = [datetime.strptime(filename.split('_')[0], '%Y-%m-%d') for filename in os.listdir(os.path.join(path, 'rgb'))]
    dates.sort()
    rgb = load_folder(os.path.join(path, 'rgb'))
    chm = rasterio.open(os.path.join(path, 'tree_map', 'CHM2020.tif')).read(1)
    forest_mask = (chm > 250).astype(bool)
    slope_map = calculate_slope_with_dates(rgb[:, 0], dates, len(rgb[:, 0]) / 2, len(rgb[:, 0])) / 100
    weights = (1 - abs(slope_map.ravel())).clip(0, 1)

    path_features = os.path.join(path, 'features')
    r_APO = rasterio.open(os.path.join(path_features, 'r_APO.tif')).read()
    amplitude_map_r, phase_map_r, offset_map_r = r_APO[0], r_APO[1], r_APO[2]
    crswir_APO = rasterio.open(os.path.join(path_features, 'crswir_APO.tif')).read()
    amplitude_map_crswir, phase_map_crswir, offset_map_crswir = crswir_APO[0], crswir_APO[1], crswir_APO[2]
    rcc_APO = rasterio.open(os.path.join(path_features, 'rcc_APO.tif')).read()
    amplitude_map_rcc, phase_map_rcc, offset_map_rcc = rcc_APO[0], rcc_APO[1], rcc_APO[2]
    dem = rasterio.open(os.path.join(path_features, 'elevation_aspect.tif')).read()
    elevation, aspect = dem[0], dem[1]

    features = {
        'amplitude_red': amplitude_map_r.ravel(),
        'phase_red': phase_map_r.ravel(),
        'offset_red': offset_map_r.ravel(),
        'amplitude_crswir': amplitude_map_crswir.ravel(),
        'phase_crswir': phase_map_crswir.ravel(),
        'offset_crswir': offset_map_crswir.ravel(),
        'amplitude_rcc': amplitude_map_rcc.ravel(),
        'phase_rcc': phase_map_rcc.ravel(),
        'offset_rcc': offset_map_rcc.ravel(),
        'elevation': elevation.ravel(),
        'aspect': aspect.ravel(),
        'tile_id': np.array([tile_id] * aspect.size)  # Add tile_id to the features
    }

    path_reference = os.path.join(path, 'reference_species')
    tif = [x for x in os.listdir(path_reference) if x.endswith('.tif')]
    reference = rasterio.open(os.path.join(path_reference, tif[0])).read()
    genus = reference[1]
    phen = reference[2]  # Assuming phenology data is stored in the third band
    valid_mask = (forest_mask & (phen != 0)).astype(bool)

    filtered_features = {k: v[valid_mask.ravel()] for k, v in features.items()}
    filtered_weights = weights[valid_mask.ravel()]
    filtered_genus = genus[valid_mask]
    filtered_phen = phen[valid_mask]

    filtered_features['genus'] = filtered_genus
    filtered_features['phen'] = filtered_phen

    df = pd.DataFrame(filtered_features)
    df = df.dropna()

    return df, filtered_weights[df.index]

def load_data(directory: str) -> pd.DataFrame:
    all_data = []
    all_weights = []
    tile_to_greco = {}

    for folder in tqdm(os.listdir(directory)):
        path = os.path.join(directory, folder)
        if folder.__contains__('.DS_Store') or folder.__contains__('.txt'):
            continue
        try:
            tile_df, tile_weight = load_data_from_tile(path)
            tile_id = os.path.basename(path).split('_')[1]
            greco_region = "_".join(os.path.basename(path).split('_')[4:-1])
            tile_to_greco[tile_id] = greco_region
            tile_df['tile_id'] = tile_id
            all_data.append(tile_df)
            all_weights.append(tile_weight)
        except Exception as e:
            print(f"Error processing {folder}: {e}")
            continue

    print(f"Loaded {len(all_data)} tiles")
    data_df = pd.concat(all_data, ignore_index=True)
    weights_array = np.concatenate(all_weights)

    return data_df, weights_array, tile_to_greco

#greco 
mapping_real_greco = {'Côtes_et_plateaux_de_la_Manche': 'Centre Nord semi-océanique',
                      'Côtes_et_plateaux_de_la_Manche': 'Centre Nord semi-océanique',
                      'Ardenne_primaire': 'Grand Est semi-continental',
                      'Préalpes_du_Nord': 'Alpes',
                      'Préalpes_du_Nord': 'Alpes',
                      'Garrigues' : 'Méditerranée',
                      'Massif_vosgien_central': 'Vosges',
                        'Premier_plateau_du_Jura': 'Jura',
                        'Piémont_pyrénéen' : 'Pyrénées',
                        'Terres_rouges': 'Sud-Ouest océanique' ,
                          'Corse_occidentale': 'Corse',
                        "Châtaigneraie_du_Centre_et_de_l'Ouest": 'Massif central' ,
                        'Ouest-Bretagne_et_Nord-Cotentin': 'Grand Ouest cristallin et océanique', 
                        'Total': 'Total'}


# metrics
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Compute accuracy, kappa, and confusion matrix components."""
    accuracy = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    scores = confusion_matrix(y_true, y_pred, labels=[1, 2]).ravel()
    tn, fp, fn, tp = scores / np.sum(scores)

    return {
        'Accuracy': accuracy,
        'Kappa': kappa,
        'True Positives': tp,
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn
    }