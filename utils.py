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

def postprocess_qa_vegetation(qa : np.array) -> np.array:
    """Postprocess quality assessment mask to yield a vegetation mask.

    Args:
        qa (np.array): Quality assessment mask. (Sentinel-2 SCL band)

    Returns:
        np.array: Vegetation mask.
    """
    mask = np.zeros_like(qa)
    mask[ qa == 4] = 1 
    return mask

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
    # first_day_offset = 2 * np.pi * (dates[0].day - 1) / 30
    
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
def load_data_from_tile(path: str, config: str) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load data from a single tile directory and process it to extract features and target variables.

    Parameters:
    path (str): Path to the tile directory.
    config (str): Configuration string indicating the specific processing configuration.

    Returns:
    tuple[pd.DataFrame, np.ndarray]: A tuple containing a DataFrame of the processed features and an array of weights.
    """
    tile_id = os.path.basename(path).split('_')[1]
    dates = [datetime.strptime(filename.split('_')[0], '%Y-%m-%d') for filename in os.listdir(os.path.join(path, 'rgb'))]
    dates.sort()
    rgb = load_folder(os.path.join(path, 'rgb'))
    vegetation_mask = load_folder(os.path.join(path, 'qa'), postprocess_qa_vegetation).mean(axis=0)
    vegetation_mask = (vegetation_mask > 0.25).astype(bool)

    chm = rasterio.open(os.path.join(path, 'tree_map', 'CHM2020.tif')).read(1)
    forest_mask = (chm > 250).astype(bool)
    slope_map = calculate_slope_with_dates(rgb[:, 0], dates, len(rgb[:, 0]) / 2, len(rgb[:, 0])) / 100
    weights = (1 - abs(slope_map.ravel())).clip(0, 1)

    path_features = os.path.join(path, 'features')
    r_APO = rasterio.open(os.path.join(path_features, f'APO_R_{config}.tif')).read()
    amplitude_map_r, phase_map_r, offset_map_r = r_APO[0], r_APO[1], r_APO[2]
    g_APO = rasterio.open(os.path.join(path_features, f'APO_G_{config}.tif')).read()
    amplitude_map_g, phase_map_g, offset_map_g = g_APO[0], g_APO[1], g_APO[2]
    b_APO = rasterio.open(os.path.join(path_features, f'APO_B_{config}.tif')).read()
    amplitude_map_b, phase_map_b, offset_map_b = b_APO[0], b_APO[1], b_APO[2]
    crswir_APO = rasterio.open(os.path.join(path_features, f'APO_CRSWIR_{config}.tif')).read()
    amplitude_map_crswir, phase_map_crswir, offset_map_crswir = crswir_APO[0], crswir_APO[1], crswir_APO[2]
    dem = rasterio.open(os.path.join(path_features, 'elevation_aspect.tif')).read()
    elevation, aspect = dem[0], dem[1]

    features = {
        'amplitude_red': amplitude_map_r.ravel(),
        'phase_red': phase_map_r.ravel(),
        'offset_red': offset_map_r.ravel(),
        'amplitude_green': amplitude_map_g.ravel(),
        'phase_green': phase_map_g.ravel(),
        'offset_green': offset_map_g.ravel(),
        'amplitude_blue': amplitude_map_b.ravel(),
        'phase_blue': phase_map_b.ravel(),
        'offset_blue': offset_map_b.ravel(),
        'amplitude_crswir': amplitude_map_crswir.ravel(),
        'phase_crswir': phase_map_crswir.ravel(), 
        'offset_crswir': offset_map_crswir.ravel(),
        'elevation': elevation.ravel(),
        'aspect': aspect.ravel(),
        'tile_id': np.array([tile_id] * aspect.size)  # Add tile_id to the features
    }

    path_reference = os.path.join(path, 'reference_species')
    tif = [x for x in os.listdir(path_reference) if x.endswith('.tif')]
    reference = rasterio.open(os.path.join(path_reference, tif[0])).read()
    genus = reference[1]
    phen = reference[2]  # Assuming phenology data is stored in the third band
    source = reference[4]
    valid_mask = (forest_mask & (phen != 0) & vegetation_mask).astype(bool)

    filtered_features = {k: v[valid_mask.ravel()] for k, v in features.items()}
    filtered_weights = weights[valid_mask.ravel()]
    filtered_genus = genus[valid_mask]
    filtered_phen = phen[valid_mask]
    filtered_source = source[valid_mask]

    filtered_features['genus'] = filtered_genus
    filtered_features['phen'] = filtered_phen
    filtered_features['source'] = filtered_source

    df = pd.DataFrame(filtered_features)
    df = df.dropna()

    return df, filtered_weights[df.index]

def load_data(directory: str, config: str) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """
    Load and process data from all tile directories within a specified directory.

    Parameters:
    directory (str): Path to the directory containing tile subdirectories.
    config (str): Configuration string indicating the specific processing configuration.

    Returns:
    tuple[pd.DataFrame, np.ndarray, dict]: A tuple containing a DataFrame of the combined processed features,
                                           an array of combined weights, and a dictionary mapping tile IDs to GRECO regions.
    """
    all_data = []
    all_weights = []
    tile_to_greco = {}

    for folder in tqdm(os.listdir(directory)):
        path = os.path.join(directory, folder)
        if folder.__contains__('.DS_Store') or folder.__contains__('.txt'):
            continue
        try:
        # if True:
            tile_df, tile_weight = load_data_from_tile(path, config)
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

def load_and_preprocess_table_data(config: str, remove_source: list = ['DSF', 'FrenchNFI'], verbose: int = 1) -> pd.DataFrame:
    """
    Load and preprocess data for a given configuration.

    Parameters:
    config (str): Configuration string indicating the specific processing configuration.
    remove_source (list): List of sources to be removed from the data. Default is ['DSF', 'FrenchNFI'].
    verbose (int): Verbosity level. Default is 1. If greater than 0, prints progress messages.

    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    if verbose > 0:
        print(f"Loading data for config {config}")

    data = pd.read_parquet(f'data/entire_dataset_{config}.parquet')

    mapping_source_reverse = {v: k for k, v in mapping_source.items()}
    remove_source = [mapping_source_reverse[source] for source in remove_source]

    if verbose > 0:
        print(f"Removing sources: {remove_source}")

    data = data[~data['source'].isin(remove_source)]

    if verbose > 0:
        print("Computing cos and sin of phase and aspect")

    data['cos_phase_red'] = np.cos(data['phase_red'])
    data['sin_phase_red'] = np.sin(data['phase_red'])
    data['cos_phase_green'] = np.cos(data['phase_green'])
    data['sin_phase_green'] = np.sin(data['phase_green'])
    data['cos_phase_blue'] = np.cos(data['phase_blue'])
    data['sin_phase_blue'] = np.sin(data['phase_blue'])
    data['cos_phase_crswir'] = np.cos(data['phase_crswir'])
    data['sin_phase_crswir'] = np.sin(data['phase_crswir'])
    data['cos_aspect'] = np.cos(np.radians(data['aspect']))
    data['sin_aspect'] = np.sin(np.radians(data['aspect']))

    # Remove inf or nan values
    if verbose > 0:
        print(f"Removing inf or nan values")
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if verbose > 0:
        print(f"Dataset shape: {data.shape}")

    return data

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

mapping_source = {
    1: 'DSF',
    2: 'calib_corse',
    3: 'calib_lozere',
    4: 'calib_meuse_vosges',
    5: 'RENECOFOR',
    6: 'FrenchNFI',
    7: 'PlantNet',
    8: 'PureForest'
}

greco_regions_fr_en = {
    "Grand Ouest cristallin et océanique": "Greater Crystalline and Oceanic West",
    "Centre Nord semi-océanique": "Semi-Oceanic North Center",
    "Grand Est semi-continental": "Greater Semi-Continental East",
    "Vosges": "Vosges",
    "Jura": "Jura",
    "Sud-Ouest océanique": "Oceanic Southwest",
    "Massif central": "Central Massif",
    "Alpes": "Alps",
    "Pyrénées": "Pyrenees",
    "Méditerranée": "Mediterranean",
    "Corse": "Corsica"
}

#evaluation
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_model(model: BaseEstimator, X: pd.DataFrame, y: pd.Series, skf: StratifiedKFold, regions: pd.Series) -> (pd.DataFrame, dict):
    """
    Evaluates a model using stratified k-fold cross-validation and returns the averaged metrics.
    Also calculates the metrics per region.

    Parameters:
    model (BaseEstimator): The machine learning model to be evaluated.
    X (pd.DataFrame): The feature matrix.
    y (pd.Series): The target vector.
    skf (StratifiedKFold): The stratified k-fold cross-validator.
    regions (pd.Series): The series indicating the region for each sample.

    Returns:
    pd.DataFrame: A DataFrame containing the averaged metrics across all folds.
    dict: A dictionary with regions as keys and DataFrames of metrics as values.
    """
    metrics_list = []
    region_metrics = {region: [] for region in regions.unique()}
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        regions_test = regions.iloc[test_index]
        y_pred = model.predict(X_test)
        
        metrics = {
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'accuracy': accuracy_score(y_test, y_pred)
        }
        metrics_list.append(metrics)
        
        for region in region_metrics.keys():
            region_mask = (regions_test == region)
            y_test_region = y_test[region_mask]
            y_pred_region = pd.Series(y_pred, index=y_test.index)[region_mask]
            if not y_test_region.empty:
                region_metrics_values = {
                    'precision': precision_score(y_test_region, y_pred_region, average='weighted', zero_division=0),
                    'recall': recall_score(y_test_region, y_pred_region, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_test_region, y_pred_region, average='weighted', zero_division=0),
                    'accuracy': accuracy_score(y_test_region, y_pred_region)
                }
                region_metrics[region].append(region_metrics_values)
    
    # Average the metrics over all the folds
    avg_metrics = {metric: np.mean([fold_metrics[metric] for fold_metrics in metrics_list]) for metric in metrics_list[0].keys()}
    
    avg_region_metrics = {}
    for region, region_metrics_list in region_metrics.items():
        avg_region_metrics[region] = {metric: np.mean([region_metrics_values[metric] for region_metrics_values in region_metrics_list]) for metric in region_metrics_list[0].keys()}
    
    return pd.DataFrame([avg_metrics]), {region: pd.DataFrame([metrics]) for region, metrics in avg_region_metrics.items()}


# Example usage:
# model = RandomForestClassifier(n_estimators=30)
# metrics_report = evaluate_model(model, X_scaled, y, skf)
# print(metrics_report)


# Function to save checkpoint
import joblib 
def save_checkpoint(name, search, checkpoint_dir):
    filename = os.path.join(checkpoint_dir, f"{name}_checkpoint.pkl")
    joblib.dump(search, filename)


# Function to load checkpoint
def load_checkpoint(name, checkpoint_dir, extra=''):
    filename = os.path.join(checkpoint_dir, f"{name}_{extra}_checkpoint.pkl")
    if os.path.exists(filename):
        return joblib.load(filename)
    return None

def crop(image: np.ndarray, factor: int) -> np.ndarray:
    crop_size = image.shape[1] // factor
    start = (image.shape[1] - crop_size) // 2
    if image.ndim == 2:
        return image[start:start+crop_size, start:start+crop_size]
    else:
        return image[:, start:start+crop_size, start:start+crop_size]
    

import rasterio
import os
import numpy as np
import multidem
from rasterio.warp import transform_bounds, reproject

def write_dem_features(path, item=-2):
    print(path)
    raster = rasterio.open(path)
    # src = raster.read().clip(0,10000)
    pr = raster.profile
    target_transform = raster.transform
    target_bounds = transform_bounds(raster.crs, {'init':'EPSG:4326'}, *raster.bounds)
    try:
        dem, transform, crs = multidem.crop(target_bounds, source="SRTM30", datum="orthometric")
        dem, transform = reproject(dem, np.zeros((1,*raster.shape)), src_transform=transform, src_crs=crs, dst_crs={'init':str(raster.crs)}, dst_transform=target_transform, dst_shape=raster.shape)
        out_path = os.path.join( "/".join(path.split('/')[:item]), 'dem.tif')
        pr['transform'] = target_transform
        pr['count'] = 1
        pr['dtype'] = 'float32'
        with rasterio.open(out_path, "w", **pr) as dest:
            dest.write(dem.astype('float32'))
        return 1
    except :
        return 0