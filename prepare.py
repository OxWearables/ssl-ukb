"""
Prepare the training data
"""

import numpy as np
import os
import pandas as pd
from tqdm.auto import tqdm
from glob import glob
from pathlib import Path
from joblib import Parallel, delayed

from omegaconf import OmegaConf

# own module imports
import utils.utils as utils

log = utils.get_logger()

def read_csv(filename):
    """ Data loader """

    data = pd.read_csv(
        filename, 
        parse_dates=['timestamp'], 
        index_col='timestamp',
        dtype={
            'x': 'f4', 
            'y': 'f4', 
            'z': 'f4',
            'mag': 'f4',
            'annotation': 'i2',
            'task_code': str,
            'tremor': 'i2',
            'bradykinesia': 'i2',
            'dyskinesia': 'i2'
        }
    )
    return data

def resample(data: pd.DataFrame, sample_rate, annotation=None, dropna=False):
    """ 
    Nearest neighbor resampling. For downsampling, it is recommended to first
    apply an antialiasing filter.
    :param data: A pandas.DataFrame of acceleration time-series. The index must be a DateTimeIndex.
    :type data: pandas.DataFrame.
    :param sample_rate: Target sample rate (Hz) to achieve.
    :type sample_rate: int or float
    :param dropna: Whether to drop NaN values after resampling. Defaults to False.
    :type dropna: bool, optional
    :return: Processed data and processing info.
    :rtype: (pandas.DataFrame, dict)
    """

    # Round-up sample_rate if non-integer
    if isinstance(sample_rate, float) and not sample_rate.is_integer():
        print(f"Found non-integer sample_rate {sample_rate},", end=" ")
        sample_rate = np.ceil(sample_rate)
        print(f"rounded-up to {sample_rate}.")

    # Create a new index with intended sample_rate. Start and end times are
    # rounded to seconds so that the number of ticks (periods) is round
    start = data.index[0].ceil('S')
    end = data.index[-1].floor('S')
    periods = int((end - start).total_seconds() * sample_rate + 1)  # +1 for the last tick
    new_index = pd.date_range(start, end, periods=periods, name='time')
    data = data.iloc[~data.index.duplicated()] # Remove records with duplicated index   
    
    if annotation == 'steps':
        # Resampling acceleration values to take the nearest acceleration value in each axis
        data_resampled = data[['x', 'y', 'z']].reindex(new_index,
                                                       method='nearest',
                                                       tolerance=pd.Timedelta('1s'),
                                                       limit=1)

        # Resampling step annotations requires finding the nearest resampled time point in the new index to the annotated step time point
        data_resampled['annotation'] = 0
        nearest_steps = new_index.get_indexer(data[data['annotation']==1].index, method='nearest')
        data_resampled.loc[data_resampled.index[nearest_steps], 'annotation'] = 1

        if data_resampled['annotation'].sum() != data['annotation'].sum():
            print("Resampling has caused a change in step count")
    
    else:
        # Resampling all data points to nearest value
        data_resampled = data.reindex(new_index,
                                      method='nearest',
                                      tolerance=pd.Timedelta('1s'),
                                      limit=1)

    if dropna:
        data_resampled = data_resampled.dropna()

    return data_resampled

def prepare_data(cfg, n_workers=-1, overwrite=True):
    if overwrite or not os.path.exists(cfg.data.processed_data+"/config.txt"):
        log.info("Processing raw data.")
        files = [(source, elem) for source in cfg.data.name.split(",") 
                                for elem in glob("{}/{}/*.csv".format(cfg.data.raw_data, source))]

        X, Ys, T = zip(*Parallel(n_jobs=n_workers)(
            delayed(prepare_participant_data)(filename, cfg.data.sample_rate, cfg.data.winsec, 
                                              cfg.data.step_threshold, data_source) 
                                                   for data_source, filename in tqdm(files)))

        X = np.concatenate(X)
        T = np.concatenate(T)
        Ys = pd.concat(Ys)

        Path(cfg.data.processed_data).mkdir(parents=True, exist_ok=True)

        np.save(cfg.data.processed_data+"/X.npy", X)
        np.save(cfg.data.processed_data+"/T.npy", T)
    
        for col in Ys.columns:
            np.save("{}/Y_{}.npy".format(cfg.data.processed_data, col), np.array(Ys[col]))
        
        with open(cfg.data.processed_data+"/config.txt", "w") as f:
            f.write(str({
                'Data Source(s)': cfg.data.name,
                'Sample Rate': "{}Hz".format(cfg.data.sample_rate),
                'Window Size': "{}s".format(cfg.data.winsec),
                'Step Walking Threshold': "{} step(s) per window".format(cfg.data.step_threshold)
            }))

    else:
        with open(cfg.data.processed_data+"/config.txt", "rt") as f:
            config = f.read()
            log.info(config)
        log.info("Using already processed data.")


def prepare_participant_data(filename, sample_rate, winsec, step_threshold, source):
    data = read_csv(filename)
    if source == 'OxWalk':
        data = resample(data, sample_rate, 'steps')
    else:
        data = resample(data, sample_rate)
    X, Y, T = utils.make_windows(data, winsec, 
                                         sample_rate, step_threshold)
    #pid = re.search(r'(P\d{2})', os.path.basename(filename)).group(1).upper()  # P01, P02, ...
    pid = os.path.basename(filename).replace(".csv", "").replace("_wrist100", "") # PXX, PYY, XX_AAA, YY_BBB
    Y['pid'] = pid
    Y['source'] = source

    return X, Y, T

if __name__ == "__main__":
    np.random.seed(42)

    cfg = OmegaConf.load("conf/config.yaml")
    log.info(str(OmegaConf.to_yaml(cfg)))

    prepare_data(cfg, 10, cfg.data.overwrite)
