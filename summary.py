"""
Calculate summary statistics from a predicted time series.
Run inference.py first to generate the required input files (.parquet and _info.csv).

Arguments:
    A DataFrame in parquet format, named {eid}.parquet
    An actipy info dict should exist alongside the input file, named {eid}_info.csv

Example usage:
    python summary.py /data/ukb/outputs/group3/1001366.parquet

Output:
    An {eid}_summary.csv file in the same folder as the input .parquet file
"""

import os
import argparse
import pandas as pd
from pathlib import Path
import utils.utils as utils
from utils.summarisation import getActivitySummary

log = utils.get_logger()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='SSL UKB Summary stats',
                                     usage='Compute summary statistics from an input .parquet file')
    parser.add_argument('input_file', type=str, help='input file')
    args = parser.parse_args()

    input_file = Path(args.input_file)
    root = input_file.parent

    log.info('Working on %s', input_file)

    # load prediction dataframe and convert to time series format
    df_ukb = pd.read_parquet(input_file, engine='pyarrow')
    df = utils.ukb_df_to_series(df_ukb, 'label_hmm')

    # extract pid and group from filename
    pid = input_file.stem.split('_')[0]
    group = input_file.parent.stem

    # read device info file
    info_file = os.path.join(root, pid + '_info.csv')
    df_info = pd.read_csv(info_file)

    # prepare summary data dict, this will be immuted by getActivitySummary with the summary stats
    summary = {
        'eid': pd.Series(pid, dtype='string'),
        'file-name': pd.Series(df_info['Filename'][0], dtype='string')
    }

    log.info('Calculating summary stats')
    data, data_imputed, _ = getActivitySummary(df, df_info, summary, utils.labels_, imputation=True)

    # output is a dataframe with 1 row
    df_summary = pd.DataFrame(summary, index=[0])

    # write {eid}_summary csv in the same path as the {eid}.parquet file
    output_file = os.path.join(root, pid + '_summary.csv')
    df_summary.to_csv(output_file, index=False)

    log.info('Summary saved to %s', output_file)
