import os
import argparse
import pandas as pd
from pathlib import Path
import utils.utils as utils
from utils.summarisation import getActivitySummary
labels = ['light', 'moderate-vigorous', 'sedentary', 'sleep']

parser = argparse.ArgumentParser(prog='SSL UKB Summary stats',
                                 usage='Compute summary statistics from an input .parquet file')
parser.add_argument('input_file', type=str, help='input file')
args = parser.parse_args()

input_file = Path(args.input_file)
root = input_file.parent

df_ukb = pd.read_parquet(input_file, engine='pyarrow')

df = utils.ukb_df_to_series(df_ukb, 'label_hmm')
pid = input_file.stem.split('_')[0]
group = input_file.parent.stem

info_file = os.path.join(root, pid + '_info.csv')
df_info = pd.read_csv(info_file)

summary = {
    'eid': pd.Series(pid, dtype='string'),
    'file-name': pd.Series(df_info['Filename'][0], dtype='string')
}

data, data_imputed, _ = getActivitySummary(df, df_info, summary, labels, imputation=True)

df_summary = pd.DataFrame(summary, index=[0])

# write {eid}_summary csv in the same path as the {eid}.parquet file
output_file = os.path.join(root, pid + '_summary.csv')
df_summary.to_csv(output_file, index=False)
