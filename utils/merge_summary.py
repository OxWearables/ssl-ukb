"""
Merge the individual {eid}_summary.csv files together in 1 aggregated summary.csv file
"""

import os
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from glob import glob

# path where to look for the output files (should contain the group folders)
# will save summary.csv here
root = '/home/azw524/work/ssl-ukb/outputs/run1/'

if __name__ == '__main__':
    files = glob(os.path.join(root, '**/*_summary.csv'))

    # hardcode some dtypes, the rest will be inferred (mostly floats)
    df = dd.read_csv(files, assume_missing=False, header=0,
                     dtype={'eid': 'int',
                            'file-firstDay(0=mon,6=sun)': 'int',
                            'quality-goodWearTime': 'int',
                            'wearTime-numNonWearEpisodes(>1hr)': 'int',
                            'errs-interrupts-num': 'int',
                            'quality-goodCalibration': 'int',
                            'totalReads': 'int',
                            'file-startTime': 'string',
                            'file-endTime': 'string',
                            })

    # Aggregate all files in 1 large dataframe, and save to csv.
    # Will take about 20 minutes for 100k files.
    # Don't set num_workers too high because these will all read from disk in parallel.
    # This needs quite a lot of memory (~15 gigs with 8 workers).
    with ProgressBar():
        (
            df.compute(scheduler='processes', num_workers=8)
            .to_csv(os.path.join(root, 'summary.csv'), index=False)
        )


