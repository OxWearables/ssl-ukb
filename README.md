# Self-Supervised Learning on the UK Biobank accelerometer dataset

This repository deploys the [Yuan et al. (2022)](https://arxiv.org/abs/2206.02909) Self-Supervised Learning (SSL) model on the UK Biobank accelerometer dataset ([UKB resource](https://biobank.ctsu.ox.ac.uk/crystal/label.cgi?id=1008)).

## Overview
The scripts in this repo perform an end-to-end activity prediction and summary statistics calculation on the UKB accelerometer data. 
The following steps are performed:

1) Model training (fine-tuning) and evaluation: The pretrained deep-learning SSL model is fine-tuned on a labelled dataset (Capture-24). The pretrained model is downloaded at runtime from [ssl-wearables](https://github.com/OxWearables/ssl-wearables). Performance is compared to a benchmark Random Forest (RF) model.
2) Prediction smoothing: The predicted time-series are smoothed with a Hidden Markov Model (HMM). For the SSL model, the HMM emission matrix is derived from the predicted probabilities of a held-out validation set. For the RF, the out-of-bag training probabilities are used.
3) UKB deployment: Activity inference is done on the UKB accelerometer time-series with the fine-tuned SSL model and smoothed with the HMM. Summary statistics are calculated from the predicted activity time-series. 

Model training and evaluation (steps 1 and 2) can be performed on a local machine (ideally with a CUDA-enabled GPU for the SSL model fine-tuning). 
The scripts for step 3 are set up for deployment on an HPC cluster in an array job.

[Capture-24](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001) is used as the labelled dataset for model training and evaluation.

Processed outputs from this deployment are available on BMRC, see [Authorship and usage](#authorship-and-usage).

## Installation
### Prepare virtual environment
Requires python >=3.7 <=3.9
```bash
git clone https://github.com/OxWearables/ssl-ukb.git
cd ssl-ukb
python -m venv venv
source venv/bin/activate
# conda alternative to venv: 'conda create -n ssl-ukb && conda activate ssl-ukb'
pip install -r requirements.txt
```

### Prepare Capture-24 data
Download the Capture-24 dataset zip [here](https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001). Extract the zip to a folder on your machine, e.g.: `/data/capture24`. This folder should contain the `.csv.gz` files. 

Open `utils/make_capture24.py` and change the `DATA_DIR` and `OUT_DIR` constants to reflect the location of the Capture-24 folder you just created and a chosen output location, respectively. Optionally adjust `NUM_WORKERS` to the number of CPU cores in your system. Now run the script to process the dataset:

```bash
python utils/make_capture24.py
```

This will save the dataset in Numpy format in the `OUT_DIR` folder (takes ~5 min with 8 workers).

### Configuration
Configuration parameters are located in `conf/config.yaml`. Below are the important settings that **need** to be changed based on your system. The other settings can stay as default.

| Config key     | Type      | Purpose                                                                                                                                       |
|----------------|-----------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| gpu            | int       | GPU id to use as master. Set to -1 for CPU processing.                                                                                        |
| multi_gpu      | bool      | Enable multi GPU model training.                                                                                                              |
| gpu_ids        | int array | List of GPU ids to use when multi_gpu: true                                                                                                   |
| num_workers    | int       | Number of CPU cores for parallel tasks. Best not to set this too high due to diminishing returns, 6-12 is a good value.                       |
| ssl_repo_path  | str       | Location of the ssl-wearables repo (for offline usage only, see below).                                                                       |
| data.data_root | str       | Location of the processed Capture-24 dataset in Numpy format. Should be the same value as `OUT_DIR` in `utils/make_capture24.py` (see above). |

### Note for offline usage
When internet connectivity is available, the scripts in this repo will download the pretrained self-supervised pytorch model from GitHub.

When working on a system without internet connectivity (e.g.: an HPC cluster worker), the [ssl-wearables](https://github.com/OxWearables/ssl-wearables) repository needs to be manually downloaded, and the `ssl_repo_path` config entry needs to point to the location of this repo:

```bash
# change /home/user/ to a folder of choice
cd /home/user/
git clone https://github.com/OxWearables/ssl-wearables.git
# Cloning into 'ssl-wearables'...
```
Change the config:
```yaml
ssl_repo_path: /home/user/ssl-wearables
```

## Usage
### Step 1: Training and evaluation
After installation and setting the relevant config entries for your system (see above), training the models is done with:

```bash
python train.py
```

With the default config this will train the RF and SSLNet and their respective HMM models, and save them in the `weights` folder. RF training (and evaluation) can be optionally disabled by setting the `rf.enabled` config flag to `false`.

SSLNet training will take ~30 min on a single Tesla V100. RF training will take a few minutes depending on the number of workers.

The models can then be evaluated on the independent test set by calling:

```bash
python eval.py
```

This will evaluate the trained RF and SSLNet on their original and HMM-smoothed test predictions. It will output the following:
- A report in CSV format for each model with per-subject mean classification performance in the `outputs` folder.
- Per-subject ground truth and predicted activity time series figures in the `plots` folder.
- Confusion matrix figures in the `plots` folder.


### Step 2: HPC cluster deployment
The `inference.py` file is intended for cluster deployment in an array job. It takes the trained SSL (using the weights obtained in Step 1) and applies it to an accelerometer file to get the predicted time series. 

After training the model and obtaining the weights in Step 1, use it as follows:

```bash
python inference.py /data/ukb-accelerometer/group1/4027057_90001_0_0.cwa.gz
```

See the top of the script for further comments. Note: to read the parquet data format used for the output file you can use `pandas.read_parquet`

The script also loads its config from `conf/config.yaml` (e.g.: the SSL and HMM weight paths), so when deploying on BMRC make sure this is set the same as in your development environment.

The typical workflow for BMRC deployment is as follows:

1) Do the installation and run Step 1 on your local machine or development system (only running`train.py` is strictly required to get the trained model)
2) After training, copy the whole `ssl-ukb` folder from your local machine to your work folder on BMRC (it should contain the trained weights in the `weights` folder)

The following steps will take place entirely on BMRC:

3) Install the virtual environment (or conda environment)
4) Clone the [ssl-wearables](https://github.com/OxWearables/ssl-wearables) repo to your work folder on BMRC, and set the config entry `ssl_repo_path` to this location
5) Make sure the config entry `gpu` is set to `-1` for CPU processing
6) Manually test `inference.py` on a single accelerometer file to see if it works and produces the expected output in the `outputs` folder. Do this in an interactive session, NOT on the login node, to check that the ssl-wearables repo is correctly loaded from disk without internet access.
7) Prepare an array job as usual, using a text file that contains line by line calls to `inference.py` for all accelerometer files you want to process. Number of cores and memory can be left as default (1 core, 15GB RAM). Using more than 1 core is not beneficial. Make sure the virtual environment (or conda) is activated at the start of the job.
8) Submit the array job. Note: if you're using conda, make sure no environment is active when you submit the job (not even the base env!). It will break things.

When the job is done, the output will be in the `outputs` folder (set by the config entry `ukb_output_path`). The output will be a Pandas dataframe in `{eid}.parquet` format with the predicted time series. 

### Post-processing

After obtaining the predicted time series, `summary.py` can be used to calculate summary statistics (same format as output by BBAA). It works on the `.parquet` file output by `inference.py`. Use it in the same way as `inference.py` on a single output file:

```bash
python summary.py /home/ssl-ukb/outputs/group1/4027057.parquet
```

You can do a second array job to run this on all your output files, or you can add it to the first array job so that it performs both `inference.py` and `summary.py` in sequence.

The output will be a `{eid}_summary.csv` file with the summary statistics.

Once you have obtained all the summary files,  `utils/merge_summary.py` can be used to collate the individual summary files into 1 large `summary.csv` file. Best to also run this on BMRC in an interactive session with 4-8 cores (set `num_workers` at the bottom of the script accordingly).

## Authorship and usage

This repo was authored by Gert Mertes `gertmertes [at] gmail {dot} com`

Processed outputs from the SSL+HMM trained on Capture-24, deployed on the UKB accelerometer dataset, are available in the BMRC group folder at `/well/doherty/projects/ukb/ssl-ukb/run2-enmofix` (predictions and summary stats). See the readme file in the parent directory for more info. 

**If you use these outputs, this repo, or parts of it, in your research, please include Gert Mertes as a co-author on papers that result from it.**

This repo also contains code snippets authored by other researchers of the OxWearables group, including Shing Chan, Hang Yuan and Aidan Acquah.

## License
This software is intended for use by academics carrying out research and not for commercial business use, see [LICENSE](LICENSE.md). If you are interested in using this software commercially,
please contact Oxford University Innovation Limited to negotiate a licence. Contact details are enquiries@innovation.ox.ac.uk