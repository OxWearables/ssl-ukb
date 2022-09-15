# Self-Supervised Learning (SSL) on the UK Biobank accelerometer dataset

This repository deploys the [Yuan et al. (2022)](https://arxiv.org/abs/2206.02909) Self-Supervised Learning model ([Github](https://github.com/OxWearables/ssl-wearables)) on the UK Biobank accelerometer dataset ([paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0169649), [UKB resource](https://biobank.ctsu.ox.ac.uk/crystal/label.cgi?id=1008)).

## Overview
WIP

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
|                |           |                                                                                                                                               |

### Note for offline usage
When internet connectivity is available, the scripts in this repo will automatically download the pre-trained self-supervised pytorch model from Github.

When working on a system without internet connectivity (e.g.: a HPC cluster worker), the [ssl-wearables](https://github.com/OxWearables/ssl-wearables) repository needs to be manually downloaded, and the `ssl_repo_path` config entry needs to point to the location of this repo:

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
WIP

## License
This software is intended for use by academics carrying out research and not for commercial business use, see [LICENSE](LICENSE.md). If you are interested in using this software commercially,
please contact Oxford University Innovation Limited to negotiate a licence. Contact details are enquiries@innovation.ox.ac.uk