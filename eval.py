"""
Evaluate the following models: RF, SSL, RF+HMM, SSL+HMM
Requires the pretrained RF, SSL and HMMs (trained with train.py)

Output:
- A report in .csv for each model with per-subject classification performance.
- Per-subject time series plots for RF+HMM and SSL+HMM in the 'plots' folder
- Confusion matrix plots in the 'plots' folder
"""

import joblib
import torch
import numpy as np
import os
import plotly.graph_objects as go
import pandas as pd
from glob import glob
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm
from joblib import Parallel, delayed
from imblearn.ensemble import BalancedRandomForestClassifier
from models.hmm_learn import HMMLearn

# own module imports
import utils.utils as utils
from models.peak_count import PeakCounter
import models.sslnet as ssl
import models.rf as rf
from models.hmm import HMM
from utils.data import NormalDataset, load_data

log = utils.get_logger()


def score(x, y_true, y_pred, steps, groups, current_pid, t, interval,
          hmm, hmm_learn, peak_counter, hmm_peak_counter, hmm_learn_peak_counter):
    subject_mask = groups == current_pid
    
    subject_x = x[subject_mask]
    subject_true = y_true[subject_mask]
    subject_pred = y_pred[subject_mask]
    subject_t = t[subject_mask]
    steps = steps[subject_mask]

    # Smooth the label predictions using HMM smoothing
    subject_pred_hmm = hmm.predict(subject_pred, subject_t, interval)
    subject_pred_hmm_learn = hmm_learn.predict(subject_pred, subject_t, interval)

    # Remove unlabelled training data identified with -1
    labelled_mask = subject_true != -1

    subject_true = subject_true[labelled_mask]
    subject_pred = subject_pred[labelled_mask]
    subject_x = subject_x[labelled_mask]
    subject_t = subject_t[labelled_mask]
    subject_pred_hmm = subject_pred_hmm[labelled_mask]
    subject_pred_hmm_learn = subject_pred_hmm_learn[labelled_mask]

    # Transform true test labels using label encoder
    subject_true = utils.le.transform(subject_true)

    # Calculate performance of walking frame detection
    result = utils.classification_scores(subject_true, subject_pred)
    result_hmm = utils.classification_scores(subject_true, subject_pred_hmm)
    result_hmm_learn = utils.classification_scores(subject_true, subject_pred_hmm_learn)
    
    # Count the number of peaks for the predicted walking windows as step count
    step_counts = {'pid': current_pid,
                   'step_tot_true': steps.sum(),
                   'step_tot_pred': peak_counter.predict(subject_x, subject_pred),
                   'step_tot_pred_hmm': hmm_peak_counter.predict(subject_x, subject_pred_hmm),
                   'step_tot_pred_hmm_learn': hmm_learn_peak_counter.predict(subject_x, subject_pred_hmm_learn)
                  }

    return result, result_hmm, result_hmm_learn, step_counts, current_pid


def evaluate_model(training_data, cfg, fold="0"):
    GPU = cfg.gpu
    if GPU != -1:
        my_device = "cuda:" + str(GPU)
    else:
        my_device = "cpu"
    
    # load raw data
    (
        _, _, _, _, _,
        _, _, _, _, _,
        x_test, y_test, group_test, time_test, steps_test,
    ) = training_data

    # save performance scores and plots for every single subject
    my_pids = np.unique(group_test)

    if cfg.sslnet.enabled:
        # load pretrained SSL model
        sslnet = ssl.get_sslnet(my_device, cfg.ssl_repo_path, cfg.sslnet.weights.format(fold))

        hmm_ssl = HMM(utils.classes, uniform_prior=cfg.hmm.uniform_prior)
        hmm_ssl.load(cfg.hmm.weights_ssl.format(fold))

        hmm_learn_ssl = HMMLearn(utils.classes, uniform_prior=cfg.hmm.uniform_prior)
        hmm_learn_ssl.load(cfg.hmm_learn.weights_ssl.format(fold))

        # Load peak counters used to count steps
        peak_counter = PeakCounter(cfg.data.winsec, cfg.data.sample_rate)
        peak_counter.load(cfg.peak_counter.weights_ssl.format('ssl', fold))
        peak_counter_hmm = PeakCounter(cfg.data.winsec, cfg.data.sample_rate)
        peak_counter_hmm.load(cfg.peak_counter.weights_ssl.format('ssl_hmm', fold))
        peak_counter_hmm_learn = PeakCounter(cfg.data.winsec, cfg.data.sample_rate)
        peak_counter_hmm_learn.load(cfg.peak_counter.weights_ssl.format('ssl_hmm_learn', fold))
 
        # SSL data loader
        test_dataset = NormalDataset(x_test, y_test, pid=group_test, name="test", is_labelled=True)

        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.sslnet.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # get test predictions
        log.info('Get SSL test predictions')
        y_test, y_test_pred, _ = ssl.predict(
            sslnet, test_loader, my_device, output_logits=False
        )

        log.info('Process results')
        # Use joblib lazy parallel cause plotting is slow
        # SSL results
        results, results_hmm, result_hmm_learn, step_counts, subjects = \
            zip(*Parallel(n_jobs=cfg.num_workers)(
                delayed(score)(x_test, y_test, y_test_pred, steps_test, group_test, current_pid,  
                               time_test, cfg.data.winsec, hmm_ssl, hmm_learn_ssl, peak_counter,
                               peak_counter_hmm, peak_counter_hmm_learn)
                for current_pid in tqdm(my_pids)
            ))

        results = np.array(results)
        results_hmm = np.array(results_hmm)
        result_hmm_learn = np.array(result_hmm_learn)

        step_df = pd.DataFrame(step_counts)
        step_df.to_csv('{}/steps_ssl_{}.csv'.format(cfg.output_path, fold), index=False)
        
        # save SSL reports
        dfr = utils.classification_report(results, subjects, 
                                      os.path.join(cfg.output_path, 'report_ssl_{}.csv'.format(fold)))
        dfr_hmm = utils.classification_report(results_hmm, subjects, 
                                          os.path.join(cfg.output_path, 'report_ssl_hmm_{}.csv'.format(fold)))
        dfr_hmm_learn = utils.classification_report(results_hmm, subjects, 
                                                os.path.join(cfg.output_path, 'report_ssl_hmm_learn_{}.csv'.format(fold)))

        log.info('Results SSL:\n%s', dfr.mean())
        log.info('Results SSL-HMM:\n%s\n', dfr_hmm.mean())
        log.info('Results SSL-HMM-Learn:\n%s\n', dfr_hmm_learn.mean())
        
    if cfg.rf.enabled:
        # load pretrained RF
        rfmodel: BalancedRandomForestClassifier = joblib.load(cfg.rf.path.format(fold))
        log.info('Loaded RF from %s', cfg.rf.path.format(fold))
        
        hmm_rf = HMM(utils.classes, uniform_prior=cfg.hmm.uniform_prior)
        hmm_rf.load(cfg.hmm.weights_rf.format(fold))
        
        hmm_learn_rf = HMMLearn(utils.classes, uniform_prior=cfg.hmm.uniform_prior)
        hmm_learn_rf.load(cfg.hmm_learn.weights_rf.format(fold))

        # Load peak counters used to count steps
        peak_counter = PeakCounter(cfg.data.winsec, cfg.data.sample_rate)
        peak_counter.load(cfg.peak_counter.weights_rf.format('rf', fold))
        peak_counter_hmm = PeakCounter(cfg.data.winsec, cfg.data.sample_rate)
        peak_counter_hmm.load(cfg.peak_counter.weights_rf.format('rf_hmm', fold))
        peak_counter_hmm_learn = PeakCounter(cfg.data.winsec, cfg.data.sample_rate)
        peak_counter_hmm_learn.load(cfg.peak_counter.weights_rf.format('rf_hmm_learn', fold))

        log.info('Extract RF features')
        x_feats = rf.extract_features(x_test, sample_rate=cfg.data.sample_rate, num_workers=cfg.num_workers)

        log.info('Get RF test predictions')
        y_test_pred_rf = np.array(rfmodel.predict(x_feats), dtype='i')

        # RF results
        results_rf, results_hmm_rf, results_hmm_learn_rf, step_counts, subjects = \
            zip(*Parallel(n_jobs=cfg.num_workers)(
                delayed(score)(x_test, y_test, y_test_pred_rf, steps_test, group_test, current_pid,  
                               time_test, cfg.data.winsec, hmm_rf, hmm_learn_rf,  peak_counter,
                               peak_counter_hmm, peak_counter_hmm_learn)
                for current_pid in tqdm(my_pids)
            ))

        results_rf = np.array(results_rf)
        results_hmm_rf = np.array(results_hmm_rf)
        results_hmm_learn_rf = np.array(results_hmm_learn_rf)

        step_df = pd.DataFrame(step_counts)
        step_df.to_csv('{}/steps_rf_{}.csv'.format(cfg.output_path, fold), index=False)

        # save RF reports
        dfr_rf = utils.classification_report(results_rf, subjects, os.path.join(cfg.output_path, 
                                                                                'report_rf_{}.csv'.format(fold)))
        dfr_hmm_rf = utils.classification_report(results_hmm_rf, subjects, os.path.join(cfg.output_path, 
                                                                                        'report_rf_hmm_{}.csv'.format(fold)))
        dfr_hmm_learn_rf = utils.classification_report(results_hmm_learn_rf, subjects, os.path.join(cfg.output_path, 
                                                                                                    'report_rf_hmm_learn_{}.csv'.format(fold)))

        log.info('Results RF:\n%s', dfr_rf.mean())
        log.info('Results RF-HMM:\n%s', dfr_hmm_rf.mean())
        log.info('Results RF-HMM-Learn:\n%s', dfr_hmm_learn_rf.mean())


def evaluate_folds(cfg, scores = ['f1', 'kappa', 'accuracy'], folds=None):
    folds = folds or 1
    summary_folder = cfg.output_path + '/Summary'
    Path(summary_folder).mkdir(parents=True, exist_ok=True)

    models = {}
    if cfg.rf.enabled:
        models.update({'rf': 'Random Forest (RF)', 
                       'rf_hmm': 'RF + Hidden Markov Model', 
                       'rf_hmm_learn': 'RF + Unsupervised Hidden Markov Model'})

    if cfg.sslnet.enabled:
        models.update({'ssl': 'Self supervised ResNet 18 (SSL)', 
                       'ssl_hmm': 'SSL + Hidden Markov Model', 
                       'ssl_hmm_learn': 'SSL + Unsupervised Hidden Markov Model'})
    
    base_models = [model for model in models.keys() if 'hmm' not in model]

    master_report = pd.concat([
        pd.concat([pd.read_csv('{}/report_{}_{}.csv'.format(cfg.output_path, model, fold), 
                               index_col=[0]) 
                   for fold in range(folds)]).add_suffix('_'+model)
        for model in models.keys()], axis=1)
    
    master_report.index.name = 'Participant'

    master_report = master_report.groupby('Participant').mean()

    with open(summary_folder+"/config.txt", "w") as f:
        f.write(str({
            'Data Source(s)': cfg.data.sources,
            'Sample Rate': "{}Hz".format(cfg.data.sample_rate),
            'Window Size': "{}s".format(cfg.data.winsec),
            'Step Walking Threshold': "{} step(s) per window".format(cfg.data.step_threshold),
            'SSLNet Settings': cfg.sslnet,
            'RF Settings': cfg.rf
        }))

    master_report.to_csv(summary_folder+'/master_report.csv')

    summary_scores = pd.DataFrame([{
        score: '{:.3f} [\u00B1{:.3f}]'.format(master_report['{}_{}'.format(score, model)].mean(), 
                                              master_report['{}_{}'.format(score, model)].std())
            for score in scores} 
        for model in models.keys()], index=models.keys())

    summary_scores.index.name='Model'
    summary_scores.to_csv(summary_folder+'/summary_scores.csv')

    fig = go.Figure(data=[go.Table(
                    columnwidth=[100, 40, 40, 40],
                    header=dict(values=['Model'] + ["{} score".format(score.capitalize()) for score in scores],
                                font_size=16,
                                height=30),
                    cells=dict(values=[list(models.values())] +
                               [summary_scores[score] for score in scores],
                               font_size=16,
                               height=30))
                    ])
    fig.write_image(summary_folder+'/modelPerformance.png', width=1200, height=400)
   
    steps_report = pd.concat([
        pd.concat([pd.read_csv('{}/steps_{}_{}.csv'.format('outputs', model, fold), 
                               index_col=[0]) 
                    for fold in range(folds)]).add_suffix('_'+model)
                        for model in base_models], axis=1)
    
    steps_report.index.name = 'Participant'
    steps_report['step_tot_true'] = steps_report['step_tot_true_'+base_models[0]]
    steps_report.drop(columns=['step_tot_true_'+model for model in base_models], inplace=True)

    steps_report = steps_report.groupby('Participant').mean()

    steps_report.to_csv(summary_folder+'/steps_report.csv')

    steps_report.dropna(inplace=True)

    if len(steps_report) > 0:
        model_cols = [col for col in steps_report if col != 'step_tot_true']

        steps_summary = pd.DataFrame([{
            'Mean Absolute Error': int(mean_absolute_error(steps_report['step_tot_true'], steps_report[model_col])),
            'Mean Absolute Percent Error [%]': "{:.2f}".format(100*mean_absolute_percentage_error(steps_report['step_tot_true'], steps_report[model_col])),
            'Root mean square error': int(mean_squared_error(steps_report['step_tot_true'], steps_report[model_col], squared=False)),
            'Root mean square percent error [%]': "{:.2f}".format(
                100*mean_squared_error(steps_report['step_tot_true'], steps_report[model_col], squared=False)/steps_report['step_tot_true'].mean()),
            'Bias [%]': "{:.2f}".format(
                100*((steps_report[model_col].sum()-steps_report['step_tot_true'].sum())/steps_report['step_tot_true'].sum()))
        } for model_col in model_cols], index=models.keys())

        steps_summary.index.name='Model'
        steps_summary.to_csv(summary_folder+'/steps_summary.csv')

        fig = go.Figure(data=[go.Table(
                            header=dict(values=['Model']+
                                               list(steps_summary.columns),
                                        font_size=16,
                                        height=30),
                            cells=dict(values=[list(models.values())] +
                                              [steps_summary.iloc[:, i] 
                                                for i in range(len(steps_summary.columns))],
                                       font_size=16,
                                       height=30))
                         ])
        fig.write_image(summary_folder+'/steps_summary.png', width=1200, height=800)

    data_sources = [cfg.training.external_val] if cfg.training.external_val is not None else cfg.data.sources.keys()
    if len(data_sources) > 1:
        def str_lookup(string, reference):
            for elem in reference:
                if elem in string:
                    return elem
            return ""

        master_report['source'] = [str_lookup(pid, data_sources) for pid in master_report.index]

        summary_scores_stratified = pd.DataFrame([
            {"{}_{}".format(score, source): 
                "{:.3f} [\u00B1{:.3f}]".format(master_report.loc[master_report["source"]==source, 
                                                                 "{}_{}".format(score, model)].mean(), 
                                               master_report.loc[master_report["source"]==source, 
                                                                 "{}_{}".format(score, model)].std()) 
                for score in scores
                    for source in data_sources}
            for model in models])

        summary_scores_stratified.to_csv(summary_folder+'/summary_scores_stratified.csv')
    
        fig2 = go.Figure(data=[go.Table(
                            header=dict(values=['Model']+
                                               ["{} study {} score".format(source, score.capitalize()) 
                                                    for score in scores for source in data_sources],
                                        font_size=16,
                                        height=30),
                            cells=dict(values=[list(models.values())] +
                                              [summary_scores_stratified["{}_{}".format(score, source)] 
                                                for score in scores for source in data_sources],
                                       font_size=16,
                                       height=30))
                         ])
        fig2.write_image(summary_folder+'/summary_scores_stratified.png', width=1200, height=800)

    # Remove all csv files in the base output path    
    for f in glob(cfg.output_path+"/*.csv"):
        os.remove(f)

if __name__ == '__main__':
    cfg = OmegaConf.load("conf/config.yaml")

    np.random.seed(42)
    torch.manual_seed(42)
    log.info(str(OmegaConf.to_yaml(cfg)))

    #evaluate_model(load_data(cfg)[0], cfg)
    evaluate_folds(cfg, folds=cfg.training.num_folds)
