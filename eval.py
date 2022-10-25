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
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pandas as pd
import sklearn.metrics as metrics
import seaborn as sns
from glob import glob

from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from accelerometer.accPlot import plotTimeSeries
from tqdm import tqdm
from joblib import Parallel, delayed
from imblearn.ensemble import BalancedRandomForestClassifier
from models.hmm_learn import HMMLearn

# own module imports
import utils.utils as utils
#from models.step_count import StepCounter
import models.sslnet as ssl
import models.rf as rf
from models.hmm import HMM
from utils.data import NormalDataset, load_data

log = utils.get_logger()

def score(name, current_pid, pid, x, y, y_pred, t, interval, hmm, hmm_learn, step_counter, classes, plotData=False):
    subject_filter = current_pid == pid
    subject_true = y[subject_filter]
    subject_pred = y_pred[subject_filter]

    #steps = step_counter.predict(x[subject_filter]).sum()
    steps=0

    subject_pred_hmm =  hmm.predict(subject_pred, t[subject_filter], interval)
    subject_pred_hmm_learn =  hmm_learn.predict(subject_pred, t[subject_filter], interval)

    result = utils.classification_scores(subject_true, subject_pred)
    result_hmm = utils.classification_scores(subject_true, subject_pred_hmm)
    result_hmm_learn = utils.classification_scores(subject_true, subject_pred_hmm_learn)

    cmatrix = metrics.confusion_matrix(subject_true, subject_pred, labels=utils.classes)
    cmatrix_hmm = metrics.confusion_matrix(subject_true, subject_pred_hmm, labels=utils.classes)
    cmatrix_hmm_learn = metrics.confusion_matrix(subject_true, subject_pred_hmm_learn, labels=utils.classes)

    # plot subject predictions
    df_true = utils.raw_to_df(x[subject_filter], subject_true, t[subject_filter], classes)
    df_pred = utils.raw_to_df(x[subject_filter], subject_pred, t[subject_filter], classes)
    df_pred_hmm = utils.raw_to_df(x[subject_filter], subject_pred_hmm, t[subject_filter], classes)
    df_pred_hmm_learn = utils.raw_to_df(x[subject_filter], subject_pred_hmm_learn, t[subject_filter], classes)

    if plotData:
        fig = plotTimeSeries(df_true)
        fig.savefig('plots/{pid}_true.png'.format(pid=current_pid), dpi=200)
        plt.close()

        fig = plotTimeSeries(df_pred)
        fig.savefig('plots/{pid}_{model}_pred.png'.format(pid=current_pid, model=name), dpi=200)
        plt.close()

        fig = plotTimeSeries(df_pred_hmm)
        fig.savefig('plots/{pid}_{model}_pred_hmm.png'.format(pid=current_pid, model=name), dpi=200)
        plt.close()

        fig = plotTimeSeries(df_pred_hmm_learn)
        fig.savefig('plots/{pid}_{model}_pred_hmm_learn.png'.format(pid=current_pid, model=name), dpi=200)
        plt.close()

    return result, result_hmm, result_hmm_learn, cmatrix, cmatrix_hmm, cmatrix_hmm_learn, steps, current_pid
    
def evaluate_model(training_data, cfg, fold="0"):
    GPU = cfg.gpu
    if GPU != -1:
        my_device = "cuda:" + str(GPU)
    else:
        my_device = "cpu"
    
    # load raw data
    (
        _, _, _, _,
        _, _, _, _,
        x_test, y_test, group_test, time_test,
    ) = training_data

    le = utils.le  # label encoder

    # save performance scores and plots for every single subject
    my_pids = np.unique(group_test)
    plots = {}
    #step_counter = StepCounter(window_sec=cfg.data.winsec,
    #                           sample_rate=cfg.data.sample_rate,
    #                           wd_params={'ssl_weights': cfg.sslnet.weights.format(fold),
    #                                      'sample_rate': cfg.data.sample_rate,
    #                                      'device': my_device,
    #                                      'hmm_path': cfg.hmm.weights_ssl.format(fold)})

    if cfg.sslnet.enabled:
        # load pretrained SSL model
        sslnet = ssl.get_sslnet(my_device, cfg.ssl_repo_path, cfg.sslnet.weights.format(fold))
        hmm_ssl = HMM(utils.classes, uniform_prior=cfg.hmm.uniform_prior)
        hmm_ssl.load(cfg.hmm.weights_ssl.format(fold))
        hmm_learn_ssl = HMMLearn(utils.classes, uniform_prior=cfg.hmm.uniform_prior)
        hmm_learn_ssl.load(cfg.hmm_learn.weights_ssl.format(fold))
 
        # SSL data loader
        test_dataset = NormalDataset(x_test, y_test, pid=group_test, name="test", is_labelled=True)

        test_loader = DataLoader(
            test_dataset,
            batch_size=50,
            shuffle=False,
            num_workers=0,
        )

        # get test predictions
        log.info('Get SSL test predictions')
        y_test, y_test_pred, pid_test = ssl.predict(
            sslnet, test_loader, my_device, output_logits=False
        )

        log.info('Process results')
        # Use joblib lazy parallel cause plotting is slow
        # SSL results
        results, results_hmm, result_hmm_learn, cmatrix, cmatrix_hmm, cmatrix_hmm_learn, step_tot, subjects = \
            zip(*Parallel(n_jobs=cfg.num_workers)(
                delayed(score)('SSL', current_pid, group_test, x_test, y_test, y_test_pred, 
                               time_test, cfg.data.winsec, hmm_ssl, hmm_learn_ssl, {}, le.classes_)
                for current_pid in tqdm(my_pids)
            ))

        results = np.array(results)
        results_hmm = np.array(results_hmm)
        result_hmm_learn = np.array(result_hmm_learn)

        cmatrix = pd.DataFrame(np.sum(cmatrix, axis=0), index=le.classes_, columns=le.classes_)
        cmatrix_hmm = pd.DataFrame(np.sum(cmatrix_hmm, axis=0), index=le.classes_, columns=le.classes_)
        cmatrix_hmm_learn = pd.DataFrame(np.sum(cmatrix_hmm_learn, axis=0), index=le.classes_, columns=le.classes_)

        step_df = pd.DataFrame({'pid': my_pids, 'tot_steps': step_tot})
        step_df.to_csv('{}/steps_{}.csv'.format(cfg.ukb_output_path, fold))

        plots = {**plots, **{
            'matrix_ssl': cmatrix,
            'matrix_ssl_hmm': cmatrix_hmm,
            'matrix_ssl_hmm_learn': cmatrix_hmm_learn
        }}
        
        # save SSL reports
        dfr = utils.classification_report(results, subjects, 
                                      os.path.join(cfg.ukb_output_path, 'report_ssl_{}.csv'.format(fold)))
        dfr_hmm = utils.classification_report(results_hmm, subjects, 
                                          os.path.join(cfg.ukb_output_path, 'report_ssl_hmm_{}.csv'.format(fold)))
        dfr_hmm_learn = utils.classification_report(results_hmm, subjects, 
                                                os.path.join(cfg.ukb_output_path, 'report_ssl_hmm_learn_{}.csv'.format(fold)))

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

        log.info('Extract RF features')
        x_feats = rf.extract_features(x_test, sample_rate=cfg.data.sample_rate, num_workers=cfg.num_workers)

        log.info('Get RF test predictions')
        y_test_pred_rf = rfmodel.predict(x_feats)

        # RF results
        results_rf, results_hmm_rf, results_hmm_learn_rf, cmatrix_rf, cmatrix_hmm_rf, cmatrix_hmm_learn_rf, step_tot, subjects = \
            zip(*Parallel(n_jobs=cfg.num_workers)(
                delayed(score)('RF', current_pid, group_test, x_test, y_test, y_test_pred_rf, 
                               time_test, cfg.data.winsec, hmm_rf, hmm_learn_rf, {}, le.classes_)
            for current_pid in tqdm(my_pids)
            ))

        results_rf = np.array(results_rf)
        results_hmm_rf = np.array(results_hmm_rf)
        results_hmm_learn_rf = np.array(results_hmm_learn_rf)

        cmatrix_rf = pd.DataFrame(np.sum(cmatrix_rf, axis=0), index=le.classes_, columns=le.classes_)
        cmatrix_hmm_rf = pd.DataFrame(np.sum(cmatrix_hmm_rf, axis=0), index=le.classes_, columns=le.classes_)
        cmatrix_hmm_learn_rf = pd.DataFrame(np.sum(cmatrix_hmm_learn_rf, axis=0), index=le.classes_, columns=le.classes_)

        # confusion matrix plots
        plots = {**plots, **{
            'matrix_rf': cmatrix_rf,
            'matrix_rf_hmm': cmatrix_hmm_rf,
            'matrix_rf_hmm_learn': cmatrix_hmm_learn_rf
        }}

        # save RF reports
        dfr_rf = utils.classification_report(results_rf, subjects, os.path.join(cfg.ukb_output_path, 
                                                                                'report_rf_{}.csv'.format(fold)))
        dfr_hmm_rf = utils.classification_report(results_hmm_rf, subjects, os.path.join(cfg.ukb_output_path, 
                                                                                        'report_rf_hmm_{}.csv'.format(fold)))
        dfr_hmm_learn_rf = utils.classification_report(results_hmm_learn_rf, subjects, os.path.join(cfg.ukb_output_path, 
                                                                                                    'report_rf_hmm_learn_{}.csv'.format(fold)))

        log.info('Results RF:\n%s', dfr_rf.mean())
        log.info('Results RF-HMM:\n%s', dfr_hmm_rf.mean())
        log.info('Results RF-HMM-Learn:\n%s', dfr_hmm_learn_rf.mean())
    
    for title in plots:
        matrix: pd.DataFrame = plots[title]
        matrix = matrix.div(matrix.sum(axis=1), axis=0).round(2)  # normalise
        plt.figure()
        sns.heatmap(matrix, annot=True, vmin=0, vmax=1)
        plt.title(title)
        plt.xticks(rotation=40, ha='right')
        # plt.ylabel('true', rotation=0), plt.xlabel('predicted')
        plt.tight_layout()
        plt.savefig('plots/{title}.png'.format(title=title), dpi=200)
        plt.close()

        log.info('Confusion %s\n%s\n', title, matrix)

def evaluate_folds(out_dir, folds, models=['rf', 'rf_hmm', 'rf_hmm_learn', 'ssl', 'ssl_hmm', 'ssl_hmm_learn'],
                   stratify_scores=False):
    master_report = pd.concat([
        pd.concat([pd.read_csv('{}/report_{}_{}.csv'.format(out_dir, model, fold), 
                               index_col=[0]) 
                   for fold in range(folds)]).add_suffix('_'+model)
        for model in models], axis=1)
    
    for f in glob(out_dir+"/*.csv"):
      os.remove(f)

    master_report.to_csv('outputs/master_report.csv')

    summary_scores = pd.DataFrame([{
      'f1': '{:.3f} [\u00B1{:.3f}]'.format(master_report['f1_'+model].mean(), master_report['f1_'+model].std()), 
      'kappa': '{:.3f} [\u00B1{:.3f}]'.format(master_report['kappa_'+model].mean(), master_report['kappa_'+model].std()),
      'accuracy': '{:.3f} [\u00B1{:.3f}]'.format(master_report['accuracy_'+model].mean(), master_report['accuracy_'+model].std())} 
      for model in models], index=models)
    
    summary_scores.to_csv(out_dir+'/summary_scores.csv')

    fig = go.Figure(data=[go.Table(
                    columnwidth=[100, 40, 40, 40],
                    header=dict(values=['Model', 
                                        'F1 score', 
                                        'Cohen\'s kappa score',
                                        'Accuracy score'],
                                        font_size=16,
                                        height=30),
                    cells=dict(values=[['Random Forest (RF)', 
                                        'RF + Hidden Markov Model', 
                                        'RF + Unsupervised Hidden Markov Model', 
                                        'Self supervised ResNet 18 (SSL)', 
                                        'SSL + Hidden Markov Model', 
                                        'SSL + Unsupervised Hidden Markov Model'], 
                               summary_scores['f1'],
                               summary_scores['kappa'],
                               summary_scores['accuracy']],
                               font_size=16,
                               height=30))
                 ])
    fig.write_image(out_dir+'/modelPerformance.png', width=1200, height=400)

    if stratify_scores:
        master_report['PD'] = ['_' in lab for lab in master_report.index]
        summary_scores_stratified = pd.DataFrame([{
          'f1_PD': '{:.3f} [\u00B1{:.3f}]'.format(master_report.loc[master_report.PD, 'f1_'+model].mean(), 
                                                  master_report.loc[master_report.PD, 'f1_'+model].std()),
          'f1_OxWalk': '{:.3f} [\u00B1{:.3f}]'.format(master_report.loc[~master_report.PD, 'f1_'+model].mean(), 
                                                      master_report.loc[~master_report.PD, 'f1_'+model].std()),
          'kappa_PD': '{:.3f} [\u00B1{:.3f}]'.format(master_report.loc[master_report.PD, 'kappa_'+model].mean(), 
                                                     master_report.loc[master_report.PD, 'kappa_'+model].std()),
          'kappa_OxWalk': '{:.3f} [\u00B1{:.3f}]'.format(master_report.loc[~master_report.PD, 'kappa_'+model].mean(), 
                                                         master_report.loc[~master_report.PD, 'kappa_'+model].std()),
          'accuracy_PD': '{:.3f} [\u00B1{:.3f}]'.format(master_report.loc[master_report.PD, 'accuracy_'+model].mean(), 
                                                        master_report.loc[master_report.PD, 'accuracy_'+model].std()),
          'accuracy_OxWalk': '{:.3f} [\u00B1{:.3f}]'.format(master_report.loc[~master_report.PD, 'accuracy_'+model].mean(), 
                                                            master_report.loc[~master_report.PD, 'accuracy_'+model].std())} 
          for model in models])
    
        summary_scores_stratified.to_csv(out_dir+'/summary_scores_stratified.csv')
    
        fig2 = go.Figure(data=[go.Table(
                            header=dict(values=['Model', 
                                                'MJFF-LR study F1 score',
                                                'OxWalk study F1 score', 
                                                'MJFF-LR Cohen\'s kappa score',
                                                'OxWalk Cohen\'s kappa score',
                                                'MJFF-LR Accuracy score',
                                                'OxWalk Accuracy score'],
                                                font_size=16,
                                                height=30),
                            cells=dict(values=[['Random Forest (RF)', 'RF + HMM', 'RF + HMMLearn', 
                                                'Self supervised ResNet 18 (SSL)', 
                                                'SSL + HMM', 'SSL + HMMLearn'], 
                                       summary_scores_stratified['f1_PD'],
                                       summary_scores_stratified['f1_OxWalk'],
                                       summary_scores_stratified['kappa_PD'],
                                       summary_scores_stratified['kappa_OxWalk'],
                                       summary_scores_stratified['accuracy_PD'],
                                       summary_scores_stratified['accuracy_OxWalk']],
                                       font_size=16,
                                       height=30))
                         ])
        fig2.write_image(out_dir+'/summary_scores_stratified.png', width=1200, height=800)

        return master_report, summary_scores, summary_scores_stratified

    return master_report, summary_scores

if __name__ == '__main__':
    cfg = OmegaConf.load("conf/config.yaml")

    np.random.seed(42)
    torch.manual_seed(42)
    log.info(str(OmegaConf.to_yaml(cfg)))

    evaluate_model(load_data(cfg)[0], cfg)
