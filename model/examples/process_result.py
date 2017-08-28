#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import argparse
import numpy as np
import pandas as pd
import sys

from model.tasks import tasks_from_events_file, tasks_from_kinect_events_file, workload_from_tasks
from model.openface import extract_features_from_video, FEATURE_SET
from model.nn import train, fit_to_size
from plot_tasks import plot_tasks
from scipy.interpolate import spline


feature_names = {
    'pose_Tx': 'Head X',
    'pose_Ty': 'Head Y',
    'pose_Tz': 'Head Z',
    'pose_Rx': 'Head rot X',
    'pose_Ry': 'Head rot Y',
    'pose_Rz': 'Head rot Z',
    'gaze_0_x': 'Gaze0 X',
    'gaze_0_y': 'Gaze0 Y',
    'gaze_1_x': 'Gaze1 X',
    'gaze_1_y': 'Gaze1 Y',
    'AU01_r': 'Inner Brow Raiser',
    'AU02_r': 'Outer Brow Raiser',
    'AU04_r': 'Brow Lowerer',
    'AU05_r': 'Upper Lid Raiser',
    'AU06_r': 'Cheek Raiser',
    'AU07_r': 'Lid Tightener',
    'AU09_r': 'Nose Wrinkler',
    'AU10_r': 'Upper Lip Raiser',
    'AU12_r': 'Lip Corner Puller',
    'AU14_r': 'Dimpler',
    'AU15_r': 'Lip Corner Depressor',
    'AU17_r': 'Chin Raiser',
    'AU20_r': 'Lip stretcher',
    'AU23_r': 'Lip Tightener',
    'AU25_r': 'Lips part**',
    'AU26_r': 'Jaw Drop',
    'AU45_r': 'Blink',
}


def get_feature_name(col):
    if col in feature_names:
        return feature_names[col]
    return col

SELECTED_FEATURES = [
    'gaze_0_x',
    'gaze_0_y',
    'AU01_r',
    'AU04_r',
    'AU06_r',
    'AU10_r',
    'AU15_r',
    'AU17_r',
    'AU20_r',
]
SELECTED_FEATURES = FEATURE_SET
SPLINE_POINTS = 500
SUBPlOT_CFG = {
    'left': 0.05,
    'right': 0.95,
    'top': 0.95,
    'bottom': 0.1,
    'hspace': 0.0
}
XTICK_POSITIONS = np.linspace(0, 180000, 10)
XTICK_LABELS = ['%d' % (i / 1000) for i in XTICK_POSITIONS]
FIG_WIDTH = 9


def main(args):
    total_time = 176000
    data_path = os.path.join(os.path.dirname(args.events), 'data.csv')

    features = extract_features_from_video(args.video)
    frames = len(features)
    if args.kinect:
        tasks = tasks_from_kinect_events_file(args.events)
    else:
        tasks = tasks_from_events_file(args.events)

    if not os.path.exists(data_path):
        target_workload = workload_from_tasks(tasks)[0:total_time]
        model, model_workload = train(target_workload, features,
                                      time_window=200,
                                      model_path=os.path.join(os.path.dirname(args.events), 'model_128_dropout_01.h5'))
        features['target_workload'] = fit_to_size(target_workload, frames)
        features['model_workload'] = model_workload
        features.to_csv(data_path)
    else:
        features = pd.read_csv(data_path)

    # PLOT
    fps = 33.32
    axix = 0
    n_features = len(SELECTED_FEATURES)
    x = np.linspace(0, total_time, len(features))
    f, axarr = plt.subplots(n_features, sharex=True, figsize=(FIG_WIDTH, 10))
    plt.subplots_adjust(**SUBPlOT_CFG)

    # FEATURES
    for col in SELECTED_FEATURES:
        y = features[col].values
        y = (y - y.min()) / (y.max() - y.min())


        xx = np.linspace(x.min(), x.max(), SPLINE_POINTS)
        poly_coeffs = 10
        #x_spline = np.linspace(x.min(), x.max(), SPLINE_POINTS)
        #y_spline = spline(x, y, x_spline)
        x_spline = xx
        y_spline = fit_to_size(y, SPLINE_POINTS)

        ax = axarr[axix]
        ax.set_ylim([0, 1])
        #ax.set_title(get_feature_name(col), y=0.2, x=0.95)
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels([''] * 3)
        ax.plot(x_spline, y_spline, color='black')
        ax.fill_between(x_spline, 0, y_spline, color='0.2', alpha=0.5, label=get_feature_name(col))
        ax.grid()
        #ax.plot(x, y, color='black', alpha=0.8)
        #ax.legend()
        axix += 1

    ax.set_xticks(XTICK_POSITIONS)
    ax.set_xticklabels(XTICK_LABELS)
    # TASKS
    #if tasks is not None:
        #plot_tasks(tasks, axarr[axix])
        #axix += 1
    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(FIG_WIDTH, 1.5), sharex=True)
    plt.subplots_adjust(**SUBPlOT_CFG)
    ax.set_xlim([0, total_time])
    plot_tasks(tasks, ax)
    plt.show()

    target_workload = features.target_workload.values
    model_workload = features.model_workload.values

    fig, axarr = plt.subplots(nrows=2, ncols=1, figsize=(FIG_WIDTH, 2), sharex=True)

    cfg = SUBPlOT_CFG
    cfg['hspace'] = 0.2
    plt.subplots_adjust(**cfg)
    axarr[0].set_ylim([0, 1])
    axarr[0].plot(x, target_workload, linestyle='--', color='black')
    axarr[0].fill_between(x, 0, target_workload, color='0.9', alpha=0.6)
    axarr[0].grid()

    axarr[1].set_ylim([0, 1])
    #axarr[1].set_yticklabels([''] * 3)
    axarr[1].plot(x, model_workload, linestyle='-', color='black')
    axarr[1].fill_between(x, 0, model_workload, color='0.5', alpha=0.6)
    axarr[1].grid()
    axarr[1].set_xticks(XTICK_POSITIONS)
    axarr[1].set_xticklabels(XTICK_LABELS)

    #plt.fill_between(x, model_workload, target_workload, color='0.9')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("events", help="Path to the events.json file")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--kinect", help="Flag to indicate a kinect events file", action='store_true')
    main(parser.parse_args())
