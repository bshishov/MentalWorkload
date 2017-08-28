#!/usr/bin/env python

import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import os

from model.tasks import read_events, process_events, read_kinect_events, process_kinect_events, tasks_from_events, cognitive_effort_from_tasks, workload_from_effort
from model.openface import extract_features_from_video
from model.nn import train


def fit_to_size(a, l):
    w = np.zeros(l)
    window_size = len(a) / (l + 1)
    for i in range(l):
        w[i] = a[window_size * i: window_size * (i + 1)].mean()
    return w


def main(video_path, events_path, is_kinect=False, model_path=None, quite=False):
    print('Extracting features from video')
    features = extract_features_from_video(video_path)

    print('Reading events')
    if not is_kinect:
        events = process_events(read_events(events_path))
    else:
        events = process_kinect_events(read_kinect_events(events_path))

    print('Breaking down events into tasks')
    tasks = tasks_from_events(events)

    print('Computing cognitive effort')
    cognitive_effort = cognitive_effort_from_tasks(tasks)

    print('Computing cognitive workload')
    cognitive_workload = workload_from_effort(cognitive_effort)

    print('Training a model')
    model, predictions = train(cognitive_workload, features, model_path=model_path)


    # COGNITIVE WORKLOAD and FEATURES
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(211)
    ax.grid()
    ax.plot(cognitive_workload, 'b-', label='Cognitive workload')
    #ax.plot(np.linspace(0, MAX_TIME, len(predictions)), predictions, 'k-', label='Predicted cognitive workload')
    ax.legend(loc='upper left')

    poly_coeffs = 15
    video_x = np.linspace(0, math.ceil(events.index.values.max()), len(features))
    au01 = np.poly1d(np.polyfit(video_x, features.AU01_r, poly_coeffs))
    au02 = np.poly1d(np.polyfit(video_x, features.AU02_r, poly_coeffs))
    au04 = np.poly1d(np.polyfit(video_x, features.AU04_r, poly_coeffs))
    au10 = np.poly1d(np.polyfit(video_x, features.AU10_r, poly_coeffs))
    au15 = np.poly1d(np.polyfit(video_x, features.AU15_r, poly_coeffs))
    au45 = np.poly1d(np.polyfit(video_x, features.AU45_r, poly_coeffs))

    ax = fig.add_subplot(212, sharex=ax)
    ax.grid()
    ax.plot(video_x, au01(video_x), 'k-', label='AU 1 (Inner Brow Raiser)')
    ax.plot(video_x, au02(video_x), 'k--', label='AU 2 (Outer Brow Raiser)')
    ax.plot(video_x, au04(video_x), 'k-.', label='AU 4 (Brow Lowerer)')
    ax.plot(video_x, au10(video_x), 'k^:', markevery=70, label='AU 10 (Upper Lip Raiser)')
    ax.plot(video_x, au15(video_x), 'k*:', markevery=70, label='AU 15 (Lip Corner Depressor)')
    #ax.plot(video_x, au45(video_x), 'k:', label='AU 45 (Blink)')
    #ax.plot(video_x, features.AU45_r, 'k:', label='AU 45 (Blink)')
    ax.legend(loc='upper left')
    plt.xlabel('Time (ms)')

    if not quite:
        plt.show()
    else:
        plt.savefig(model_path + '.workload.png')
        plt.close(fig)
    return cognitive_workload, predictions


def process_all_kinect():
    workload = []
    predictions = []

    base_dir = 'Y:\\Учеба\\Kinect\\recordings\\'
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isdir(path):
            print('##########################')
            print('Processing path %s' % path)

            video = None
            for filename in os.listdir(path):
                if filename.endswith("_color.avi"):
                    video = filename
                    break

            w, p = main(
                os.path.join(path, video),
                os.path.join(path, 'game.csv'),
                is_kinect=True,
                model_path=os.path.join(path, 'model.h5'), quite=True)

            w = fit_to_size(w, len(p))
            print(w.shape)
            print(p.shape)
            res = np.array([w, p])
            np.savetxt('res/' + name + '.csv', res)

if __name__ == '__main__':
    #process_all_kinect()


    if len(sys.argv) < 3:
        print('You must specify a source video file and events file')
    else:
        main(sys.argv[1], sys.argv[2], model_path=sys.argv[1] + '.model.h5')
