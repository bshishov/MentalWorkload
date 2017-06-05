#!/usr/bin/env python
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import os
import glob
import json

from model.events import read_events, process_events, read_kinect_events, process_kinect_events
from model.openface import extract_features_from_video
from model.train import samples_as_sequences


TTL = 1300
PERCEPTUAL_CYCLE = 100
MISSCLICK_COGNITION_CYCLE = 70
AGGR_WINDOW = 10 * 1000
FITTS_LAW_A = 0
FITTS_LAW_B = 150  # calculated empirically
MAX_TIME = 4 * 60 * 1000  # ms (total experiment time)


def fitts_law(distance, width, a=0, b=1):
    return a + b * np.log2(2 * distance / width)


def aggregate_with_window(x, time_window):
    x_aggr = np.zeros(len(x))
    for i in range(time_window, len(x)):
        x_aggr[i] = np.mean(x[i - AGGR_WINDOW:i])
    return x_aggr


# Function to plot boxes
def box(ax, start, end, y, annotation=None, **kwargs):
    if annotation is not None:
        ax.text(start + 0.01, y + 0.01, annotation)
    ax.barh(y, end-start, 0.8, start, **kwargs)


def fill(x, fill_from, fill_to, val=1.0):
    x[math.floor(fill_from):math.ceil(fill_to)] = 1.0


class Task(object):
    pass


def breakdown_tasks(events):
    tasks = []
    motor_end = 0

    for time, row in events.iterrows():
        if row.event == 'click_on_circle':
            start = row.task_start_time

            # MOTOR TIME CALCULATION
            approx_motor_time = fitts_law(row.distance, row.circle_width, a=FITTS_LAW_A, b=FITTS_LAW_B)
            motor_start = max(start + row.reaction - approx_motor_time, motor_end)
            motor_time = start + row.reaction - motor_start
            motor_end = start + row.reaction

            task = Task()
            task.id = row.task_id
            task.name = row.event
            task.time = (start, start + TTL)
            task.perception_time = (start, start + PERCEPTUAL_CYCLE)
            task.cognition_time = (start + PERCEPTUAL_CYCLE, start + row.reaction - motor_time)
            task.motor_time = (motor_start, motor_end)
            tasks.append(task)

        if row.event == 'circle_missed':
            start = row.task_start_time
            end = row.task_start_time + PERCEPTUAL_CYCLE

            task = Task()
            task.name = row.event
            task.time = (start, start + TTL)
            task.perception_time = (start, end)
            task.cognition_time = None
            task.motor_time = None
            tasks.append(task)

        if row.event == 'missclick':
            # MOTOR TIME CALCULATION
            approx_motor_time = fitts_law(row.distance, row.circle_width, a=FITTS_LAW_A, b=FITTS_LAW_B)
            motor_start = max(time - approx_motor_time, motor_end)  # preventing motor overlapping
            motor_end = time
            motor_time = motor_end - motor_start

            missclick_subtask_start = time - motor_time
            missclick_subtask_end = time + PERCEPTUAL_CYCLE + MISSCLICK_COGNITION_CYCLE

            task = Task()
            task.id = row.task_id
            task.name = row.event
            task.time = (missclick_subtask_start, time + PERCEPTUAL_CYCLE + MISSCLICK_COGNITION_CYCLE)
            task.perception_time = (time, missclick_subtask_end)
            task.cognition_time = (time + PERCEPTUAL_CYCLE, time + PERCEPTUAL_CYCLE + MISSCLICK_COGNITION_CYCLE)
            task.motor_time = (missclick_subtask_start, time)
            tasks.append(task)

    return tasks


def compute_cognitive_effort(tasks):
    cognitive_effort = np.zeros(MAX_TIME)

    for task in tasks:
        if task.cognition_time is not None:
            fill(cognitive_effort, task.cognition_time[0], task.cognition_time[1])

    return cognitive_effort


def plot_tasks(tasks, ax, with_breakdown=False):
    concurrent = []

    def concurrent_box(start, end, **kwargs):
        y = -1
        for i, t in enumerate(concurrent):
            if start > t:
                concurrent[i] = end
                y = i
                break

        if y < 0:
            concurrent.append(end)
            y = len(concurrent) - 1
        box(ax, start, end, y, **kwargs)
        return y

    for task in tasks:
        if with_breakdown:
            y = concurrent_box(task.time[0], task.time[1], color='grey', alpha=0.2)

            if task.perception_time is not None:
                box(ax, task.perception_time[0], task.perception_time[1], y, color='yellow')

            if task.cognition_time is not None:
                box(ax, task.cognition_time[0], task.cognition_time[1], y, color='blue')

            if task.motor_time is not None:
                box(ax, task.motor_time[0], task.motor_time[1], y, color='red')
        else:
            y = concurrent_box(task.time[0], task.time[1], color='grey', alpha=1)

    if with_breakdown:
        ax.legend(handles=[
            mpatches.Patch(color='grey', label='Task'),
            mpatches.Patch(color='yellow', label='Perception'),
            mpatches.Patch(color='blue', label='Cognition'),
            mpatches.Patch(color='red', label='Motor'),
        ])
    else:
        ax.legend(handles=[mpatches.Patch(color='grey', label='Task')])


def fit_to_size(a, l):
    w = np.zeros(l)
    window_size = len(a) / (l + 1)
    for i in range(l):
        w[i] = a[window_size * i: window_size * (i + 1)].mean()
    return w


def train(workload, features, model_path=None):
    train_cols = [
        'pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz',
        'gaze_0_x', 'gaze_0_y', 'gaze_1_x', 'gaze_1_y',
        'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r',
        'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r',
        'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r',
    ]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    x = scaler.fit_transform(features[train_cols].values)
    y = fit_to_size(workload, len(features))

    time_window = 100
    model = None
    if model_path is not None and os.path.exists(model_path):
        from keras.models import load_model
        print('Found already trained model %s, loading' % model_path)
        model = load_model(model_path)
    else:
        from model.train import train_rnn, train_dense

        print('Training RNN LTSM model')
        model = train_rnn(x, y, time_window=time_window, save_to=model_path)

    predictions = model.predict(samples_as_sequences(x, time_window=time_window)).flatten()
    return model, predictions


def main(video_path, events_path, is_kinect=False, model_path=None, quite=False):
    print('Extracting features from video')
    features = extract_features_from_video(video_path)

    print('Reading events')
    if not is_kinect:
        events = process_events(read_events(events_path))
    else:
        events = process_kinect_events(read_kinect_events(events_path))

    print('Breaking down events into tasks')
    tasks = breakdown_tasks(events)

    print('Computing cognitive effort')
    cognitive_effort = compute_cognitive_effort(tasks)

    print('Computing cognitive workload')
    cognitive_workload = aggregate_with_window(cognitive_effort, AGGR_WINDOW)

    print('Training a model')
    model, predictions = train(cognitive_workload, features, model_path=model_path)

    # TASKS
    fig = plt.figure(figsize=(16,3))
    ax = fig.add_subplot(111)
    ax.grid()
    plot_tasks(tasks, ax, with_breakdown=True)
    plt.xlabel('Time (ms)')
    if not quite:
        plt.show()
    else:
        plt.savefig(model_path + '.tasks.png')
        plt.close(fig)

    # COGNITIVE WORKLOAD and FEATURES
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(211)
    ax.grid()
    ax.plot(cognitive_workload, 'k:', label='Cognitive workload')
    ax.plot(np.linspace(0, MAX_TIME, len(predictions)), predictions, 'k-', label='Predicted cognitive workload')
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
    ax.plot(video_x, features.AU45_r, 'k:', label='AU 45 (Blink)')
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
    process_all_kinect()
    pass

    if len(sys.argv) < 3:
        print('You must specify a source video file and events file')
    else:
        main(sys.argv[1], sys.argv[2], model_path=sys.argv[1] + '.model.h5')
