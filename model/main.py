#!/usr/bin/env python
import numpy as np
import sys
import matplotlib.pyplot as plt
import math

from model.events import read_events, process_events
from model.openface import extract_features_from_video


TTL = 1300
PERCEPTUAL_CYCLE = 100
MISSCLICK_COGNITION_CYCLE = 70
AGGR_WINDOW = 10 * 1000
FITTS_LAW_A = 0
FITTS_LAW_B = 150  # calculated empirically


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


def plot_tasks(events, features):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(311)
    ax.grid()

    motor_end = 0
    max_time = math.ceil(events.index.values.max())
    cognitive_workload = np.zeros(max_time)
    perceptual_workload = np.zeros(max_time)
    motor_workload = np.zeros(max_time)
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

    for time, row in events.iterrows():
        if row.event == 'click_on_circle':
            start = row.task_start_time
            approx_motor_time = fitts_law(row.distance, row.circle_width, a=FITTS_LAW_A, b=FITTS_LAW_B)
            motor_start = max(start + row.reaction - approx_motor_time, motor_end)
            motor_time = start + row.reaction - motor_start
            motor_end = start + row.reaction

            # background
            y = concurrent_box(start, start + TTL, color='grey', alpha=0.5)

            # perceptual
            box(ax, start, start + PERCEPTUAL_CYCLE, y, color='yellow')
            fill(perceptual_workload, start, start + PERCEPTUAL_CYCLE)

            # cognitive
            box(ax, start + PERCEPTUAL_CYCLE, start + row.reaction - motor_time, y, color='blue')
            fill(cognitive_workload, start + PERCEPTUAL_CYCLE, start + row.reaction - motor_time)

            # motor
            box(ax, motor_start, motor_end, y, color='red')
            fill(motor_workload, motor_start, motor_end)

        if row.event == 'circle_missed':
            y = concurrent_box(row.task_start_time, row.task_start_time + TTL, color='grey', alpha=0.5)

            # perceptual
            box(ax, row.task_start_time, row.task_start_time + PERCEPTUAL_CYCLE, y, color='yellow')
            fill(perceptual_workload, row.task_start_time, row.task_start_time + PERCEPTUAL_CYCLE)

        if row.event == 'missclick':
            approx_motor_time = fitts_law(row.distance, row.circle_width, a=0, b=150)
            motor_start = max(time - approx_motor_time, motor_end)  # preventing motor overlapping
            motor_end = time
            motor_time = motor_end - motor_start

            missclick_subtask_start = time - motor_time
            missclick_subtask_end = time + PERCEPTUAL_CYCLE + MISSCLICK_COGNITION_CYCLE

            # background
            y = concurrent_box(missclick_subtask_start, time + PERCEPTUAL_CYCLE,
                               annotation='MC', color='purple', alpha=0.5)

            # perception
            box(ax, time, missclick_subtask_end, y, color='yellow')
            fill(perceptual_workload, time, missclick_subtask_end)

            # cognitive
            box(ax, time + PERCEPTUAL_CYCLE, time + PERCEPTUAL_CYCLE + MISSCLICK_COGNITION_CYCLE, y, color='blue')
            fill(cognitive_workload, time + PERCEPTUAL_CYCLE, time + PERCEPTUAL_CYCLE + MISSCLICK_COGNITION_CYCLE)

            # motor
            box(ax, missclick_subtask_start, time, y, color='red')
            fill(motor_workload, missclick_subtask_start, time)

    ax = fig.add_subplot(312, sharex=ax)
    ax.fill_between(np.linspace(0, max_time, len(cognitive_workload)), cognitive_workload, alpha=0.2, linewidth=0)
    ax.plot(aggregate_with_window(cognitive_workload, AGGR_WINDOW), color='blue', label='Cognitive workload')
    ax.plot(aggregate_with_window(perceptual_workload, AGGR_WINDOW), color='yellow', label='Perceptual workload')
    ax.plot(aggregate_with_window(motor_workload, AGGR_WINDOW), color='red', label='Motor workload')
    ax.legend(loc='upper left')

    ax = fig.add_subplot(313, sharex=ax)
    features_x = np.linspace(0, max_time, len(features))
    ax.plot(features_x, features.AU02_r.values)
    ax.plot(features_x, features.gaze_1_x.values)
    ax.plot(features_x, features.gaze_0_x.values)

    plt.show()


def main(video_path, events_path):
    print('Extracting features from video')
    features = extract_features_from_video(video_path)

    print('Reading events')
    events = process_events(read_events(events_path))

    plot_tasks(events, features)

    #taks_events = events.where(events.event == 'click_on_circle')
    #plt.plot(taks_events.index, taks_events.reaction)
    #plt.show()

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('You must specify a source video file and events file')
    else:
        main(sys.argv[1], sys.argv[2])
