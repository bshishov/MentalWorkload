#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import argparse

from model.tasks import tasks_from_events_file, tasks_from_kinect_events_file


# Function to plot boxes
def box(ax, start, end, y, annotation=None, **kwargs):
    if annotation is not None:
        ax.text(start + 0.01, y + 0.01, annotation)
    ax.barh(y, end-start, 0.8, start, **kwargs)


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


def main(args):
    if args.kinect:
        tasks = tasks_from_kinect_events_file(args.events_file)
    else:
        tasks = tasks_from_events_file(args.events_file)
    fig = plt.figure(figsize=(16, 3))
    ax = fig.add_subplot(111)
    ax.grid()
    plot_tasks(tasks, ax, with_breakdown=True)
    plt.xlabel('Time (ms)')
    plt.savefig(args.events_file + '.tasks.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("events_file", help="Path to the events.json file")
    parser.add_argument("--kinect", help="Flag to indicate a kinect events file", action='store_true')
    main(parser.parse_args())
