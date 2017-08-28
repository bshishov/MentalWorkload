#!/usr/bin/env python

import sys
import pandas
import json
import numpy as np
import math


TTL = 1300
CIRCLE_WIDTH = 35 * 2
AVERAGE_KINECT_DISTANCE = 200

PERCEPTUAL_CYCLE = 100
MISSCLICK_COGNITION_CYCLE = 70
FITTS_LAW_A = 0
FITTS_LAW_B = 150  # calculated empirically
MAX_TIME = 4 * 60 * 1000  # ms (total experiment time)
AGGR_WINDOW = 10 * 1000


def _distance(x1, y1, x2, y2):
    x = x2 - x1
    y = y2 - y1
    return np.sqrt(x * x + y * y)


def read_events(filename):
    with open(filename, encoding="utf8") as events_file:
        return json.loads(events_file.read())


def process_events(events):
    df = pandas.DataFrame(columns=[
        'time',
        'event',
        'task_id',
        'task_delay',
        'reaction',
        'distance',
        'circle_width',
        'task_start_time'], dtype=np.uint8)

    last_x = 0
    last_y = 0
    last_click_x = 0
    last_click_y = 0
    last_task_time = 0

    for event in events:
        t = event['time']
        name = event['name']
        if 'args' in event:
            args = event['args']

        if name == 'click_on_circle':
            start_time = t - args['reaction']
            df = df.append({
                "time": t,
                "event": name,
                "task_id": args['id'],
                "task_delay": start_time - last_task_time,
                "reaction": args['reaction'],
                "distance": _distance(last_x, last_y, last_click_x, last_click_y),
                "circle_width": CIRCLE_WIDTH,
                "task_start_time": start_time
            }, ignore_index=True)

        elif name == 'missclick':
            df = df.append({
                "time": t,
                "event": name,
                "distance": _distance(last_x, last_y, last_click_x, last_click_y),
                "circle_width": CIRCLE_WIDTH,
            }, ignore_index=True)

        elif name == 'circle_missed':
            start_time = t - TTL
            df = df.append({
                "time": t,
                "event": name,
                "task_id": args['id'],
                "task_delay": start_time - last_task_time,
                "circle_width": CIRCLE_WIDTH,
                "task_start_time": start_time
            }, ignore_index=True)

        elif name == 'spawn':
            last_task_time = t

        elif name == 'mouse_move':
            last_x = args['x']
            last_y = args['y']

        elif name == 'mouse_click':
            last_click_x = last_x
            last_click_y = last_x

    df = df.set_index('time')
    print(df.head())
    return df


# read events for results produced by Kinect software (http://github.com/bshishov/Emotions)
def read_kinect_events(filename):
    df = pandas.read_csv(filename, skiprows=1)
    df = df.set_index('FrameNumber')
    t0 = df.ix[0].Time
    df.Time = (df.Time - t0) / 10000
    return df


# process events for results produced by Kinect software (http://github.com/bshishov/Emotions)
def process_kinect_events(events):
    df = pandas.DataFrame(columns=[
        'time',
        'event',
        'task_id',
        'task_delay',
        'reaction',
        'distance',
        'circle_width',
        'task_start_time'], dtype=np.uint8)
    last_missed = 0
    last_missclicks = 0
    last_scored = 0
    last_failed = 0  # red circles
    last_task_time = 0
    id = 0

    for frame, row in events.iterrows():
        if row.Scored > last_scored:
            start_time = row.Time - row.ReactionTime
            df = df.append({
                "time": row.Time,
                "event": 'click_on_circle',
                "task_id": id,
                "task_delay": start_time - last_task_time,
                "reaction": row.ReactionTime,
                "distance": AVERAGE_KINECT_DISTANCE,
                "circle_width": CIRCLE_WIDTH,
                "task_start_time": start_time
            }, ignore_index=True)
            last_task_time = max(last_task_time, start_time)
            id += 1

        if row.Missclicks > last_missclicks:
            df = df.append({
                "time": row.Time,
                "event": 'missclick',
                "distance": AVERAGE_KINECT_DISTANCE,
                "circle_width": CIRCLE_WIDTH,
            }, ignore_index=True)

        if row.Missed > last_missed:
            start_time = row.Time - TTL
            df = df.append({
                "time": row.Time,
                "event": 'circle_missed',
                "task_id": id,
                "task_delay": start_time - last_task_time,
                "circle_width": CIRCLE_WIDTH,
                "task_start_time": start_time
            }, ignore_index=True)
            last_task_time = max(last_task_time, start_time)
            id += 1

        # update last values
        last_missed = row.Missed
        last_missclicks = row.Missclicks
        last_scored = row.Scored
        last_failed = row.Failed

    df = df.set_index('time')
    print(df.head())
    return df


class Task(object):
    id = None
    name = None
    time = 0
    perception_time = 0
    cognition_time = 0
    motor_time = 0


def fitts_law(distance, width, a=0, b=1):
    return a + b * np.log2(2 * distance / width)


def tasks_from_events(events):
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


def tasks_from_events_file(filename):
    events = process_events(read_events(filename))
    return tasks_from_events(events)


def tasks_from_kinect_events_file(filename):
    events = process_kinect_events(read_kinect_events(filename))
    return tasks_from_events(events)


def _fill(x, fill_from, fill_to, val=1.0):
    x[math.floor(fill_from):math.ceil(fill_to)] = val


def _aggregate_with_window(x, time_window):
    x_aggr = np.zeros(len(x))
    for i in range(time_window):
        x_aggr[i] = np.mean(x[0:i])
    for i in range(time_window, len(x)):
        x_aggr[i] = np.mean(x[i - time_window:i])
    return x_aggr


def cognitive_effort_from_tasks(tasks):
    cognitive_effort = np.zeros(MAX_TIME)

    for task in tasks:
        if task.cognition_time is not None:
            _fill(cognitive_effort, task.cognition_time[0], task.cognition_time[1])

    return cognitive_effort


def workload_from_effort(effort, window_size=AGGR_WINDOW):
    return _aggregate_with_window(effort, window_size)


def workload_from_tasks(tasks, window_size=AGGR_WINDOW):
    effort = cognitive_effort_from_tasks(tasks)
    return workload_from_effort(effort, window_size)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('You must specify an events JSON file')
    else:
        process_events(read_events(sys.argv[1]))
