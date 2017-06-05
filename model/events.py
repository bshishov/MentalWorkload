#!/usr/bin/env python
import sys
import pandas
import json
import numpy as np


TTL = 1300
WIDTH = 35 * 2
AVERAGE_KINECT_DISTANCE = 200

def read_events(filename):
    with open(filename, encoding="utf8") as events_file:
        return json.loads(events_file.read())


# DEPRECATED
def read_kinect_events(filename):
    df = pandas.read_csv(filename, skiprows=1)
    df = df.set_index('FrameNumber')
    t0 = df.ix[0].Time
    df.Time = (df.Time - t0) / 10000
    return df


def _distance(x1, y1, x2, y2):
    x = x2 - x1
    y = y2 - y1
    return np.sqrt(x * x + y * y)


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
                "circle_width": WIDTH,
                "task_start_time": start_time
            }, ignore_index=True)

        elif name == 'missclick':
            df = df.append({
                "time": t,
                "event": name,
                "distance": _distance(last_x, last_y, last_click_x, last_click_y),
                "circle_width": WIDTH,
            }, ignore_index=True)

        elif name == 'circle_missed':
            start_time = t - TTL
            df = df.append({
                "time": t,
                "event": name,
                "task_id": args['id'],
                "task_delay": start_time - last_task_time,
                "circle_width": WIDTH,
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


# DEPRECATED
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
                "circle_width": WIDTH,
                "task_start_time": start_time
            }, ignore_index=True)
            last_task_time = max(last_task_time, start_time)
            id += 1

        if row.Missclicks > last_missclicks:
            df = df.append({
                "time": row.Time,
                "event": 'missclick',
                "distance": AVERAGE_KINECT_DISTANCE,
                "circle_width": WIDTH,
            }, ignore_index=True)

        if row.Missed > last_missed:
            start_time = row.Time - TTL
            df = df.append({
                "time": row.Time,
                "event": 'circle_missed',
                "task_id": id,
                "task_delay": start_time - last_task_time,
                "circle_width": WIDTH,
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


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('You must specify an events JSON file')
    else:
        process_events(read_events(sys.argv[1]))
