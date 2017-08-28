#!/usr/bin/env python

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np

from model.openface import extract_features_from_video
from model.tasks import workload_from_tasks, tasks_from_kinect_events_file
from model.nn import train


def main(args):
    workload = []
    predictions = []

    for name in os.listdir(args.dir):
        path = os.path.join(args.dir, name)
        if os.path.isdir(path):
            print('##########################')
            print('Processing path %s' % path)

            video = None
            for filename in os.listdir(path):
                if filename.endswith("_color.avi"):
                    video = filename
                    break

            features = extract_features_from_video(os.path.join(path, video))
            tasks = tasks_from_kinect_events_file(os.path.join(path, 'game.csv'))
            workload = workload_from_tasks(tasks)
            model, predictions = train(workload, features, model_path=os.path.join(path, 'model_dropout_01.h5'))

            #time = len(workload)
            #plt.plot(np.linspace(0, time, len(workload)), workload, label='Workload')
            #plt.plot(np.linspace(0, time, len(predictions)), predictions, label='Prediction')
            #plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory containing kinect game results")
    main(parser.parse_args())
