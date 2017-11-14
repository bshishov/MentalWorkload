#!/usr/bin/env python

import argparse
import os
import numpy as np

from model.openface import extract_features_from_video
from model.tasks import tasks_from_kinect_events_file, cognitive_effort_from_tasks, workload_from_effort


def main(args):
    index = 0
    log = []
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

            features = extract_features_from_video(os.path.join(path, video), raw=True)
            features.to_csv(os.path.join(args.output, '{0:02d}_features.csv'.format(index)))

            tasks = tasks_from_kinect_events_file(os.path.join(path, 'game.csv'))

            effort = cognitive_effort_from_tasks(tasks)
            workload = workload_from_effort(effort)

            target = np.stack((effort, workload), axis=-1)
            np.save(os.path.join(args.output, '{0:02d}_target.npy'.format(index)), target)
            np.savetxt(os.path.join(args.output, '{0:02d}_target.csv'.format(index)), target, delimiter=',')
            log.append((index, path))
            index += 1
    with open(os.path.join(args.output, 'log.txt')) as f:
        for index, path in log:
            f.write('{0:02d}, {1}\n'.format(index, path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory containing kinect game results")
    parser.add_argument("output", help="Directory containing kinect game results")
    main(parser.parse_args())
