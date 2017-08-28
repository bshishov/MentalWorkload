#!/usr/bin/env python

import sys
import os
import subprocess
import pandas

OPENFACE_FEATURE_EXTRACTION_PATH = 'D:\\OpenFace\\FeatureExtraction.exe'

# OPENFACE output format:
# AUXX_r - intensity, [0..5]
# AUXX_c - presence, [0..1]
FEATURES_AU = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r',
    'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r',
    'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r',
]
FEATURES_GAZE = [
    'gaze_0_x', 'gaze_0_y', 'gaze_1_x', 'gaze_1_y',
]
FEATURES_POSE = [
    'pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz',
]

FEATURE_SET = FEATURES_POSE + FEATURES_GAZE + FEATURES_AU
#FEATURE_SET = FEATURES_AU  # ONLY AU


def extract_features_from_video(source, save_to=None, quite=False, output_video=None, verbose=True,
                                process_if_exists=False):
    if save_to is None:
        save_to = source + '.features.csv'
    if process_if_exists or not os.path.exists(save_to):
        args = [OPENFACE_FEATURE_EXTRACTION_PATH, '-f', source, '-of', save_to, '-world_coord 0', '-no2Dfp', '-no3Dfp']
        if quite:
            args.append('-q')
        if verbose:
            args.append('-verbose')
        if output_video is not None:
            args += ['-ov', output_video]
        subprocess.call(args)
    else:
        print('Video \"%s\" is already preprocessed, skipping feature extraction. '
              'Features are described in the file: \"%s\"' % (source, save_to))
    data = pandas.read_csv(save_to, sep=', ')
    data[FEATURES_AU] /= 5
    return data[FEATURE_SET]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('You must specify a video source')
    else:
        extract_features_from_video(sys.argv[1])
