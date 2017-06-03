#!/usr/bin/env python
import sys
import os
import subprocess
import pandas

OPENFACE_FEATURE_EXTRACTION_PATH = 'D:\\OpenFace\\FeatureExtraction.exe'


def extract_features_from_video(source, save_to=None, quite=False, output_video=None, verbose=True,
                                process_if_exists=False):
    if save_to is None:
        save_to = source + '.features.csv'
    if process_if_exists or not os.path.exists(save_to):
        args = [OPENFACE_FEATURE_EXTRACTION_PATH, '-f', source, '-of', save_to]
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
    return pandas.read_csv(save_to, sep=', ')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('You must specify a video source')
    else:
        extract_features_from_video(sys.argv[1])
