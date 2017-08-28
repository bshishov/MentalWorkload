#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

SHOT_EVERY = 5400.0 / 8.0
SHOT_COLS = 10
SHOT_ROWS = 1
SHOT_WIDTH = 100
SHOW_HEIGHT = 100

ROI = (210, 20, 200, 200)


def process_video(path):
    cv2.namedWindow('main')
    cap = cv2.VideoCapture(path)
    index = SHOT_EVERY - 5
    shot = 0
    shots = np.zeros((SHOT_ROWS * SHOW_HEIGHT, SHOT_COLS * SHOT_WIDTH, 3), np.uint8)
    row, col = 0, 0
    while cap.isOpened():
        result, frame = cap.read()
        if not result:
            break
        index += 1
        part = frame[ROI[1]:ROI[1] + ROI[3], ROI[0]:ROI[0] + ROI[2]]
        part = cv2.resize(part, (SHOT_WIDTH, SHOW_HEIGHT), interpolation=cv2.INTER_CUBIC)
        if index > SHOT_EVERY:
            index = 0
            col = shot % SHOT_COLS
            row = shot / SHOT_COLS
            shot += 1

        shots[row * SHOW_HEIGHT:(row + 1) * SHOW_HEIGHT, col * SHOT_WIDTH:(col + 1) * SHOT_WIDTH] = part
        cv2.rectangle(shots,
                      (col * SHOT_WIDTH, row * SHOW_HEIGHT),
                      ((col + 1) * SHOT_WIDTH, (row + 1) * SHOW_HEIGHT - 1),
                      (0, 0, 0), 1)

        cv2.imshow('main', shots)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.imwrite(path + '.shots.png', shots)
    cv2.waitKey()
    cap.release()


if __name__ == '__main__':
    #process_video('Y:\\Учеба\\Kinect\\recordings\\18-kinect20150528150849 - ок 5\\kinect20150528150849_color.avi')
    #process_video('Y:\\Учеба\\Kinect\\recordings\\19-kinect20150528163006 - ок 5\\kinect20150528163006_color.avi')
    #process_video('Y:\\Учеба\\Kinect\\recordings\\12-kinect20150528144037 - ок 5\\kinect20150528144037_color.avi')
    process_video('Y:\\Учеба\\Kinect\\recordings\\13-kinect20150528144516 - ок 5\\kinect20150528144516_color.avi')
    #process_video('Y:\\Учеба\\Kinect\\recordings\\13-kinect20150528144516 - ок 5\\tracked.avi')
