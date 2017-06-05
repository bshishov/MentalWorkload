#!/usr/bin/env python
import numpy as np
import os
import matplotlib.pyplot as plt


WORK_DIR = 'res/'
mse_errors = []
workloads = []

for name in os.listdir(WORK_DIR):
    path = os.path.join(WORK_DIR, name)
    if 'kinect' in name and name.endswith('.csv'):
        res = np.loadtxt(os.path.join(WORK_DIR, name))
        if len(res) > 0:
            w = res[0]
            p = res[1]
            mse = ((w - p) ** 2).mean()
            mse_errors.append(mse)
            workloads.append(np.pad(w, (0, 5500 - len(w)), mode='constant'))

workloads = np.array(workloads)
print(workloads.shape)
means = workloads.mean(axis=1)
stds = workloads.std(axis=1)
mins = workloads.min(axis=1)
maxs = workloads.min(axis=1)

wt = workloads.T
w = np.zeros(shape=(60, 21))
window = wt.shape[0] / (w.shape[0] + 1)
print(window)
for i in range(w.shape[0]):
    w[i, :] = wt[i*window:window*(i+1), :].mean(axis=0)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot(w.T)
ax.grid()
TICKS = 15
ax.set_xticks(np.linspace(0, w.shape[0], TICKS))
ax.set_xticklabels(np.round(np.linspace(0, 4 * 60 * 1000, TICKS)))
ax.tick_params(axis='x', pad=20)
plt.xlabel('Time (ms)')
plt.show()
print(means.shape)

mse_errors = np.array(mse_errors)
plt.plot(mse_errors)
plt.show()

print('MSE Mean: %f' % mse_errors.mean())
print('MSE STD: %f' % mse_errors.std())
