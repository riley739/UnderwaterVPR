import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from config import config

cv2.namedWindow("Camera Output")

sonar_config = config['agents'][0]['sensors'][-1]["configuration"]
azi = sonar_config['Azimuth']
minR = sonar_config['RangeMin']
maxR = sonar_config['RangeMax']
binsR = sonar_config['RangeBins']
binsA = sonar_config['AzimuthBins']

plt.ion()
fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(8,5))
ax.set_theta_zero_location("N")
ax.set_thetamin(-azi/2)
ax.set_thetamax(azi/2)

theta = np.linspace(-azi/2, azi/2, binsA)*np.pi/180
r = np.linspace(minR, maxR, binsR)
T, R = np.meshgrid(theta, r)
z = np.zeros_like(T)

plt.grid(False)
plot = ax.pcolormesh(T, R, z, cmap='gray', shading='auto', vmin=0, vmax=1)
plt.tight_layout()
fig.canvas.draw()
fig.canvas.flush_events()

dataset = "datasets/clear_3targets"

with open(f'{dataset}/dataset.csv', mode ='r')as file:
  csvFile = csv.reader(file)
  for line in csvFile:
    # print(line)

    if line[1] == "camera":
        frame = cv2.imread(f"{dataset}/camera/im_{int(line[2]):05}.jpg")
        cv2.imshow("Camera Output", frame)
        cv2.waitKey(1)
    if line[1] == "sonar":
        s = np.loadtxt(f"{dataset}/sonar/im_{int(line[2]):05}.txt")
        plot.set_array(s.ravel())

        fig.canvas.draw()
        fig.canvas.flush_events()

plt.ioff()
cv2.destroyAllWindows()