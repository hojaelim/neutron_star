import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

data = np.genfromtxt('output.txt', delimiter = ' ', skip_header=0)

data = data.T

pressure = data[0]
mass = data[1]
radius = data[2]

f1 = interpolate.interp1d(pressure, radius, kind='linear')
f2 = interpolate.interp1d(pressure, mass, kind='linear')

new_pressure = np.geomspace(1.0*10**34, 1.0*10**36, 10000)
new_rad = f1(new_pressure)
new_mass = f2(new_pressure)

max_index=np.where(new_mass==np.max(new_mass))[0][0]
print(new_pressure[max_index], new_mass[max_index], new_rad[max_index])

"""
fig,ax = plt.subplots()
ax.plot(new_pressure, new_rad, color='red')
ax2 = ax.twinx()
ax2.plot(new_pressure,new_mass, color='blue')
ax.set_xscale('log')
plt.show()
"""
