import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker


data = np.genfromtxt('output.txt', delimiter=' ', skip_header=0)

data = data.T

pressure = data[0]
mass = data[1]
radius = data[2]

max_index = mass.argmax()

print(pressure[max_index], mass[max_index], radius[max_index])


fig,ax = plt.subplots()
line1 = ax.plot(pressure, radius, linestyle ='dashed', color='red', label='R')
ax.set_xlabel(r"$p_0$ in Pa")
ax.set_ylabel("Radius in km")
ax2=ax.twinx()
line2 = ax2.plot(pressure, mass, color='blue', label='M')
ax2.set_ylabel(r"Mass in $M_0$")
ax.set_xscale('log')
ax.set_ylim(bottom=0)
ax2.set_ylim(bottom=0, top=0.8)
ax.set_xlim(left=1e29, right=1e39)

locmaj = matplotlib.ticker.LogLocator(base=10,numticks=6)
ax.xaxis.set_major_locator(locmaj)
locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

lines = line1+line2
labs = [l.get_label() for l in lines]
ax.legend(lines, labs, loc='best')
plt.savefig('pure_neutron_mass_radius_pressure.png')

plt.figure()
plt.plot(radius, mass)
plt.xlabel('R in km')
plt.ylabel(r'Mass in $M_0$')
plt.xlim(left=2.5, right=22.5)
plt.ylim(bottom=0.15, top=0.80)
plt.savefig('pure_neutron_mass_radius.png')

plt.show()

