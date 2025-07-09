import numpy as np
import matplotlib.pyplot as plt

#Constants
M_s = 1.98847 * 10**30
m_n = 1.67492749804 * 10**(-27)
hbar = 1.054571817 * 10**(-34)
m_e = 9.1093837 * 10**(-31)
G = 6.6743 * 10**(-11)
c = 299792458
R_0 = G*M_s/(c**2) 
gamma = 4/3
K = (hbar*c)/(12*np.pi**2) * (((3*np.pi**2)/(2*m_n*c**2))**gamma)
p0 = 5.62 * 10**24 #in pascals

#Functions
def dm_rdr(r,p,m_r):
    return ((4*np.pi*r**2)/(M_s*c**2)) * ((p/K)**(1/gamma)) 

def dp_dr(r,p,m_r):
    return -(R_0*p**(1/gamma)*m_r)/(r**2*K**(1/gamma))

#Parameters
m0 = 0
r0 = 0.000001
r = r0
m_r = m0
p = p0
dr = 100
r_end = 6000000

r_list = [r0]
m_r_list = [m0]
p_list = [p0]

#RK4 Algorithm
def RungeKuttaCoupled(r, p, m, dr, dp_dr, dm_dr):
    
    k1 = dr*dp_dr(r, p, m)
    h1 = dr*dm_dr(r, p, m)
    k2 = dr*dp_dr(r+dr/2., p+k1/2., m+h1/2.)
    h2 = dr*dm_dr(r+dr/2., p+k1/2., m+h1/2.)
    k3 = dr*dp_dr(r+dr/2., p+k2/2., m+h2/2.)
    h3 = dr*dm_dr(r+dr/2., p+k2/2., m+h2/2.)
    k4 = dr*dp_dr(r+dr, p+k3, m+h3)
    h4 = dr*dm_dr(r+dr, p+k3, m+h3)

    p = p + 1./6.*(k1+2*k2+2*k3+k4)
    m = m + 1./6.*(h1+2*h2+2*h3+h4)
    r = r + dr
    
    return r, p, m

while r <= r_end:
    
    r, p, m_r = RungeKuttaCoupled(r, p, m_r, dr, dp_dr, dm_rdr)
    
    r_list.append(r/1000) #in km
    m_r_list.append(m_r)
    p_list.append(p)

"""
#Newton-Raphson Algorithm
i= np.where(np.array(p_list) > 0)[0][-1]  #Find a starting point for the Newton-Raphson method

def next_r_function(previous_r):
    index = np.where(np.array(r_list) == previous_r)[0][0]
    return previous_r - p_list[index] / dp_dr(previous_r, p_list[index], m_r_list[index])

def newton_raphson(r_start=r_list[i], tolerance=0.01,
                   next_r=next_r_function):
    difference = 1
    counter = 0
    r_root = r_start

    while difference > tolerance:

        counter += 1

        r_test = r_root
        r_root = next_r(r_root)

        difference = abs(r_test - r_root)

    return r_root

r_solution = np.real(newton_raphson())

#print(r_list[i], np.real(m_r_list[i]), np.real(p_list[i]))
"""
j = np.where(np.imag(p_list)==0)[0][-1] #Indicates the last point before values return imaginary numbers
print(p_list[j], r_list[j], m_r_list[j])

#Plot
fig,ax = plt.subplots()
ax.plot(r_list, p_list, linestyle='dashed', color='red', label='Pressure')
ax.set_xlabel("r in km")
ax.set_ylabel("Pressure in Pa", color='red')
ax2=ax.twinx()
ax2.plot(r_list, m_r_list, label='Mass', color='blue')
ax2.set_ylabel(r"Mass in $M_0$ ", color='blue')
#ax.set_title("Relativistic White Dwarf")
ax.set_xlim(left=0)
ax.ticklabel_format(style='scientific')
plt.savefig('relativistic_white_dwarf.png')
plt.show()
