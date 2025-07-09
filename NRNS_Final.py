import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import newton

#Constants
M_s = 1.98847 * 10**30 #Solar Mass
m_n = 1.67492749804 * 10**(-27) #Neutron Mass
hbar = 1.054571817 * 10**(-34) #Reduced Planck Const.
m_e = 9.1093837 * 10**(-31) #Electron Mass
G = 6.6743 * 10**(-11) #Gravitational Constant
c = 299792458 #Light Speed
R_0 = G*M_s/(c**2) #Schwarzchild Radius
p_array= np.sort(np.linspace(1.0*10**29, 1.0*10**39, 100)).tolist() #Generates initial pressure values
e0 = m_n**4*c**5/(np.pi**2*hbar**3) #Epsilon 0 value

#Parameters
m0 = 0 #Initial mass
r0 = 0.000001 #Initial radius
r = r0
m_r = m0
dr = 33 #Step size; must be lower for pressure values above 10^39.
r_end = 45000 #Maximum Operation Radius

#Functions
def epsilon(x): #Equation for energy density, in terms of fermi momentum
    return e0/8 * ((2*x**3+x)*(1+x**2)**(1/2)-np.arcsinh(x))

def pressure(x): #Equation for pressure, in terms of fermi momentum
    return e0/24 * ((2*x**3-3*x)*(1+x**2)**(1/2)+3*np.arcsinh(x))

def find_x(p): #Function to find the appropriate x value for a given pressure, using Newton Raphson.

    def f(x): #Condition to find x
        return pressure(x)-p

    def get_initial_guess(p):
        if p<=1.0e35:
            x_min = 0
            x_max = 1.04
            dx = 0.00001
            x_vals = np.arange(x_min, x_max, dx)
            p_vals = pressure(x_vals)
        
        elif p<=1.0e37:
            x_min = 1
            x_max = 3
            dx = 0.00001
            x_vals = np.arange(x_min, x_max, dx)
            p_vals = pressure(x_vals)
        
        elif p<=1.0e38:
            x_min = 3
            x_max = 5.3
            dx = 0.00001
            x_vals = np.arange(x_min, x_max, dx)
            p_vals = pressure(x_vals)

        elif p<=1.0e39:
            x_min = 5.3
            x_max = 9.3
            dx = 0.00001
            x_vals = np.arange(x_min, x_max, dx)
            p_vals = pressure(x_vals)
        
        elif p<=1.0e40:
            x_min = 9.3
            x_max = 16.45
            dx = 0.00001
            x_vals = np.arange(x_min, x_max, dx)
            p_vals = pressure(x_vals) 
        
        elif p<=1.0e41:
            x_min = 16.45
            x_max = 29.22
            dx = 0.00001
            x_vals = np.arange(x_min, x_max, dx)
            p_vals = pressure(x_vals)

        idx = np.argmin(np.abs(p_vals - p))
        return x_vals[idx]
    
    x_guess = get_initial_guess(p)
    x_root = newton(f, x_guess, maxiter=100000, tol=1e-6)
    return x_root

def epsilon_value(p): #Calculates the Epsilon value for a given x.
    x = find_x(p)
    value = epsilon(x)
    return value

def dmdr(r,p,m): #Mass equation
    return 4*np.pi*r**2*epsilon_value(p)/(c**2)

def dpdr(r,p,m): #Pressure equation, TOV
    return -((G*m*epsilon_value(p))/(c**2*r**2))*(1+(p/(epsilon_value(p))))*(1+(4*np.pi*r**3*p)/(m*c**2))*(1-((2*G*m)/(c**2*r)))**(-1)

def RungeKuttaCoupled(r, p, m, dr, dp_dr, dm_dr): #Runge Kutta Algorithm
    
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

r_list=[] #Radius list
m_list=[] #Mass list
p_list=[] #Pressure list

for j in p_array: #Execute Runge-Kutta and populate the lists
    r=r0
    m=r0
    temp_r=[r0]
    temp_p=[j]
    temp_m=[m0]

    p=j

    while r <= r_end:
        r, p, m = RungeKuttaCoupled(r, p, m, dr, dpdr, dmdr)
    
        temp_r.append(r/1000) 
        temp_m.append(m/M_s)
        temp_p.append(p)

    r_list.append(temp_r)
    p_list.append(temp_p)
    m_list.append(temp_m)

p0_list=[] #Initial Pressure list
mf_list=[] #Mass of star list
rf_list=[] #Radius of star list

for i in range(len(p_list)): #Find the radius and mass of star for given initial pressure
    index = np.where(np.array(p_list)[i]>0)[0][-1] #Indicates the last point before pressure return negative values
    temp_p0, temp_mf, temp_rf = p_list[i][0], m_list[i][index], r_list[i][index]
    p0_list.append(temp_p0)
    mf_list.append(temp_mf)
    rf_list.append(temp_rf)
    

final = [p0_list, mf_list, rf_list]
final = list(zip(*final)) #Transpose array

with open('output.txt', 'w') as f: #Save file of array
    for row in final:
        f.write(' '.join(map(str, row)) + '\n')

