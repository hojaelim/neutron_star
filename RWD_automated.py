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
eta = 2
crit_density = (m_n*eta*m_e**3*c**3)/(3*np.pi**2*hbar**3)
#p0 = 5.62 * 10**24 #in pascals
p_array= np.sort(np.geomspace(5.5555*10**22, 4.44444*10**24, 50))
p_array = p_array.tolist()

#Functions
def dm_rdr(r,p,m_r):
    return ((4*np.pi*r**2)/(M_s*c**2)) * ((p/K)**(1/gamma))

def dp_dr(r,p,m_r):
    return -(R_0*p**(1/gamma)*m_r)/(r**2*K**(1/gamma))

def r_fit(initial_pressure):
    rho_0_list = []
    for i in initial_pressure:
        rho_0 = (i/(K*c**(2*gamma)))**(1/gamma)
        rho_0_list.append(rho_0)

    r_fit_list = []
    for j in rho_0_list:
        r_fit = 0.5 * (3*np.pi)**0.5 * (6.89685) * ((hbar**1.5)/(c**0.5*G**0.5*m_e*m_n*eta))*(crit_density/j)**(1/3)
        r_fit_list.append(r_fit/1000)

    return r_fit_list

print('Calculating radius...')
r_fit_list = r_fit(p_array)


#Parameters
m0 = 0
r0 = 0.000001
r = r0
m_r = m0
dr = 1000
r_end = 60000000

r_list = []
m_r_list = []
p_list = []

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

print('Solving equations through RK4...')
for i in p_array:
    r=r0
    m_r=m0
    temp_r=[r0]
    temp_p=[i]
    temp_m=[m0]
    p=i
    while r <= r_end:
        r, p, m_r = RungeKuttaCoupled(r, p, m_r, dr, dp_dr, dm_rdr)

        temp_r.append(r/1000)
        temp_m.append(m_r)
        temp_p.append(p)

    r_list.append(temp_r)
    p_list.append(temp_p)
    m_r_list.append(temp_m)

p0_list=[]
mf_list=[]
rf_list=[]

print('Finding mass and fit radius for given pressure values...')
for i in range(len(p_list)):
    index = np.where(np.imag(p_list[i])==0)[0][-1] #Indicates the last point before values return imaginary numbers
    temp_p0, temp_mf, temp_rf = p_list[i][0], m_r_list[i][index], r_list[i][index]
    p0_list.append(temp_p0)
    mf_list.append(temp_mf)
    rf_list.append(temp_rf)


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

print('Plotting graph...')

#Plot
fig,ax = plt.subplots()
line1 = ax.plot(p0_list, rf_list, linestyle='dashed', color='red', label='Radius')
line2 = ax.plot(p0_list, r_fit_list, 'go', label = r'$\mathrm{R}_{\mathrm{FIT}}$', alpha=0.5, markersize=5)
ax.set_xlabel(r"$p_0$ in Pa")
ax.set_ylabel("Radius in km", color='red')
ax2=ax.twinx()
line3 = ax2.plot(p0_list, mf_list, label='Mass', color='blue')
ax2.set_ylabel(r"Mass in $M_0$ ", color='blue')
#ax.set_title("Relativistic White Dwarf")
ax.set_xscale('log')
ax.set_ylim(bottom=4000, top=20000)
ax2.set_ylim(bottom=1.40, top=1.44)
ax.set_xlim(left=5.555e22, right=4.444e24)

lines = line1+line2+line3
labs = [l.get_label() for l in lines]
ax.legend(lines, labs, loc=0)

plt.savefig('relativistic_white_dwarf_r_0.png')
plt.show()
