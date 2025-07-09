import numpy as np
import matplotlib.pyplot as plt

#Constants
M_s = 1.98847 * 10**30 #Solar Mass
m_n = 1.67492749804 * 10**(-27) #Neutron Mass
hbar = 1.054571817 * 10**(-34) #Reduced Planck Const.
m_e = 9.1093837 * 10**(-31) #Electron Mass
G = 6.6743 * 10**(-11) #Gravitational Constant
c = 299792458
R_0 = G*M_s/(c**2) #Schwarzchild Radius
gamma = 5/3
K = (hbar**2)/(15*(np.pi**2)*m_n) * (((3*np.pi**2)/(m_n*c**2))**gamma)
p_array= np.sort(np.geomspace(1.0*10**30, 1.0*10**33, 1000)).tolist() #Generates initial pressure values

#Parameters
m0 = 0 #Initial mass
r0 = 0.000001 #Initial radius
r = r0
m_r = m0
dr = 10 #Step size
r_end = 34000 #Operation limit

#Functions
def dm_rdr(r,p,m_r):
    return ((4*np.pi*r**2)/(M_s*c**2)) * ((p/K)**(1/gamma)) 

def dp_dr(r,p,m_r):
    return -(R_0*p**(1/gamma)*m_r)/(r**2*K**(1/gamma))

def tov_m(r,p,m):
    rho=(p/(K*c**(2*gamma)))**(1/gamma)
    return 4*np.pi*r**2*rho

def tov_p(r,p,m):
    rho=(p/(K*c**(2*gamma)))**(1/gamma)
    return -((G*m*rho)/(r**2))*(1+(p/(rho*c**2)))*(1+(4*np.pi*r**3*p)/(m*c**2))*(1-((2*G*m)/(c**2*r)))**(-1)

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

#Newtonian
r_list = []
m_r_list = []
p_list = []

print('Evaluating RK4 for Newtonian...')
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

for i in range(len(p_list)):
    index = np.where(np.imag(p_list[i])==0)[0][-1] #Indicates the last point before values return imaginary numbers
    temp_p0, temp_mf, temp_rf = p_list[i][0], m_r_list[i][index], r_list[i][index]
    p0_list.append(temp_p0)
    mf_list.append(temp_mf)
    rf_list.append(temp_rf)

#TOV
TOV_r_list=[]
TOV_m_list=[]
TOV_p_list=[]

print('Evaluating RK4 for TOV...')
for j in p_array:
    r=r0
    m=r0
    temp_r=[r0]
    temp_p=[j]
    temp_m=[m0]
    p=j
    while r <= r_end:
        r, p, m = RungeKuttaCoupled(r, p, m, dr, tov_p, tov_m)
    
        temp_r.append(r/1000) 
        temp_m.append(m/M_s)
        temp_p.append(p)

    TOV_r_list.append(temp_r)
    TOV_p_list.append(temp_p)
    TOV_m_list.append(temp_m)

TOV_p0_list=[]
TOV_mf_list=[]
TOV_rf_list=[]

for i in range(len(TOV_p_list)):
    index = np.where(np.imag(TOV_p_list[i])==0)[0][-1] #Indicates the last point before values return imaginary numbers
    temp_p0, temp_mf, temp_rf = TOV_p_list[i][0], TOV_m_list[i][index], TOV_r_list[i][index]
    TOV_p0_list.append(temp_p0)
    TOV_mf_list.append(temp_mf)
    TOV_rf_list.append(temp_rf)

#Plot
fig,ax = plt.subplots()
line1 = ax.plot(p0_list, rf_list, linestyle='dashed', color='red', label='R Newton')
line2 = ax.plot(TOV_p0_list, TOV_rf_list, linestyle ='dotted', color='green', label='R TOV')
ax.set_xlabel(r"$p_0$ in Pa")
ax.set_ylabel("Radius in km")
ax2=ax.twinx()
line3 = ax2.plot(p0_list, mf_list, label='M Newton', color='blue')
line4 = ax2.plot(TOV_p0_list, TOV_mf_list, color='black', label='M TOV')
ax2.set_ylabel(r"Mass in $M_0$ ")
#ax.set_title("Non-relativistic Neutron Star")
ax.set_ylim(bottom=15, top=33)
ax2.set_ylim(bottom=0.0, top=0.55)
ax.set_xscale('log')
ax.set_xlim(left=2e30, right=4e32)

lines = line1+line2+line3+line4
labs = [l.get_label() for l in lines]
ax.legend(lines, labs, loc='upper center')

plt.savefig('non_relativistic_neutron_star_r_0.png', dpi=600)
plt.show()
