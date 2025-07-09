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
gamma = 5/3
K = (hbar**2)/(15*(np.pi**2)*m_e) * (((3*np.pi**2)/(2*m_n*c**2))**gamma)

p_array= np.sort(np.linspace(0.01*10**21, 4.5*10**21, 100))
p_array = p_array.tolist()
print(p_array)
#Parameters
m0 = 0
r0 = 0.000001
r = r0
m_r = m0
dr = 200
r_end = 17000000

r_list = []
m_r_list = []
p_list = []

#Functions
def dm_rdr(r,p,m_r):
    return ((4*np.pi*r**2)/(M_s*c**2)) * ((p/K)**(1/gamma)) 

def dp_dr(r,p,m_r):
    return -(R_0*p**(1/gamma)*m_r)/(r**2*K**(1/gamma))

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
    print(index)
    temp_p0, temp_mf, temp_rf = p_list[i][0], m_r_list[i][index], r_list[i][index]
    p0_list.append(temp_p0)
    mf_list.append(temp_mf)
    rf_list.append(temp_rf)

print(p0_list, mf_list, rf_list)

#Plot
fig,ax = plt.subplots()
ax.plot(p0_list, rf_list, linestyle='dashed', color='red', label='Radius')
ax.set_xlabel(r"$p_0$ in Pa")
ax.set_ylabel("Radius in km", color='red')
ax2=ax.twinx()
ax2.plot(p0_list, mf_list, label='Mass', color='blue')
ax2.set_ylabel(r"Mass in $M_0$ ", color='blue')
#ax.set_title("Non-relativistic White Dwarf")
ax.set_ylim(bottom=10000, top=17000)
ax2.set_ylim(bottom=0.10, top=0.55)
ax.set_xlim(left=0, right=4.5e21)
ax.ticklabel_format(style='scientific')
plt.savefig('non_relativistic_white_dwarf_r_0.png')
plt.show()
