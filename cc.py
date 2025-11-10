import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

R_E=6371*(10**3)
G=6.6743*(10**-11)
M_E=5.97*(10**24)
X_0=(10**8,0,0)#starting position in cartesian coordinates origin is centre of the earth
V_0=(-10**3,0,0)#starting velocity
area=np.pi*10**-6#area of particle(assume its a sphere so we meen projected circular area)
C=0.47 #
M=10**-8#mass of particle

mu_a=1.8*(10**-5)#this is a function of temperatuer so may need to make it work for distance
def R(X):
    return(np.sqrt((X[0]**2)+(X[1]**2)+(X[2]**2)))
def density(r):
    return(1.3*np.exp (-(r-R_E )/ 7000 ))#isothermal model from space academy.net Ill try to gain better model or we assume things to get better model by hand
def G_acceleration(X):
    return(-G*M_E*X/(R(X)**3)) #its X3 at bottom as we times by X for direction of force (- as force pushes towards earth)
def air_resistance(X,V,A):
    mag=(6*np.pi*mu_a*np.sqrt(A)*np.sqrt(np.sum(V**2))+density(R(X))*np.sum(V**2)*C*A/2)#magnitude of air resistance I believe this equation should work well for cosmic dust around 1mm but not for ones which are like one molecule
    return(-V*mag/(np.sqrt(np.sum(V**2))))#times normalised vector in direction of air resistance(- as goes other way)
        

def  dF(t,S):#this is to describe differential equation for solve_ivp
    X=S[:3]
    V=S[3:]
    Xp=np.array(V)
    Vp=np.array((air_resistance(X,V,area)/M)+G_acceleration(X))
    return(np.append(Xp,Vp))
    
def over(t,S):
    return(R(np.array([S[0],S[1],S[2]]))-R_E)
over.terminal=True
t=np.linspace(0,80000,40000)
S_0=np.concatenate((X_0,V_0))
trajectory=solve_ivp(dF,t_span=(0,80000),y0=S_0,t_eval=t,events=over)
X_path=trajectory.y[0]
Y_path=trajectory.y[1]
Z_path=trajectory.y[2]
time=trajectory.t
R_Sol=R((X_path,Y_path,Z_path))

plt.plot(time,R_Sol-R_E)
plt.xlabel("time/s")
plt.ylabel("Height above surface")
plt.show()

