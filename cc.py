# cc.py — compact cosmic-dust entry demo with plots
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------------
# Constants & particle params
# -----------------------------
G   = 6.67430e-11          # m^3 kg^-1 s^-2
M_E = 5.972e24             # kg
R_E = 6_371_000.0          # m (Earth mean radius)

rho0 = 1.3                 # kg/m^3, sea-level air density
H    = 7000.0              # m, scale height (simple exponential model)
mu   = 1.8e-5              # Pa·s, dynamic viscosity of air

# Particle (tweak these)
r_p      = 5e-4            # m, particle radius (1 mm diameter -> 5e-4 m)
rho_p    = 3000.0          # kg/m^3, particle bulk density (rocky)
Cd       = 2.0             # ~blunt/small sphere-ish
A        = np.pi * r_p**2  # m^2, cross-sectional area
m        = (4/3)*np.pi*r_p**3 * rho_p  # kg, mass
C_linear = 6*np.pi*mu*r_p  # Stokes linear drag coefficient

# -----------------------------
# Utility fields
# -----------------------------
def norm(x):
    return np.linalg.norm(x)

def air_density(r):
    """Exponential atmosphere: rho(r) = rho0 * exp(-(r - R_E)/H), floored at 0."""
    return rho0 * np.exp(-(r - R_E)/H)

def gravity_accel(X):
    r = norm(X)
    return -G*M_E * X / (r**3)

def drag_accel(X, V):
    """Linear + quadratic drag acceleration."""
    v = norm(V)
    if v == 0.0:
        return np.zeros(3)
    r   = norm(X)
    rho = air_density(r)
    # Linear (Stokes) term + quadratic term
    F_lin  = -C_linear * V
    F_quad = -0.5 * Cd * rho * A * v * V
    return (F_lin + F_quad) / m

# -----------------------------
# ODE system: d/dt [X, V] = [V, g + drag]
# -----------------------------
def f(t, y):
    X = y[0:3]
    V = y[3:6]
    dX = V
    dV = gravity_accel(X) + drag_accel(X, V)
    return np.hstack([dX, dV])

# Stop when we hit the ground
def hit_ground(t, y):
    return norm(y[0:3]) - R_E
hit_ground.terminal  = True
hit_ground.direction = -1  # crossing downward

# -----------------------------
# Initial conditions
# -----------------------------
h0   = 120_000.0  # m, start 120 km altitude
X0   = np.array([R_E + h0, 0.0, 0.0])
V0   = np.array([0.0, -3000.0, 0.0])  # 3 km/s, mostly tangential/downward
y0   = np.hstack([X0, V0])

# Integration controls—tweak for speed/accuracy
t_final  = 3000.0  # seconds
rtol     = 1e-6
atol     = 1e-9
max_step = 0.5     # small fixed-ish max step helps event detection & stability

sol = solve_ivp(
    f,
    (0.0, t_final),
    y0,
    events=hit_ground,
    rtol=rtol,
    atol=atol,
    max_step=max_step,
    dense_output=False
)

t  = sol.t
Y  = sol.y.T
X  = Y[:, 0:3]
V  = Y[:, 3:6]

alt_km = (np.linalg.norm(X, axis=1) - R_E)/1000.0
speed  = np.linalg.norm(V, axis=1)

print(f"Status: {sol.message}")
if sol.t_events[0].size > 0:
    print(f"Hit ground at t = {sol.t_events[0][0]:.2f} s")

# -----------------------------
# Plots
# -----------------------------
plt.figure()
plt.plot(t, alt_km)
plt.xlabel("Time (s)")
plt.ylabel("Altitude (km)")
plt.title("Altitude vs Time")

plt.figure()
plt.plot(t, speed/1000.0)
plt.xlabel("Time (s)")
plt.ylabel("Speed (km/s)")
plt.title("Speed vs Time")

# Simple ground-track projection in x–y (just for intuition)
plt.figure()
plt.plot(X[:,0]/1000.0, X[:,1]/1000.0)
plt.xlabel("x (km)")
plt.ylabel("y (km)")
plt.axis("equal")
plt.title("Ground-Track (x–y)")

plt.show()
