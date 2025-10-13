"""
nd=1 Continuous Active Inference Thermostat

- Beliefs:    mu   (temperature belief)
              mu_p (belief about temperature rate)
- Observations: y (noisy thermometer), y_dot (noisy rate measurement)
- Generative model: g(mu)=mu, f(mu)=(mu_d - mu)/tau
- Free-energy minimised via gradient flows on mu, mu_p and actions u

Using simplified equations from https://arxiv.org/pdf/1909.12768
"""

import numpy as np

# -------------------------- SIMULATION SETTINGS --------------------------
# Numerics
dt = 1.0                # integration step (s) -- decrease if dynamics unstable
T_total = 1200.0        # total sim time (s)
n_steps = int(T_total / dt)

# Plant (real world) parameters - change to model different rooms / heaters
T_env = 15.0            # ambient temperature [degC]
T0 = 18.0               # initial room temperature [degC]
alpha = 0.1            # heat-loss rate [1/s]; larger = faster cooling
beta = 0.5             # heater efficiency [degC/s per action unit]

# Action limits (physical actuator bounds)
u_min = 0.0             # heater off
u_max = 1.5             # heater full power (normalized)

# Generative model / controller parameters - main knobs for AIF behavior
mu_d = 22.0             # desired temperature (setpoint) [degC] - change to what you want
tau = 1.0             # timescale of the prior dynamics f(mu) = (mu_d - mu)/tau

# Learning rates (gains)
k_mu = 0.05             # belief update gain (κ_μ) - increase => faster belief updates
k_a = 2.0              # action update gain (κ_a) - increase => more aggressive heating

# Precisions (inverse variances) - influence trust in sensors vs model
sensor_var = 0.1       # variance of thermometer noise (degC^2) -- smaller = more trust in sensor
sensor_dot_var = 0.1    # variance of derivative noise (degC/s)^2 -- increase to trust y' less
dynamics_var = 0.2      # variance in prior dynamics (how confident agent is in f) -- larger = less confident

# Action sensitivity approximation: how much action changes observed temp immediately
# For our simple plant, we approximate C ≈ beta (plant heating gain)
C = beta
C_dot = 0.0             # often zero (we don't assume a separate direct immediate effect on y')

# Logging / plotting
print_every = max(1, n_steps // 6)   # how often to print status; adjust for simulation length
save_plots = False                   # set True to save figures to disk

# -------------------------- Derived parameters / matrices --------------------------------
Sigma_y_inv = 1.0 / sensor_var
Sigma_yprime_inv = 1.0 / sensor_dot_var
Sigma_mu_inv = 1.0 / dynamics_var

# For clarity we also show 1x1 matrix equivalents (unused mathematically here)
P_y0 = np.array([[Sigma_y_inv]])      # precision for y
P_y1 = np.array([[Sigma_yprime_inv]]) # precision for y'
P_mu0 = np.array([[Sigma_mu_inv]])    # precision for dynamics error

# -------------------------- Allocate arrays --------------------------------------------
time = np.arange(n_steps + 1) * dt

T = np.zeros(n_steps + 1)         # real temperature
Tdot = np.zeros(n_steps + 1)      # real temperature derivative
mu = np.zeros(n_steps + 1)        # belief about temperature
mu_p = np.zeros(n_steps + 1)      # belief about temperature rate
u = np.zeros(n_steps + 1)         # action (heater power)
y = np.zeros(n_steps + 1)         # noisy sensor reading
y_dot = np.zeros(n_steps + 1)     # noisy rate reading

free_energy = np.zeros(n_steps + 1)
SPE = np.zeros(n_steps + 1)
SPE_dot = np.zeros(n_steps + 1)

# -------------------------- Initial conditions ----------------------------------------
T[0] = T0
mu[0] = T0    # start believing actual initial temperature (change if you want a mismatch)
mu_p[0] = 0.0
u[0] = 0.0

rng = np.random.default_rng(0)
sensor_noise_sd = np.sqrt(sensor_var)
sensor_dot_noise_sd = np.sqrt(sensor_dot_var)

# -------------------------- Helper: prior dynamics -------------------------------------
def f_prior(mu_val):
    """Prior dynamics f(mu) = (mu_d - mu)/tau
       This is the agent's expectation that mu should move toward mu_d.
       Change `tau` to make the prior expect faster/slower convergence.
    """
    return (mu_d - mu_val) / tau

# -------------------------- Simulation loop -------------------------------------------
print("Starting continuous AIF thermostat (nd=1) simulation")
print(f"dt={dt}s, total={T_total}s, steps={n_steps}")
print(f"Setpoint (mu_d)={mu_d}°C, tau={tau}s, k_mu={k_mu}, k_a={k_a}")
print(f"Precisions: Sigma_y_inv={Sigma_y_inv}, Sigma_yprime_inv={Sigma_yprime_inv}, Sigma_mu_inv={Sigma_mu_inv}")
print("Change variables at the top of this file to experiment.\n")

for i in range(n_steps):
    # 1) Real plant integration (forward Euler)
    #    dT/dt = -alpha*(T - T_env) + beta * u
    Tdot_i = -alpha * (T[i] - T_env) + beta * u[i]
    Tdot[i] = Tdot_i
    T[i + 1] = T[i] + dt * Tdot_i

    # 2) Sensors (noisy)
    y[i] = T[i] + rng.normal(0.0, sensor_noise_sd)
    y_dot[i] = Tdot_i + rng.normal(0.0, sensor_dot_noise_sd)

    # 3) Belief (perception) updates (nd = 1)
    #    mu_dot = mu' + k_mu*( Sigma_y_inv*(y - mu) - Sigma_mu_inv*(mu' - f(mu)) )
    #    mu'_dot = k_mu*( Sigma_yprime_inv*(y_dot - mu') - Sigma_mu_inv*(mu' - f(mu)) )
    f_mu = f_prior(mu[i])
    mu_dot = mu_p[i] + k_mu * (Sigma_y_inv * (y[i] - mu[i]) - Sigma_mu_inv * (mu_p[i] - f_mu))
    mu_dot_p = k_mu * (Sigma_yprime_inv * (y_dot[i] - mu_p[i]) - Sigma_mu_inv * (mu_p[i] - f_mu))

    # Euler integrate beliefs
    mu[i + 1] = mu[i] + dt * mu_dot
    mu_p[i + 1] = mu_p[i] + dt * mu_dot_p

    # numerical safety check: if values blow up, break and suggest lowering gains/precisions or dt
    if not (np.isfinite(mu[i + 1]) and np.isfinite(mu_p[i + 1])):
        print(f"Numerical instability detected at step {i} (t={i*dt}s).")
        print("Suggestions: reduce k_mu or k_a, reduce precisions (Sigma_*_inv), or reduce dt.")
        break

    # 4) Action update (control)
    #    dot_u = -k_a * ( C*Sigma_y_inv*(y - mu) + C_dot*Sigma_yprime_inv*(y_dot - mu') )
    dot_u = -k_a * (C * Sigma_y_inv * (y[i] - mu[i]) + C_dot * Sigma_yprime_inv * (y_dot[i] - mu_p[i]))
    u[i + 1] = u[i] + dt * dot_u
    # saturate actuator
    u[i + 1] = np.clip(u[i + 1], u_min, u_max)

    # 5) Diagnostics: free-energy and sensory prediction errors
    fe = 0.5 * (y[i] - mu[i]) ** 2 * Sigma_y_inv + \
         0.5 * (y_dot[i] - mu_p[i]) ** 2 * Sigma_yprime_inv + \
         0.5 * (mu_p[i] - f_mu) ** 2 * Sigma_mu_inv
    free_energy[i] = fe
    SPE[i] = 0.5 * (y[i] - mu[i]) ** 2 * Sigma_y_inv
    SPE_dot[i] = 0.5 * (y_dot[i] - mu_p[i]) ** 2 * Sigma_yprime_inv

    # Optional logging
    if (i % print_every) == 0:
        print(f"t={i*dt:6.1f}s | T={T[i]:5.2f}°C | mu={mu[i]:5.2f}°C | mu'={mu_p[i]:6.3f} | u={u[i]:5.3f} | F={free_energy[i]:6.3f}")

# final sensor reading fill-in for plotting convenience
y[-1] = T[-1] + rng.normal(0.0, sensor_noise_sd)
y_dot[-1] = Tdot[-1] + rng.normal(0.0, sensor_dot_noise_sd)
Tdot[-1] = -alpha * (T[-1] - T_env) + beta * u[-1]

print("\nSimulation complete.")
print(f"Final: T={T[-1]:.3f}°C, mu={mu[-1]:.3f}°C, u={u[-1]:.3f}")

# # -------------------------- Plots ----------------------------------------------------
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 4))
# plt.plot(time, T, label='True T (plant)')
# plt.plot(time, mu, label='Belief µ')
# plt.plot(time, mu * 0 + mu_d, '--', label='Setpoint µ_d')
# plt.xlabel('Time [s]')
# plt.ylabel('Temperature [°C]')
# plt.title('Temperature and belief over time')
# plt.legend()
# plt.grid(True)
# if save_plots:
#     plt.savefig('aif_thermostat_temp.png', dpi=200)
# plt.show()

# plt.figure(figsize=(10, 3))
# plt.plot(time, u, label='Heater action (u)')
# plt.xlabel('Time [s]')
# plt.ylabel('Action (normalized)')
# plt.title('Action (heater) over time')
# plt.legend()
# plt.grid(True)
# if save_plots:
#     plt.savefig('aif_thermostat_action.png', dpi=200)
# plt.show()

# plt.figure(figsize=(10, 3))
# plt.plot(time, free_energy, label='Free energy')
# plt.xlabel('Time [s]')
# plt.ylabel('Free energy (arb. units)')
# plt.title('Free energy over time')
# plt.legend()
# plt.grid(True)
# if save_plots:
#     plt.savefig('aif_thermostat_free_energy.png', dpi=200)
# plt.show()

# plt.figure(figsize=(10, 3))
# plt.plot(time, SPE, label='Sensory PE (pos)')
# plt.plot(time, SPE_dot, label='Sensory PE (vel)', linestyle=':')
# plt.xlabel('Time [s]')
# plt.ylabel('Weighted prediction error')
# plt.title('Sensory prediction errors')
# plt.legend()
# plt.grid(True)
# if save_plots:
#     plt.savefig('aif_thermostat_spe.png', dpi=200)
# plt.show()

# # -------------------------- Quick tuning notes (repeat inside script for convenience) ----------
# print("\n--- Quick tuning tips ---")
# print("If the simulation explodes (NaNs or inf):")
# print("  - reduce k_mu and k_a (gains), reduce Sigma_yprime_inv or Sigma_mu_inv, or reduce dt")
# print("If controller is too slow to heat:")
# print("  - increase k_a or beta (heater efficiency) or increase u_max")
# print("If belief oscillates or is noisy:")
# print("  - increase sensor_var (trust sensor less), reduce k_mu, or increase dynamics_var (weaker prior)")
# print("To model a more powerful heater: increase beta")
# print("To model a leakier room: increase alpha")
# print("To make the agent expect faster convergence: reduce tau")
# print("To disable velocity observation entirely: set Sigma_yprime_inv = 0 (and/or C_dot = 0)")
