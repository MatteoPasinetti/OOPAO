# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 18:14:12 2026

@author: mpasinetti
"""

# %% =========================================================================
# ----------------------- 8. VALIDATION: ELEVATION IMPACT --------------------
# ============================================================================
"""
This section validates how OOPAO handles the target's elevation angle.
When observing away from the zenith (90°), the light travels through a thicker 
slice of the atmosphere. This is quantified by the 'Airmass' (~ 1/sin(elevation)).

Physically, a higher airmass means:
1. Stronger turbulence: r0 decreases as (Airmass)^(-3/5)
2. Stronger scintillation: Rytov variance increases as (Airmass)^(11/6)
3. Higher effective altitudes: The physical distance to the layers increases 
   as (Airmass).

OOPAO automatically handles these scaling laws when you update `atm.elevation`.
We will sweep from Zenith (90°) down to low elevation (15°) and compare the 
internal OOPAO values against the theoretical analytical formulas.
"""
import time
import numpy as np
import matplotlib.pyplot as plt

print("\n--- 8. Physics Validation: Elevation Impact ---")

# Define elevation angles from low elevation (15°) to Zenith (90°)
elevations_deg = np.array([15, 30, 45, 60, 75, 90])

# Lists to store simulation results
simulated_rytov = []
simulated_r0 = []
simulated_altitudes = []
sigma_sim = []
sigma_theo = []

# Lists to store theoretical analytical curves
theoretical_rytov_scaling = []
theoretical_r0 = []
theoretical_altitudes = []

# Store baseline values at Zenith (Airmass = 1)
base_altitudes = np.array(atm.altitude) 
base_r0 = atm.r0_500 if hasattr(atm, 'r0_500') else atm.r0 

# ============================================================================
# --- 1. SWEEPING THROUGH ELEVATION ANGLES ---
# ============================================================================
for elev in elevations_deg:
    # Calculate Airmass for theoretical predictions
    airmass = 1.0 / np.sin(np.deg2rad(elev))
    
    print(f"Updating Atmosphere for Elevation: {elev:2d}° | Airmass: {airmass:.2f}", end='\r')
    
    # --- UPDATE ATMOSPHERE ---
    # Changing the elevation automatically recalculates r0, altitudes, and phase screens
    atm.elevation = elev 
    
    # --- Extract OOPAO Internal Values ---
    simulated_r0.append(atm.r0)
    simulated_altitudes.append(atm.altitude) 

    # --- Compute Theoretical Analytical Values ---
    theoretical_r0.append(base_r0 * (airmass)**(-3.0/5.0))
    theoretical_altitudes.append(base_altitudes * airmass)

    # --- Simulate frames to accumulate Rytov Variance ---
    # Scintillation is a statistical process, we need to average over a few frames
    rytov_frames = []
    for _ in range(10):
        atm.update()
        ngs ** atm * tel
        rytov_frames.append(ngs.rytov_variance)

    simulated_rytov.append(np.mean(rytov_frames))
    theoretical_rytov_scaling.append(airmass**(11.0/6.0))
    
    # OOPAO's internal theoretical predictors
    sigma_sim.append(ngs.rytov_variance)
    sigma_theo.append(ngs.get_theoretical_rytov(atm))

print("\nElevation sweep completed. Generating validation plots...")

# ============================================================================
# --- 2. DATA PREPARATION FOR PLOTTING ---
# ============================================================================
simulated_rytov = np.array(simulated_rytov)
theoretical_rytov_scaling = np.array(theoretical_rytov_scaling) 

# Normalize the theoretical Airmass scaling curve to match the simulation at Zenith
if simulated_rytov[-1] > 0:
    theoretical_rytov_scaling = theoretical_rytov_scaling * (simulated_rytov[-1] / theoretical_rytov_scaling[-1])

simulated_r0 = np.array(simulated_r0)
theoretical_r0 = np.array(theoretical_r0)

# Transpose altitude arrays so each row corresponds to a specific atmospheric layer
simulated_altitudes = np.array(simulated_altitudes).T
theoretical_altitudes_plot = np.array(theoretical_altitudes).T

# ============================================================================
# --- 3. PLOTTING THE RESULTS ---
# ============================================================================
# Global plot configuration for a modern look
plt.rcParams.update({
    'font.size': 12,             
    'axes.titlesize': 15,        
    'axes.labelsize': 13,        
    'axes.labelweight': 'bold',  
    'legend.fontsize': 11,       
    'figure.titlesize': 18
})

fig, axs = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

# --- Plot 1: Rytov Variance ---
axs[0].plot(elevations_deg, simulated_rytov, marker='o', color='#1f77b4', markersize=7, linewidth=2.5, label='Simulation mean (10 frames)')
axs[0].plot(elevations_deg, sigma_sim, marker='s', color='#ff7f0e', markersize=6, linewidth=2, alpha=0.8, label='OOPAO internal (Last frame)') 
axs[0].plot(elevations_deg, theoretical_rytov_scaling, linestyle='--', color='#2ca02c', linewidth=2.5, label=r'Theory Scaling (Airmass$^{11/6}$)')
axs[0].plot(elevations_deg, sigma_theo, linestyle=':', color='#d62728', linewidth=2.5, label=r'Theory Formula') 

axs[0].set_xlabel('Elevation Angle [Degrees]')
axs[0].set_ylabel(r'Rytov Variance $\sigma_R^2$')
axs[0].set_title('Scintillation vs Elevation') 
axs[0].grid(True, linestyle='--', alpha=0.5, color='gray') 
axs[0].legend(frameon=True, edgecolor='black', shadow=True)

# --- Plot 2: r0 Evolution ---
# Convert to cm for better readability on the Y-axis
axs[1].plot(elevations_deg, simulated_r0 * 100, marker='o', color='#17becf', markersize=7, linewidth=2.5, label='OOPAO Internal')
axs[1].plot(elevations_deg, theoretical_r0 * 100, linestyle='--', color='#d62728', linewidth=2.5, label=r'Theory (Airmass$^{-3/5}$)')

axs[1].set_xlabel('Elevation Angle [Degrees]')
axs[1].set_ylabel('Effective $r_0$ [cm]') 
axs[1].set_title('$r_0$ vs Elevation')
axs[1].grid(True, linestyle='--', alpha=0.5, color='gray')
axs[1].legend(frameon=True, edgecolor='black', shadow=True)

# --- Plot 3: Layer Altitudes ---
cmap = plt.get_cmap('tab10') 

for i in range(theoretical_altitudes_plot.shape[0]):
    c = cmap(i % 10) 
    
    # 1. Plot Theory first (zorder=1) as a thick, semi-transparent background highlighter
    axs[2].plot(elevations_deg, theoretical_altitudes_plot[i] / 1000, color=c, 
                linestyle='--', linewidth=3.5, alpha=0.5, zorder=1)

    # 2. Plot Simulation on top (zorder=2) with sharp markers and thin lines
    axs[2].plot(elevations_deg, simulated_altitudes[i] / 1000, marker='o', color=c, 
                linestyle='-', linewidth=1.5, markersize=6, zorder=2, label=f'Layer {i+1} (OOPAO)')

# Dummy line for the Theory legend entry
axs[2].plot([], [], color='black', linestyle='--', linewidth=3.5, alpha=0.5, label='Theory (Airmass)')

axs[2].set_xlabel('Elevation Angle [Degrees]')
axs[2].set_ylabel('Effective Altitude [km]')
axs[2].set_title('Layer Altitudes vs Elevation')
axs[2].grid(True, linestyle='--', alpha=0.5, color='gray')
axs[2].legend(loc='best', frameon=True, edgecolor='black', shadow=True)

plt.tight_layout() 
plt.show()