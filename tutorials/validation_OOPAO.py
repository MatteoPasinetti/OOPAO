# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:54:02 2026

@author: mpasinetti
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 16:04:55 2026

@author: mpasinetti

Tutorial Description — Scintillation in OOPAO

This tutorial provides a focused walkthrough of modeling Scintillation in OOPAO, 
illustrating an end-to-end setup to correctly configure and simulate amplitude fluctuations within a full optical chain.

Starting from the initialization of a telescope and source, the script introduces the essential Fresnel padding checks required for accurate physical optics. 
It explores atmospheric propagation in detail, showing how to switch between and compare the geometric and diffractive regimes.

The tutorial highlights practical challenges and specific configurations needed for scintillation:
    - applying the proper scaling of Deformable Mirrors on padded grids
    - evaluating WFS behavior (both Pyramid and Shack-Hartmann)
    - comparing optical responses with and without scintillation effects present in the beam.

Finally, the simulation setup undergoes strict scientific validation, where users can analyze the Power Spectral Density (DSP) and verify the analytical scaling laws with respect to the target's elevation.

"""


import numpy as np
import matplotlib.pyplot as plt

from OOPAO.Atmosphere import Atmosphere
from OOPAO.Detector import Detector
from OOPAO.Pyramid import Pyramid
from OOPAO.ShackHartmann import ShackHartmann


# =========================================================================
# ----------------------- 0. SIMULATION PARAMETERS ------------------------
# =========================================================================
plt.ion()
n_subaperture = 16
res_factor = 8

# %%-----------------------     TELESCOPE   ----------------------------------
from OOPAO.Telescope import Telescope
print("\n--- 1. Initializing Telescope & Sources ---")
tel = Telescope(resolution           = res_factor * n_subaperture, 
                diameter             = 0.6,                       
                samplingTime         = 1/1000,                     
                centralObstruction   = 0,                     
                display_optical_path = False,                      
                fov                  = 2 )           
              
plt.figure()
plt.imshow(tel.pupil)
plt.title("Non padded telescope")
#%% -----------------------     NGS   ----------------------------------
from OOPAO.Source import Source

ngs = Source(optBand     = 'V', magnitude = 1, coordinates = [0,0])
src = Source(optBand     = 'K',  magnitude = 1, coordinates = [1,0])

#%% ------------------- FRESNEL PROPAGATION: PADDING VERIFICATION -------------------
"""
Since scintillation modeling relies on Fresnel propagation, grid padding must be 
thoroughly validated to ensure a physically accurate optical propagation. 

This cell demonstrates how to verify four essential sampling criteria against the 
simulation's physical parameters. These criteria are extracted from the reference 
book "Numerical Simulation of Optical Wave Propagation" by Jason D. Schmidt.
"""
print("\n--- 2. Verifying Fresnel Sampling Grid ---")
from OOPAO.tools.tools import compute_fresnel_padding
# Simulation parameter 
elevation = 30  # elevation angle in deg
airmass = 1.0 / np.sin(np.deg2rad(elevation)) # Airmass calculation (1 / sin(elevation)
r0_500 = 0.06 # Zenith reference values
altitudes = [0, 1000, 2000]
wavelengths = [ngs.wavelength, src.wavelength]
# Calculate r0 for each wavelength
r0_wvl_list = [r0_500 * (wvl / 500e-9)**(6/5) for wvl in wavelengths]

z_max = max(altitudes) if altitudes else 0
alts_sorted = sorted(altitudes + [0.0], reverse=True)
max_layer_step = max([alts_sorted[i] - alts_sorted[i+1] for i in range(len(alts_sorted)-1)]) if len(alts_sorted) > 1 else 0

# Function to calculate the padding needed for the simulation
diagnostics = compute_fresnel_padding(
    D_tel=tel.D, resolution=tel.resolution, wavelengths=wavelengths,
    z_max=z_max, r0_wvl_list=r0_wvl_list, max_layer_step=max_layer_step,
    res_factor=res_factor 
)
max_N_required = max([diag['N_quantized'] for diag in diagnostics.values()])

if max_N_required > tel.resolution:
    # Calculate the total difference
    diff = max_N_required - tel.resolution
    
    # Ensure the difference is even so the padding remains symmetric
    if diff % 2 != 0:
        diff += 1
        
    pad_val = diff // 2
    print(f"=> Automatically padding the telescope by {pad_val} pixels per side...")
    tel.pad(padding_values=pad_val)
    tel.print_properties()
    # Final safety check
    if tel.resolution % 2 != 0:
        print("Safety check: forcing even resolution")
        tel.pad(padding_values=1) # Add 1 px to force an even number if needed
else:
    print("=> Grid is robust. No padding required.")

# The padding must be done first as it will force the simualation grid size
ngs ** tel
src ** tel

plt.figure()
plt.imshow(tel.pupil)
plt.title("Padded telescope")
# %%


from OOPAO.Atmosphere import Atmosphere

# create the Atmosphere object   
atm = Atmosphere(telescope     = tel,                                          
                 r0            = 0.06,                            
                 L0            = 20,                                
                 fractionalR0  = [0.45, 0.2, 0.35],                
                 windSpeed     = [10   ,12   ,11],                
                 windDirection = [0, 72, 288],             
                 altitude      = altitudes)                      
atm.initializeAtmosphere(tel)
atm.angular_spectrum_propagation = True
atm.unwrap_diffractive_phase = True

# %% ------------------- SIMULATING A SPECIFIC RYTOV REGIME -------------------
"""
Direct Rytov Variance Scaling in OOPAO:
Once the Atmosphere object is defined (with fractional weights and altitudes), 
you don't need to manually recompute the Cn2 values to change the turbulence 
strength. 

By simply setting `atm.rytov_var = target`, OOPAO automatically:
  1. Scales the entire Cn2 profile proportionally to reach the requested Rytov variance.
  2. Preserves the fractional distribution across all atmospheric layers.
  3. Re-initializes the physical phase screens to reflect this new strength.

However, as the rytov variance increase the r0 will decrease meaning needing a bigger padding 
to ensure a physical simulation.

This section runs a single sweep from the weak fluctuation regime to deep 
saturation, extracting metrics to plot 3 distinct physical validations.
"""
print("\n--- Starting Atmospheric Scaling & Saturation Sweep ---")

# We sweep the Rytov Variance from 0.01 up to 16.0
# (Rytov = 16 corresponds to a log-amplitude variance of 4.0)
target_rytov_values = np.linspace(0.01, 16.0, 30)
atm.rytov_wvl = ngs.wavelength

# Lists to store the extracted metrics
x_theo_chi_R    = [] # Theoretical log-amplitude variance (Sigma_chi_R^2)
sim_scint_index = [] # Simulated Scintillation Index (Sigma_I^2)
sim_4_var_chi   = [] # Simulated 4 * Log-amplitude variance
r0_simulated    = [] # Simulated Fried parameter

# Pre-compute pupil mask to avoid edge artifacts in intensity calculations
mask = tel.pupil > 0 if hasattr(tel, 'pupil') and tel.pupil is not None else slice(None)

for i, target in enumerate(target_rytov_values):
    print(f"Step {i+1:02d}/{len(target_rytov_values)} | Target Rytov: {target:.3f}", end='\r')
    
    # 1. Update atmosphere (scales Cn2 and regenerates screens) and propagate
    atm.rytov_var = target
    ngs ** atm * tel
    
    # 2. Retrieve theoretical Rytov and r0
    theo_rytov = float(ngs.get_theoretical_rytov(atm))
    theo_chi_R = theo_rytov / 4.0
    
    # 3. Retrieve intensities on the valid pupil
    I = ngs.scintillation[mask]
    I_mean = np.mean(I)
    
    # --- METRICS COMPUTATION ---
    # Scintillation Index (Sigma_I^2)
    scint_idx = np.var(I) / (I_mean**2)
    
    # 4 * Log-Amplitude Variance (4 * Sigma_chi^2)
    chi = 0.5 * np.log(np.maximum(I / I_mean, 1e-12))
    var_chi_4 = 4.0 * np.var(chi)
    
    # Store data
    x_theo_chi_R.append(theo_chi_R)
    sim_scint_index.append(scint_idx)
    sim_4_var_chi.append(var_chi_4)
    r0_simulated.append(atm.r0)

print("\nSweep completed! Generating physical validation plots...")

# Convert to arrays for mathematical operations
x_arr = np.array(x_theo_chi_R)
theo_rytov_arr = x_arr * 4.0

# ============================================================================
# --- PLOT 1: WEAK FLUCTUATION REGIME (Linear relation)
# ============================================================================
"""
Plot 1 focuses on the weak turbulence regime (Rytov < ~0.5).
It demonstrates that for weak perturbations, the simulated variance linearly 
follows the theoretical Rytov approximation before saturation begins to appear.
"""
plt.figure(figsize=(8, 6))

plt.plot(theo_rytov_arr, theo_rytov_arr, 'k--', label='Ideal (Theoretical = Simulated)')
plt.plot(theo_rytov_arr, sim_4_var_chi, 'o-', color='tab:blue', linewidth=2, label='Simulated (OOPAO)')

plt.xlim(0, 2.0)
plt.ylim(0, 2.0)
plt.xlabel(r'Theoretical Rytov Variance $\sigma_R^2$', fontsize=12)
plt.ylabel(r'Simulated Variance $4\sigma_\chi^2$', fontsize=12)
plt.title('Scintillation simulation vs theory', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()

# ============================================================================
# --- PLOT 2: DEEP SATURATION (Vedrenne 2008 Reproduction)
# ============================================================================
"""
Plot 2 reproduces the classic saturation behavior (e.g., Vedrenne 2008).
In strong fluctuation regimes, the Rytov approximation breaks down. Multiple 
scattering causes the Scintillation Index to peak and asymptote towards 1, 
while the log-amplitude variance heavily saturates.
"""
fig, ax = plt.subplots(figsize=(8, 6))

# Theoretical linear boundary
ax.plot(x_arr, 4 * x_arr, '-.', color='red', linewidth=2, label=r'$4\sigma_{\chi_R}^2$ (Theory)')
# Simulated log-amplitude variance
ax.plot(x_arr, sim_4_var_chi, '--', color='red', linewidth=2, label=r'$4\sigma_\chi^2$ (Simulated)')
# Simulated Scintillation Index
ax.plot(x_arr, sim_scint_index, '-', color='red', linewidth=2, label=r'$\sigma_I^2$ (Scintillation Index)')

ax.set_xlim(0, 4.0)
ax.set_ylim(0, 2.6)
ax.set_xlabel(r'$\sigma_{\chi_R}^2$ (Log-amplitude variance)', fontsize=12)
ax.set_ylabel('Scintillation Index [-]', fontsize=12)
ax.set_title('Scintillation Saturation (Vedrenne 2008 equivalent)', fontsize=14, fontweight='bold')
ax.grid(True, which='both', linestyle='--', alpha=0.5)
ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
plt.tight_layout()
plt.show()

# ============================================================================
# --- PLOT 3: r0 EVOLUTION vs RYTOV VARIANCE
# ============================================================================
"""
Plot 3 validates the physical coupling between Rytov variance and Fried parameter.
Since both depend on the integrated Cn2 profile:
- Rytov is proportional to integral(Cn2)
- r0 is proportional to integral(Cn2)^(-3/5)
Therefore, scaling the atmosphere yields: r0_new = r0_ref * (Rytov_new / Rytov_ref)^(-3/5)
"""
plt.figure(figsize=(8, 6))

r0_arr = np.array(r0_simulated)

# Compute theoretical r0 scaling using the first data point as reference
r0_ref = r0_arr[0]
rytov_ref = theo_rytov_arr[0]
r0_theoretical = r0_ref * (theo_rytov_arr / rytov_ref)**(-3/5)

plt.plot(theo_rytov_arr, r0_theoretical, 'k--', linewidth=2, label='Theoretical $r_0 \propto (\sigma_R^2)^{-3/5}$')
plt.plot(theo_rytov_arr, r0_arr, 'o', color='tab:orange', markersize=6, alpha=0.8, label='Simulated $r_0$ (OOPAO)')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Theoretical Rytov Variance $\sigma_R^2$', fontsize=12)
plt.ylabel(r'Fried Parameter $r_0$ [m]', fontsize=12)
plt.title(r'Physical Coupling: $r_0$ vs Rytov Variance', fontsize=14, fontweight='bold')
plt.grid(True, which='both', linestyle=':', alpha=0.7)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()


# %% ----------------------- 6. PHYSICS VALIDATION: PSD -------------------------
"""
This section validates the physical accuracy of the simulated wavefronts 
by comparing their Power Spectral Densities (PSD) against theoretical models.

It also highlights a critical trap in physical optics simulations:
- If you compute the PSD (or project onto a modal basis like Zernikes/KL) using 
  the raw diffractive phase (which is wrapped modulo 2π), the artificial 2π 
  phase jumps will be interpreted as high-frequency noise. This completely 
  destroys the low-frequency information of the turbulence.
- Therefore, to perform accurate wavefront analysis or close an AO loop with a 
  modal reconstructor, you MUST use either the geometric phase or the properly 
  unwrapped diffractive phase.
"""

import numpy as np
import matplotlib.pyplot as plt

print("\n--- 6. Physics Validation: Power Spectral Density (PSD) ---")
# --- 0. Compute PSD for the GEOMETRIC Phase ---
print("Computing PSD for geometric phase...")
atm.scintillation = False
atm.unwrap_diffractive_phase = False
ngs ** atm * tel
kappa, dsp_phi_geom, _ = ngs.get_dsp(tel)
# --- 1. Compute PSD for the WRAPPED Phase (The "Trap") ---
print("Computing PSD for wrapped phase...")
atm.unwrap_diffractive_phase = False
atm.scintillation = True
ngs ** atm * tel
# We only care about the phase here, so we ignore the rest
_, dsp_phi_wrapped, _ = ngs.get_dsp(tel)

# --- 2. Compute PSD for the UNWRAPPED Phase (The Correct Way) ---
print("Computing PSD for unwrapped phase...")
atm.unwrap_diffractive_phase = True
ngs ** atm * tel
# Extract spatial frequencies (kappa), phase PSD, and log-amplitude PSD
kappa, dsp_phi_unwrapped, dsp_chi_sim = ngs.get_dsp(tel)
atm.unwrap_diffractive_phase = False # Reset for safety

# --- 3. Compute Theoretical PSDs ---
print("Computing Theoretical PSDs...")
dsp_phi_theo = np.zeros_like(kappa)
dsp_chi_theo = np.zeros_like(kappa)

for i in range(atm.nLayer):
    # Scale r0 for the specific layer fraction
    r0_layer = atm.r0 * (atm.fractionalR0[i])**(-3/5) 

    # Get theoretical Phase (p_layer) and Scintillation (c_layer) PSDs
    p_layer, c_layer = ngs.get_theoretical_dsp(
        kappa=kappa,
        r0=r0_layer * (ngs.wavelength / 500e-9)**(6/5),
        L0=atm.L0,
        altitude=atm.altitude[i]
    )

    dsp_phi_theo += p_layer
    dsp_chi_theo += c_layer

# --- 4. Calculate Fresnel Knees (for plotting) ---
lambda_ = ngs.wavelength  # [m]
kappa_f_list = []

for i in range(atm.nLayer):
    h = atm.altitude[i]
    if h > 0:  # Ground layer (h=0) produces no scintillation
        kappa_f = 1 / np.sqrt(lambda_ * h)
        kappa_f_list.append(kappa_f)

# ============================================================================
# --- 5. PLOTTING THE RESULTS ---
# ============================================================================
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Power Spectral Density Validation', fontsize=16)

# --- Subplot 1: Phase PSD ---
axs[0].loglog(kappa, dsp_phi_wrapped, 'g-', alpha=0.6, linewidth=1.5, label='Simulation Wrapped Phase')
axs[0].loglog(kappa, dsp_phi_unwrapped, 'b-', alpha=0.8, linewidth=2, label='Simulation Unwrapped Phase')
axs[0].loglog(kappa, dsp_phi_theo , 'r--', linewidth=2.5, label='Theory (von Karman)')
axs[0].loglog(kappa, dsp_phi_geom, 'k:', linewidth=2, label='Simulation Geometric Phase')

axs[0].set_xlabel(r'Spatial Frequency $\kappa$ [$m^{-1}$]', fontsize=12)
axs[0].set_ylabel(r'Phase PSD [$rad^2 \cdot m^2$]', fontsize=12)
axs[0].set_title('Phase Power Spectral Density', fontsize=14)
axs[0].grid(True, which="both", ls="--", alpha=0.4)
axs[0].set_ylim(10e-8, 1e4)
axs[0].legend(fontsize=11)

# --- Subplot 2: Scintillation (Log-Amplitude) PSD ---
axs[1].loglog(kappa, dsp_chi_sim, 'b-', alpha=0.8, linewidth=2, label='Simulation')
axs[1].loglog(kappa, dsp_chi_theo, 'r--', linewidth=2.5, label='Theory (Fresnel Filter)')

# Add vertical lines for Fresnel knees
for i, kappa_f in enumerate(kappa_f_list):
    # Only add the label to the legend once
    label = 'Fresnel Knees' if i == 0 else ""
    axs[1].axvline(kappa_f, linestyle=':', color='gray', alpha=0.8, linewidth=2, label=label)

axs[1].set_xlabel(r'Spatial Frequency $\kappa$ [$m^{-1}$]', fontsize=12)
axs[1].set_ylabel(r'Log-Amplitude PSD [$\chi^2 \cdot m^2$]', fontsize=12)
axs[1].set_title('Scintillation Power Spectral Density', fontsize=14)
axs[1].set_ylim(1e-10, 1e-3)
axs[1].grid(True, which="both", ls="--", alpha=0.4)
axs[1].legend(fontsize=11)

plt.tight_layout()
plt.show()

# ============================================================================
# --- 6. RYTOV VARIANCE SANITY CHECK ---
# ============================================================================
print("\n--- Rytov Variance Comparison ---")

sigma_sim = ngs.rytov_variance
sigma_theo = ngs.get_theoretical_rytov(atm)

print(f"Simulated Rytov Variance  : {sigma_sim:.4e}")
print(f"Theoretical Rytov Variance: {sigma_theo:.4e}")
print(f"Ratio (Sim / Theory)      : {sigma_sim/sigma_theo:.2f}")


# %% =========================================================================
# ----------------------- 7. MODAL PROJECTION (ZERNIKE) ----------------------
# ============================================================================
"""
This section demonstrates the impact of the "Wrapped Phase Trap" on a modal 
reconstructor (which is heavily used in standard Adaptive Optics control loops).

We project the 3 phase regimes (Geometric, Unwrapped, Wrapped) onto a basis 
of 100 Zernike polynomials. 

Because Zernike polynomials are continuous across the pupil, they cannot 
physically fit the sharp, discontinuous 2π jumps of the wrapped phase. 
Attempting to project a wrapped phase results in catastrophic modal aliasing: 
the energy of the macroscopic turbulence is artificially scattered into 
high-order modes.
"""
from OOPAO.Zernike import Zernike

print("\n--- 7. Modal Projection Analysis ---")

nModes = 600
print(f"Generating {nModes} Zernike modes for the telescope pupil...")
Z = Zernike(telObject=tel, J=nModes)
Z.computeZernike(tel)

# The pseudo-inverse matrix is used to project the wavefront onto the modes.
# We use the pseudo-inverse (pinv) because the central obstruction makes 
# the Zernike basis slightly non-orthogonal.
proj_matrix = np.linalg.pinv(Z.modes)

# --- 1. Get Geometric Phase ---
atm.scintillation = False
atm.unwrap_diffractive_phase = False
ngs ** atm * tel
# FIX: Flatten the 2D phase array BEFORE applying the 1D logical indices
phase_geom_1d = ngs.phase.flatten()[tel.pupilLogical]

# --- 2. Get Unwrapped Diffractive Phase ---
atm.scintillation = True
atm.unwrap_diffractive_phase = True
ngs ** atm * tel
phase_unwrapped_1d = ngs.phase.flatten()[tel.pupilLogical]

# --- 3. Get Wrapped Diffractive Phase ---
atm.unwrap_diffractive_phase = False
ngs ** atm * tel
phase_wrapped_1d = ngs.phase.flatten()[tel.pupilLogical]

# --- PROJECT ONTO ZERNIKE BASIS ---
print("Projecting phases onto modal basis...")
coefs_geom = proj_matrix @ phase_geom_1d
coefs_unwrapped = proj_matrix @ phase_unwrapped_1d
coefs_wrapped = proj_matrix @ phase_wrapped_1d

# ============================================================================
# --- PLOTTING THE MODAL COEFFICIENTS ---
# ============================================================================
plt.figure(figsize=(10, 6))

plt.semilogy(np.abs(coefs_geom), 'k--', linewidth=3, label='Geometric Phase (Reference)')
plt.semilogy(np.abs(coefs_unwrapped), 'b-', alpha=0.7, linewidth=2, label='Unwrapped Diffractive Phase')
plt.semilogy(np.abs(coefs_wrapped), 'r-', alpha=0.7, linewidth=2, label='Wrapped Diffractive (The Trap)')

plt.xlabel('Zernike Mode Index', fontsize=12)
plt.ylabel('Absolute Modal Coefficient |a_i| [rad]', fontsize=12)
plt.title('Zernike Modal Energy Distribution', fontsize=14, fontweight='bold')
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(fontsize=11)

plt.tight_layout()
plt.show()

# Print the RMS error
err_unwrapped = np.sqrt(np.mean((coefs_unwrapped - coefs_geom)**2))
err_wrapped = np.sqrt(np.mean((coefs_wrapped - coefs_geom)**2))

print(f"\nRMS Modal Error (vs Geometric Reference):")
print(f" -> Unwrapped Phase: {err_unwrapped:.4e} rad")
print(f" -> Wrapped Phase  : {err_wrapped:.4e} rad")


# %% =========================================================================
# ----------------------- 7. MODAL VARIANCE (STATISTICAL) --------------------
# ============================================================================
"""
Statistical analysis of modal variance over multiple atmospheric realizations.

This code propagates the wavefront through the atmosphere for N iterations, 
moving the phase screens (wind) at each step. We compare the pure Geometric 
phase against the Unwrapped Diffractive phase.

Physical expectation: At high Zernike modes (high spatial frequencies), the 
Diffractive phase variance will drop below the Geometric phase variance. 
This is due to Fresnel diffraction, which acts as a cos^2 low-pass filter on 
the phase spectrum, converting some phase fluctuations into amplitude 
fluctuations (scintillation).
"""
import numpy as np
import matplotlib.pyplot as plt
from OOPAO.Zernike import Zernike
import time

print("\n--- 7. Statistical Modal Variance Analysis ---")

nModes = 200
nIters = 5000  

print(f"Generating {nModes} Zernike modes for projection...")
Z = Zernike(telObject=tel, J=nModes)
Z.computeZernike(tel)
proj_matrix = np.linalg.pinv(Z.modes)

# Tableaux pour accumuler l'énergie (carré des coefficients)
var_geom_accum = np.zeros(nModes)
var_diff_accum = np.zeros(nModes)

print(f"Running {nIters} iterations to compute statistical variance. This may take a moment...")
t0 = time.time()

for i in range(nIters):
    # Affichage de la progression
    print(f"Propagating frame {i+1}/{nIters}...", end='\r')
    
    # --- 1. GEOMETRIC PHASE ---
    atm.scintillation = False
    atm.unwrap_diffractive_phase = False
    ngs ** atm * tel
    phase_geom_1d = ngs.phase.flatten()[tel.pupilLogical]
    coefs_geom = proj_matrix @ phase_geom_1d
    
    # --- 2. DIFFRACTIVE PHASE (UNWRAPPED) ---
    atm.scintillation = True
    atm.unwrap_diffractive_phase = True
    ngs ** atm * tel
    phase_diff_1d = ngs.phase.flatten()[tel.pupilLogical]
    coefs_diff = proj_matrix @ phase_diff_1d
    
    # --- Accumuler les variances (a_i^2) ---
    var_geom_accum += coefs_geom**2
    var_diff_accum += coefs_diff**2
    
    # --- 3. FAIRE AVANCER LE VENT ---
    # Déplace les écrans de phase pour la prochaine itération
    atm.update()

print(f"\nSimulation finished in {time.time() - t0:.1f} seconds.")

# Calcul de la moyenne statistique
var_geom = var_geom_accum / nIters
var_diff = var_diff_accum / nIters

# Remise à zéro des flags par sécurité pour la suite de votre script
atm.scintillation = False
atm.unwrap_diffractive_phase = False

# ============================================================================
# --- MODÈLE ANALYTIQUE APPROCHÉ (Filtre cos² sur Zernike) ---
# ============================================================================
print("Calcul du filtre de Fresnel (cos²) effectif par mode Zernike...")

# 1. Calcul de l'ordre radial (n) pour chaque indice Zernike (j)
x_modes = np.arange(1, nModes + 1)
# Formule exacte pour retrouver l'ordre radial n à partir de l'indice de Noll/ANSI j
n_radial = np.floor((-1 + np.sqrt(1 + 8 * x_modes)) / 2)

# 2. Fréquence spatiale effective (kappa) pour chaque mode
# Un mode d'ordre n a une période caractéristique D/n, donc kappa ~ n / D
kappa_eff = n_radial / tel.D

# 3. Calcul du filtre cos² moyen pondéré par l'atmosphère
cos2_filter = np.zeros(nModes)

for h, weight in zip(atm.altitude, atm.fractionalR0):
    # Filtre de Fresnel pour la couche d'altitude h : cos²(pi * lambda * h * kappa²)
    # (Note: pour h=0 au sol, le cos² vaut 1, la phase n'est pas filtrée)
    fresnel_term = np.cos(np.pi * ngs.wavelength * h * kappa_eff**2)**2
    
    # On pondère ce filtre par la force de la turbulence (fractionalR0) de la couche
    cos2_filter += weight * fresnel_term

# 4. Création de la courbe théorique "Model with cos²"
var_model_diff = var_geom * cos2_filter

# ============================================================================
# --- PLOTTING (PAPER FIGURE 3.13 STYLE) ---
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.set_xscale('log')
ax.set_yscale('log')

# Les données simulées E2E (ronds pleins)
ax.plot(x_modes, var_diff, marker='o', color='b', linestyle='none', label='E2E diffractive phase')
ax.plot(x_modes, var_geom, marker='o', color='r', linestyle='none', label='E2E geometric phase')

# Le modèle analytique approché avec le filtre cos² (croix cyan)
ax.plot(x_modes, var_model_diff, marker='+', color='c', markersize=8, linestyle='none', label=r'Model with $\cos^2$')

ax.set_xlabel(r'$i^{th}$ Zernike', fontsize=12)
ax.set_ylabel(r'$\sigma^2(\Phi_{res})$ [rad$^2$]', fontsize=12)
ax.set_title('Modal Phase Variance', fontsize=14, fontweight='bold')

ax.grid(True, which="both", ls="-", alpha=0.5)
ax.legend(loc='lower left', fontsize=12, framealpha=0.9)

plt.tight_layout()
plt.show()

# %%

"""
Statistical analysis of modal variance over multiple atmospheric realizations.
Configuration : 100 seeds x 50 frames = 5000 iterations totales.
"""
import numpy as np
import matplotlib.pyplot as plt
from OOPAO.Zernike import Zernike
import time

print("\n--- Statistical Modal Variance Analysis ---")

# ============================================================================
# --- PARAMÈTRES DE CONVERGENCE ---
# ============================================================================
nSeeds = 100
frames_per_seed = 50
total_frames = nSeeds * frames_per_seed
nModes = 150

# Génération d'une liste de 100 seeds uniques (par exemple de 42 à 141)
list_seeds = np.arange(42, 42 + nSeeds)

# ============================================================================
# --- PRÉPARATION ZERNIKE ET STOCKAGE ---
# ============================================================================
print(f"Generating {nModes} Zernike modes for projection...")
Z = Zernike(telObject=tel, J=nModes)
Z.computeZernike(tel)
proj_matrix = np.linalg.pinv(Z.modes)

# Matrices pour stocker l'historique complet des coefficients (5000, 200)
coefs_geom_history = np.zeros((total_frames, nModes))
coefs_diff_history = np.zeros((total_frames, nModes))

print(f"Running {total_frames} iterations ({nSeeds} seeds). This will take a moment...")
t0 = time.time()

# Compteur global pour remplir les matrices d'historique
global_idx = 0 

# ============================================================================
# --- BOUCLE DE SIMULATION STATISTIQUE ---
# ============================================================================
for seed in list_seeds:
    # 1. On impose la seed de numpy
    np.random.seed(seed)
    
    # 2. On génère le nouvel écran de phase initial pour cette seed
    # (Décommente la ligne suivante si la fonction est bien implémentée dans ton objet atm)
    # atm.generateNewPhaseScreen(seed=seed)
    
    for i in range(frames_per_seed):
        # --- PHASE GÉOMÉTRIQUE ---
        atm.scintillation = False
        atm.unwrap_diffractive_phase = False
        ngs * atm * tel
        phase_geom_1d = ngs.phase.flatten()[tel.pupilLogical]
        
        # Projection et stockage
        coefs_geom_history[global_idx, :] = proj_matrix @ phase_geom_1d
        
        # --- PHASE DIFFRACTIVE (UNWRAPPED) ---
        atm.scintillation = True
        atm.unwrap_diffractive_phase = True
        ngs * atm * tel
        phase_diff_1d = ngs.phase.flatten()[tel.pupilLogical]
        
        # Projection et stockage
        coefs_diff_history[global_idx, :] = proj_matrix @ phase_diff_1d
        
        # --- FAIRE AVANCER LE VENT ---
        atm.update()
        
        global_idx += 1
        
    # Affichage de la progression par seed
    print(f"Seed {seed} completed. Progress: {global_idx}/{total_frames} frames.", end='\r')

print(f"\nSimulation finished in {time.time() - t0:.1f} seconds.")

# ============================================================================
# --- CALCUL DES VARIANCES STATISTIQUES ---
# ============================================================================
# Calcul robuste de la vraie variance temporelle sur la dimension des itérations (axis=0)
var_geom = np.var(coefs_geom_history, axis=0)
var_diff = np.var(coefs_diff_history, axis=0)

# Remise à zéro des flags de l'atmosphère
atm.scintillation = False
atm.unwrap_diffractive_phase = False

# ============================================================================
# --- MODÈLE ANALYTIQUE APPROCHÉ (Filtre cos² sur Zernike) ---
# ============================================================================
print("Calcul du modèle théorique de Fresnel...")

x_modes = np.arange(1, nModes + 1)

# Formule de Noll exacte pour retrouver l'ordre radial n à partir de l'indice j
n_radial = np.ceil((-1 + np.sqrt(1 + 8 * x_modes)) / 2) - 1

# Fréquence spatiale effective (approximation kappa ~ n / D)
kappa_eff = n_radial / tel.D

# Calcul du filtre cos² moyen pondéré par les couches de l'atmosphère
cos2_filter = np.zeros(nModes)
for h, weight in zip(atm.altitude, atm.fractionalR0):
    fresnel_term = np.cos(np.pi * ngs.wavelength * h * kappa_eff**2)**2
    cos2_filter += weight * fresnel_term

var_model_diff = var_geom * cos2_filter

# ============================================================================
# --- PLOTTING ---
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 6))

ax.set_xscale('log')
ax.set_yscale('log')

# On trace à partir de l'indice 1 (on exclut le Piston pour ne pas écraser l'échelle Y)
idx_plot = slice(1, None)

ax.plot(x_modes[idx_plot], var_diff[idx_plot], marker='o', color='b', linestyle='none', alpha=0.7, label='E2E diffractive phase')
ax.plot(x_modes[idx_plot], var_geom[idx_plot], marker='o', color='r', linestyle='none', alpha=0.7, label='E2E geometric phase')
ax.plot(x_modes[idx_plot], var_model_diff[idx_plot], marker='+', color='c', markersize=8, linestyle='none', label=r'Model with $\cos^2$')

ax.set_xlabel(r'$i^{th}$ Zernike', fontsize=12)
ax.set_ylabel(r'$\sigma^2(\Phi_{res})$ [rad$^2$]', fontsize=12)
ax.set_title(f'Modal Phase Variance ({nSeeds} seeds, {frames_per_seed} frames/seed)', fontsize=14, fontweight='bold')

ax.grid(True, which="both", ls="-", alpha=0.5)
ax.legend(loc='lower left', fontsize=12, framealpha=0.9)

plt.tight_layout()
plt.show()
# %%

import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # À installer avec : pip install SciencePlots

# =============================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES
# =============================================================================

# Chargement des données Soapy
soapy_data = np.load("dsp_results.npy", allow_pickle=True).item()
kappa_s = soapy_data["kappa"]
dsp_phi_sim_s = soapy_data["dsp_phi_sim"]
dsp_chi_sim_s = soapy_data["dsp_chi_sim"]
kappa_fresnel_s = soapy_data["kappa_fresnel_eq"]

# Troncature pour forcer OOPAO et Soapy à avoir exactement la même taille
min_len = min(len(kappa), len(kappa_s))

# On renomme proprement les variables OOPAO tronquées
kappa_o = kappa[:min_len]
dsp_phi_sim_o = dsp_phi_sim[:min_len]
dsp_chi_sim_o = dsp_chi_sim[:min_len]
# On garde uniquement la théorie OOPAO comme référence absolue
dsp_phi_theo_ref = dsp_phi_theo[:min_len]
dsp_chi_theo_ref = dsp_chi_theo[:min_len]

# =============================================================================
# 2. CONFIGURATION DU STYLE ET DE LA FIGURE
# =============================================================================

# Activation du style SciencePlots
plt.style.use(['science', 'grid', 'muted']) 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Couleurs standardisées
C_OOPAO = '#1f77b4'  # Bleu classique pour simu OOPAO
C_SOAPY = '#d62728'  # Rouge classique pour simu Soapy
C_THEO  = 'black'    # Noir pour la théorie de référence

# =============================================================================
# 3. PANNEAU 1 : DSP DE PHASE
# =============================================================================

# La Théorie (Référence Commune)
ax1.loglog(kappa_o, dsp_phi_theo_ref, color=C_THEO, linestyle='--', linewidth=2, label='Theory (Reference)')

# Les Simulations
ax1.loglog(kappa_o, dsp_phi_sim_o, color=C_OOPAO, linestyle='-', alpha=0.8, label='Simulation OOPAO')
ax1.loglog(kappa_s, dsp_phi_sim_s, color=C_SOAPY, linestyle='-', alpha=0.8, label='Simulation Soapy')

ax1.set_xlabel(r'Spatial Frequency $\kappa$ [1/m]')
ax1.set_ylabel(r'Phase PSD [$rad^2 \cdot m^2$]')
ax1.set_title('Phase Power Spectral Density')
ax1.set_ylim(1e-7, 1e2)
ax1.legend(loc='lower left', fontsize=10)

# =============================================================================
# 4. PANNEAU 2 : DSP DE SCINTILLATION
# =============================================================================

# La Théorie (Référence Commune)
ax2.loglog(kappa_o, dsp_chi_theo_ref, color=C_THEO, linestyle='--', linewidth=2, label='Theory (Reference)')

# Les Simulations
ax2.loglog(kappa_o, dsp_chi_sim_o, color=C_OOPAO, linestyle='-', alpha=0.8, label='Simulation OOPAO')
ax2.loglog(kappa_s, dsp_chi_sim_s, color=C_SOAPY, linestyle='-', alpha=0.8, label='Simulation Soapy')

# # Ajout des genoux de Fresnel (OOPAO uniquement, pour ne pas surcharger)
# for kf in kappa_f_list:
#     ax2.axvline(kf, linestyle=':', color='gray', alpha=0.5)

# # Entrée "fantôme" pour la légende des genoux
# ax2.plot([], [], linestyle=':', color='gray', alpha=0.5, label='Fresnel Knees')

ax2.set_xlabel(r'Spatial Frequency $\kappa$ [1/m]')
ax2.set_ylabel(r'Log-Amplitude PSD [$\chi^2 \cdot m^2$]')
ax2.set_title('Scintillation Power Spectral Density')
ax2.set_ylim(1e-8, 1e-3)
ax2.legend(loc='lower left', fontsize=10)

# =============================================================================
# 5. RENDU FINAL
# =============================================================================
plt.tight_layout()
plt.show()

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


# %% =========================================================================
# ----------------------- 9. TEMPORAL CONVERGENCE & STATISTICS ---------------
# ============================================================================
"""
When running End-to-End (E2E) AO simulations, we must ensure that the 
stochastic process is ergodic and converges to the theoretical statistical 
moments over time.

In this section, we run a long exposure (many frames) to verify two things:
1. Variance Stability: The instantaneous Rytov variance fluctuates heavily, 
   but its cumulative moving average must perfectly converge to the 
   theoretical analytical value.
2. Intensity PDF: In the weak fluctuation regime, the theory dictates that 
   the intensity field follows a Log-Normal probability density function (PDF).
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

print("\n--- 9. Temporal Convergence and PDF Validation ---")

n_frames = 500

# 1. Force a weak fluctuation regime to perfectly see the Log-Normal behavior
atm.rytov_var = 0.15 
atm.scintillation = True

print(f"Target Theoretical Rytov Variance: {atm.rytov_var:.4f}")
theo_rytov = float(ngs.get_theoretical_rytov(atm))

# Lists to store temporal data
inst_variance = np.zeros(n_frames)
# We will store a subset of pupil pixels to build the histogram (avoiding memory overload)
pupil_mask = tel.pupil > 0
intensity_samples = []

t0 = time.time()
print(f"Running {n_frames} iterations. Please wait...")

for i in range(n_frames):
    if i % 50 == 0:
        print(f" -> Frame {i:03d}/{n_frames}", end='\r')
        
    atm.update()       # Move the wind
    ngs ** atm * tel   # Propagate
    
    # Store instantaneous variance
    inst_variance[i] = ngs.rytov_variance
    
    # Store intensity values (normalize by mean to get I / <I>)
    I_pupil = ngs.scintillation[pupil_mask]
    I_norm = I_pupil / np.mean(I_pupil)
    
    # Keep a random subset of 1000 pixels per frame to build the PDF smoothly
    intensity_samples.extend(np.random.choice(I_norm, size=1000, replace=False))

print(f"\nSimulation finished in {time.time() - t0:.1f} seconds.")

# --- Computations for Plotting ---
# Cumulative moving average of the variance
cum_mean_variance = np.cumsum(inst_variance) / np.arange(1, n_frames + 1)

# Convert list to array for histogram
intensity_samples = np.array(intensity_samples)

# ============================================================================
# --- PLOTTING ---
# ============================================================================
fig, axs = plt.subplots(1, 2, figsize=(15, 6), dpi=120)

# --- PLOT 1: Convergence of Rytov Variance ---
frames_x = np.arange(1, n_frames + 1)

axs[0].plot(frames_x, inst_variance, color='lightgray', alpha=0.8, label='Instantaneous Variance')
axs[0].plot(frames_x, cum_mean_variance, color='#1f77b4', linewidth=2.5, label='Cumulative Moving Average')
axs[0].axhline(theo_rytov, color='#d62728', linestyle='--', linewidth=2.5, label='Theoretical Expectation')

axs[0].set_xlabel('Simulation Frame')
axs[0].set_ylabel(r'Rytov Variance $\sigma_R^2$')
axs[0].set_title('Temporal Convergence of Scintillation Variance', fontsize=14)
axs[0].set_xlim(1, n_frames)
axs[0].grid(True, linestyle='--', alpha=0.5)
axs[0].legend(loc='upper right', framealpha=0.9)

# --- PLOT 2: Intensity Probability Density Function (PDF) ---
# Plot empirical histogram
counts, bins, _ = axs[1].hist(intensity_samples, bins=100, density=True, alpha=0.6, 
                              color='#2ca02c', label='Simulated Data Histogram')

# Compute theoretical Log-Normal curve
# The variance of log-amplitude (sigma_chi^2) is approx 1/4 of Rytov variance in weak regime
sigma_chi_sq = theo_rytov / 4.0
shape = np.sqrt(4 * sigma_chi_sq)     # Shape parameter (sigma_I)
scale = np.exp(-2 * sigma_chi_sq)     # Scale parameter to ensure <I> = 1

x_pdf = np.linspace(min(bins), max(bins), 200)
pdf_theo = lognorm.pdf(x_pdf, s=shape, scale=scale)

axs[1].plot(x_pdf, pdf_theo, color='#d62728', linewidth=2.5, linestyle='--', label='Theoretical Log-Normal PDF')

axs[1].set_xlabel(r'Normalized Intensity $I / \langle I \rangle$')
axs[1].set_ylabel('Probability Density')
axs[1].set_title('Intensity Distribution (Weak Fluctuation Regime)', fontsize=14)
axs[1].grid(True, linestyle='--', alpha=0.5)
axs[1].legend(loc='upper right', framealpha=0.9)

plt.tight_layout()
plt.show()

print("==========================================================================")
print(f"Final Simulated Mean Variance: {cum_mean_variance[-1]:.4f}")
print(f"Theoretical Target Variance  : {theo_rytov:.4f}")
print(f"Error                        : {abs(cum_mean_variance[-1]-theo_rytov)/theo_rytov * 100:.2f} %")
print("==========================================================================")


# %%
import numpy as np
import matplotlib.pyplot as plt
import time

# ============================================================================
# --- PARAMÈTRES DE CONVERGENCE ---
# ============================================================================
nSeeds = 100
frames_per_seed = 100
total_frames = nSeeds * frames_per_seed

# Génération d'une liste de 100 seeds uniques
list_seeds = np.arange(42, 42 + nSeeds)

# ============================================================================
# 1. INITIALISATION DE LA SIMULATION
# ============================================================================
print("\n--- Physics Validation: Power Spectral Density (PSD) [STATISTICAL CONVERGENCE] ---")
atm.unwrap_diffractive_phase = True
atm.scintillation = True

# Accumulateurs
dsp_phi_accum = None
dsp_chi_accum = None
rytov_accum = 0.0
kappa = None

print(f"Lancement de {total_frames} itérations ({nSeeds} seeds x {frames_per_seed} frames).")
t0 = time.time()

# ============================================================================
# 2. BOUCLE DE CONVERGENCE (SEEDS & FRAMES)
# ============================================================================
for seed in list_seeds:
    
    # 1. On impose la seed globale de Numpy par sécurité
    np.random.seed(seed)
    
    # 2. On génère un NOUVEL écran de phase statique de départ pour cette seed
    atm.generateNewPhaseScreen(seed=seed)
    
    for i in range(frames_per_seed):
        # Fait avancer le vent (génère la nouvelle portion d'écran de phase)
        atm.update()
        
        # Propagation de l'étoile guide à travers l'atmosphère et le télescope
        ngs * atm * tel
        
        # Récupération de la DSP simulée pour CETTE itération
        kappa_frame, dsp_phi_frame, dsp_chi_frame = ngs.get_dsp(tel)
        
        # Initialisation des tableaux au tout premier passage global
        if dsp_phi_accum is None:
            dsp_phi_accum = np.zeros_like(dsp_phi_frame)
            dsp_chi_accum = np.zeros_like(dsp_chi_frame)
            kappa = kappa_frame
            
        # Ajout aux accumulateurs
        dsp_phi_accum += dsp_phi_frame
        dsp_chi_accum += dsp_chi_frame
        rytov_accum += ngs.rytov_variance
        
    # Affichage de la progression par seed plutôt que par frame pour éviter le spam console
    print(f"Seed {seed} traitée... Progression globale : {(np.where(list_seeds == seed)[0][0] + 1) * frames_per_seed}/{total_frames} frames", end='\r')

print(f"\nSimulation terminée en {time.time() - t0:.1f} secondes.")

# ============================================================================
# 3. CALCUL DES MOYENNES
# ============================================================================
print("Calcul des moyennes statistiques...")
dsp_phi_sim_avg = dsp_phi_accum / total_frames
dsp_chi_sim_avg = dsp_chi_accum / total_frames
sigma_sim_avg = rytov_accum / total_frames

# Remise à zéro des flags de l'atmosphère par sécurité
atm.unwrap_diffractive_phase = False
atm.scintillation = False

# ============================================================================
# 4. CALCULS THÉORIQUES (Statiques)
# ============================================================================
dsp_phi_theo = np.zeros_like(kappa)
dsp_chi_theo = np.zeros_like(kappa)

for i in range(atm.nLayer):
    r0_layer = atm.r0 * (atm.fractionalR0[i])**(-3/5) 
    
    p_layer, c_layer = ngs.get_theoretical_dsp(
        kappa=kappa,
        r0=r0_layer * (ngs.wavelength / 500e-9)**(6/5),
        L0=atm.L0,
        altitude=atm.altitude[i]
    )
    dsp_phi_theo += p_layer
    dsp_chi_theo += c_layer

# Calcul des genoux de Fresnel (pour l'affichage OOPAO)
lambda_ = ngs.wavelength 
kappa_f_list = []
for i in range(atm.nLayer):
    h = atm.altitude[i]
    if h > 0:
        kappa_f = 1 / np.sqrt(lambda_ * h)
        kappa_f_list.append(kappa_f)
        
sigma_theo = ngs.get_theoretical_rytov(atm)

# ============================================================================
# 5. AFFICHAGE DES RÉSULTATS OOPAO SEULS
# ============================================================================
print("\n--- Rytov Variance Comparison ---")
print(f"Simulated Rytov Variance (Avg) : {sigma_sim_avg:.4e}")
print(f"Theoretical Rytov Variance     : {sigma_theo:.4e}")
print(f"Ratio sim/theory               : {sigma_sim_avg/sigma_theo:.2f}\n")

plt.figure(figsize=(12, 5))

# --- Phase PSD ---
plt.subplot(1, 2, 1)
plt.loglog(kappa, dsp_phi_sim_avg, 'b-', alpha=0.8, label='Simulation OOPAO (Avg)')
plt.loglog(kappa, dsp_phi_theo , 'r--', linewidth=2, label='Theory (von Karman)')
plt.xlabel(r'Spatial Frequency $\kappa$ [1/m]')
plt.ylabel(r'Phase PSD [$rad^2 \cdot m^2$]')
plt.title(f'Phase Power Spectral Density ({total_frames} frames)')
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.ylim(10e-8, 1e4)
plt.legend()

# --- Scintillation PSD ---
plt.subplot(1, 2, 2)
plt.loglog(kappa, dsp_chi_sim_avg, 'b-', alpha=0.8, label='Simulation OOPAO (Avg)')
plt.loglog(kappa, dsp_chi_theo, 'r--', linewidth=2, label='Theory (Fresnel Filter)')
for kappa_f in kappa_f_list:
    plt.axvline(kappa_f, linestyle=':', color='gray', alpha=0.5)
plt.plot([], [], 'k:', label='Fresnel knees')
plt.xlabel(r'Spatial Frequency $\kappa$ [1/m]')
plt.ylabel(r'Log-Amplitude PSD [$\chi^2 \cdot m^2$]')
plt.title(f'Scintillation Power Spectral Density ({total_frames} frames)')
plt.ylim(1e-10, 1e-2)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()

plt.tight_layout()
plt.show()

# ============================================================================
# 6. SAUVEGARDE POUR LE SCRIPT DE COMPARAISON
# ============================================================================
oopao_results = {
    "kappa": kappa,
    "dsp_phi_sim_avg": dsp_phi_sim_avg,
    "dsp_chi_sim_avg": dsp_chi_sim_avg,
    "dsp_phi_theo": dsp_phi_theo,
    "dsp_chi_theo": dsp_chi_theo,
    "rytov_sim_avg": sigma_sim_avg,
    "total_frames_averaged": total_frames
}

np.save("dsp_results_oopao_converged.npy", oopao_results)
print("✅ Données statistiques OOPAO convergées et sauvegardées !")
# %%


import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# --- PARAMÈTRES DE CONVERGENCE ---
# ============================================================================
list_seeds = [10, 42, 100]  # Doit correspondre à Soapy
frames_per_seed = 1000       # Doit correspondre à Soapy
total_frames = len(list_seeds) * frames_per_seed

# ============================================================================
# 1. INITIALISATION DE LA SIMULATION
# ============================================================================
print("\n--- Physics Validation: Power Spectral Density (PSD) [STATISTICAL CONVERGENCE] ---")
atm.unwrap_diffractive_phase = True
atm.scintillation = True

# Accumulateurs
dsp_phi_accum = None
dsp_chi_accum = None
rytov_accum = 0.0
kappa = None

# ============================================================================
# 2. BOUCLE DE CONVERGENCE (SEEDS & FRAMES)
# ============================================================================
for seed in list_seeds:
    print(f"\nLancement de la seed : {seed}")
    
    # 1. On impose la seed globale de Numpy par sécurité
    np.random.seed(seed)
    
    # 2. On génère un NOUVEL écran de phase statique de départ pour cette seed
    atm.generateNewPhaseScreen(seed=seed)
    
    for i in range(frames_per_seed):
        # Fait avancer le vent (génère la nouvelle portion d'écran de phase)
        atm.update()
        
        # Propagation de l'étoile guide à travers l'atmosphère et le télescope
        ngs * atm * tel
        
        # Récupération de la DSP simulée pour CETTE itération
        kappa_frame, dsp_phi_frame, dsp_chi_frame = ngs.get_dsp(tel)
        
        # Initialisation des tableaux au premier passage
        if dsp_phi_accum is None:
            dsp_phi_accum = np.zeros_like(dsp_phi_frame)
            dsp_chi_accum = np.zeros_like(dsp_chi_frame)
            kappa = kappa_frame
            
        # Ajout aux accumulateurs
        dsp_phi_accum += dsp_phi_frame
        dsp_chi_accum += dsp_chi_frame
        rytov_accum += ngs.rytov_variance
        
        if (i + 1) % 10 == 0:
            print(f"   Frame {i + 1}/{frames_per_seed} traitée")

# ============================================================================
# 3. CALCUL DES MOYENNES
# ============================================================================
print(f"\nCalcul des moyennes sur {total_frames} frames...")
dsp_phi_sim_avg = dsp_phi_accum / total_frames
dsp_chi_sim_avg = dsp_chi_accum / total_frames
sigma_sim_avg = rytov_accum / total_frames

# ============================================================================
# 4. CALCULS THÉORIQUES (Statiques)
# ============================================================================
dsp_phi_theo = np.zeros_like(kappa)
dsp_chi_theo = np.zeros_like(kappa)

for i in range(atm.nLayer):
    r0_layer = atm.r0 * (atm.fractionalR0[i])**(-3/5) 
    
    p_layer, c_layer = ngs.get_theoretical_dsp(
        kappa=kappa,
        r0=r0_layer * (ngs.wavelength / 500e-9)**(6/5),
        L0=atm.L0,
        altitude=atm.altitude[i]
    )
    dsp_phi_theo += p_layer
    dsp_chi_theo += c_layer

# Calcul des genoux de Fresnel (pour l'affichage OOPAO)
lambda_ = ngs.wavelength 
kappa_f_list = []
for i in range(atm.nLayer):
    h = atm.altitude[i]
    if h > 0:
        kappa_f = 1 / np.sqrt(lambda_ * h)
        kappa_f_list.append(kappa_f)
        
sigma_theo = ngs.get_theoretical_rytov(atm)

# ============================================================================
# 5. AFFICHAGE DES RÉSULTATS OOPAO SEULS
# ============================================================================
print("\n--- Rytov Variance Comparison ---")
print(f"Simulated Rytov Variance (Avg) : {sigma_sim_avg:.4e}")
print(f"Theoretical Rytov Variance     : {sigma_theo:.4e}")
print(f"Ratio sim/theory               : {sigma_sim_avg/sigma_theo:.2f}\n")

plt.figure(figsize=(12, 5))

# --- Phase PSD ---
plt.subplot(1, 2, 1)
plt.loglog(kappa, dsp_phi_sim_avg, 'b-', alpha=0.8, label='Simulation OOPAO (Avg)')
plt.loglog(kappa, dsp_phi_theo , 'r--', linewidth=2, label='Theory (von Karman)')
plt.xlabel(r'Spatial Frequency $\kappa$ [1/m]')
plt.ylabel(r'Phase PSD [$rad^2 \cdot m^2$]')
plt.title('Phase Power Spectral Density')
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.ylim(10e-8, 1e4)
plt.legend()

# --- Scintillation PSD ---
plt.subplot(1, 2, 2)
plt.loglog(kappa, dsp_chi_sim_avg, 'b-', alpha=0.8, label='Simulation OOPAO (Avg)')
plt.loglog(kappa, dsp_chi_theo, 'r--', linewidth=2, label='Theory (Fresnel Filter)')
for kappa_f in kappa_f_list:
    plt.axvline(kappa_f, linestyle=':', color='gray', alpha=0.5)
plt.plot([], [], 'k:', label='Fresnel knees')
plt.xlabel(r'Spatial Frequency $\kappa$ [1/m]')
plt.ylabel(r'Log-Amplitude PSD [$\chi^2 \cdot m^2$]')
plt.title('Scintillation Power Spectral Density')
plt.ylim(1e-10, 1e-2)
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend()

plt.tight_layout()
plt.show()

# ============================================================================
# 6. SAUVEGARDE POUR LE SCRIPT DE COMPARAISON
# ============================================================================
# On sauvegarde avec les mêmes clés attendues par le script comparatif
oopao_results = {
    "kappa": kappa,
    "dsp_phi_sim_avg": dsp_phi_sim_avg,
    "dsp_chi_sim_avg": dsp_chi_sim_avg,
    "dsp_phi_theo": dsp_phi_theo,
    "dsp_chi_theo": dsp_chi_theo,
    "rytov_sim_avg": sigma_sim_avg,
    "total_frames_averaged": total_frames
}

np.save("dsp_results_oopao_converged.npy", oopao_results)
print("✅ Données statistiques OOPAO convergées et sauvegardées !")

# %%

import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # À installer avec : pip install SciencePlots

# =============================================================================
# 1. CHARGEMENT ET PRÉPARATION DES DONNÉES CONVERGÉES
# =============================================================================

# Chargement des données Soapy (Fichier généré à l'étape précédente)
soapy_data = np.load("dsp_results_soapy_converged.npy", allow_pickle=True).item()

# ATTENTION : On utilise bien les clés avec "_avg" issues de la convergence
kappa_s = soapy_data["kappa"]
dsp_phi_sim_s = soapy_data["dsp_phi_sim_avg"]
dsp_chi_sim_s = soapy_data["dsp_chi_sim_avg"]

# Chargement des données OOPAO
# (Assure-toi d'avoir sauvegardé les résultats convergés d'OOPAO dans un dictionnaire similaire)
oopao_data = np.load("dsp_results_oopao_converged.npy", allow_pickle=True).item()

kappa_o = oopao_data["kappa"]
dsp_phi_sim_o = oopao_data["dsp_phi_sim_avg"]  # Adapte la clé si OOPAO n'utilise pas "_avg"
dsp_chi_sim_o = oopao_data["dsp_chi_sim_avg"]

# Théorie de référence (On prend celle d'OOPAO par exemple, elles devraient être identiques)

dsp_phi_theo_ref = oopao_data["dsp_phi_theo"]
dsp_chi_theo_ref = oopao_data["dsp_chi_theo"]

# Troncature pour forcer OOPAO et Soapy à avoir exactement la même taille 
# (utile si les résolutions de grille diffèrent légèrement)
min_len_phi = min(len(kappa_o), len(kappa_s))
min_len_chi = min_len_phi # Généralement identique

# On tronque toutes les variables à la même taille
kappa_o_plot = kappa_o
kappa_s_plot = kappa_s

dsp_phi_sim_o_plot = dsp_phi_sim_o
dsp_phi_sim_s_plot = dsp_phi_sim_s
dsp_phi_theo_plot = dsp_phi_theo_ref

dsp_chi_sim_o_plot = dsp_chi_sim_o
dsp_chi_sim_s_plot = dsp_chi_sim_s
dsp_chi_theo_plot = dsp_chi_theo_ref

# # On tronque toutes les variables à la même taille
# kappa_o_plot = kappa_o[:min_len_phi]
# kappa_s_plot = kappa_s[:min_len_phi]

# dsp_phi_sim_o_plot = dsp_phi_sim_o[:min_len_phi]
# dsp_phi_sim_s_plot = dsp_phi_sim_s[:min_len_phi]
# dsp_phi_theo_plot = dsp_phi_theo_ref[:min_len_phi]

# dsp_chi_sim_o_plot = dsp_chi_sim_o[:min_len_chi]
# dsp_chi_sim_s_plot = dsp_chi_sim_s[:min_len_chi]
# dsp_chi_theo_plot = dsp_chi_theo_ref[:min_len_chi]


# =============================================================================
# 2. CONFIGURATION DU STYLE ET DE LA FIGURE
# =============================================================================

# Activation du style SciencePlots
plt.style.use(['science', 'grid', 'muted']) 

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Couleurs standardisées
C_OOPAO = '#1f77b4'  # Bleu classique pour simu OOPAO
C_SOAPY = '#d62728'  # Rouge classique pour simu Soapy
C_THEO  = 'black'    # Noir pour la théorie de référence

# =============================================================================
# 3. PANNEAU 1 : DSP DE PHASE
# =============================================================================

# La Théorie (Référence Commune)
ax1.loglog(kappa_o_plot, dsp_phi_theo, color=C_THEO, linestyle='--', linewidth=2, label='Theory (Reference)')

# Les Simulations (Convergées)
ax1.loglog(kappa_o_plot, dsp_phi_sim_avg, color=C_OOPAO, linestyle='-', alpha=0.8, label='Simulation OOPAO (Avg)')
ax1.loglog(kappa_s_plot, dsp_phi_sim_s_plot, color=C_SOAPY, linestyle='-', alpha=0.8, label='Simulation Soapy (Avg)')
# # La Théorie (Référence Commune)
# ax1.loglog(kappa_o_plot, dsp_phi_theo_plot, color=C_THEO, linestyle='--', linewidth=2, label='Theory (Reference)')

# # Les Simulations (Convergées)
# ax1.loglog(kappa_o_plot, dsp_phi_sim_o_plot, color=C_OOPAO, linestyle='-', alpha=0.8, label='Simulation OOPAO (Avg)')
# ax1.loglog(kappa_s_plot, dsp_phi_sim_s_plot, color=C_SOAPY, linestyle='-', alpha=0.8, label='Simulation Soapy (Avg)')

ax1.set_xlabel(r'Spatial Frequency $\kappa$ [1/m]')
ax1.set_ylabel(r'Phase PSD [$rad^2 \cdot m^2$]')
ax1.set_title('Phase Power Spectral Density')
# J'ai ajusté les limites pour qu'elles collent mieux aux ordres de grandeur habituels
ax1.set_ylim(1e-7, 1e2) 
ax1.set_xlim(1, 1e2) 

ax1.legend(loc='lower left', fontsize=10)

# =============================================================================
# 4. PANNEAU 2 : DSP DE SCINTILLATION
# =============================================================================

# La Théorie (Référence Commune)
ax2.loglog(kappa_o_plot, dsp_chi_theo, color=C_THEO, linestyle='--', linewidth=2, label='Theory (Reference)')

# Les Simulations (Convergées)
ax2.loglog(kappa_o_plot, dsp_chi_sim_avg, color=C_OOPAO, linestyle='-', alpha=0.8, label='Simulation OOPAO (Avg)')
ax2.loglog(kappa_s_plot, dsp_chi_sim_s_plot, color=C_SOAPY, linestyle='-', alpha=0.8, label='Simulation Soapy (Avg)')
# # La Théorie (Référence Commune)
# ax2.loglog(kappa_o_plot, dsp_chi_theo_plot, color=C_THEO, linestyle='--', linewidth=2, label='Theory (Reference)')

# # Les Simulations (Convergées)
# ax2.loglog(kappa_o_plot, dsp_chi_sim_o_plot, color=C_OOPAO, linestyle='-', alpha=0.8, label='Simulation OOPAO (Avg)')
# ax2.loglog(kappa_s_plot, dsp_chi_sim_s_plot, color=C_SOAPY, linestyle='-', alpha=0.8, label='Simulation Soapy (Avg)')

ax2.set_xlabel(r'Spatial Frequency $\kappa$ [1/m]')
ax2.set_ylabel(r'Log-Amplitude PSD [$\chi^2 \cdot m^2$]')
ax2.set_title('Scintillation Power Spectral Density')
ax2.set_ylim(1e-8, 1e-3)
ax2.legend(loc='lower left', fontsize=10)

# =============================================================================
# 5. RENDU FINAL
# =============================================================================
plt.tight_layout()
plt.show()