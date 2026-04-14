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
                fov                  = 0 )           
              
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
elevation = 45  # elevation angle in deg
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

#%% ------------------- PROPAGATION REGIMES COMPARISON -------------------
"""
This section explores the 4 available propagation channels in OOPAO.

1. Pure Geometric (No Scintillation)
   - Physics: Ray optics approximation. The phase is purely integrated along 
     the optical path. The amplitude remains uniform (flat intensity).
   - Reality: Valid for near-field, large wavelengths, or observing at zenith.
   - Limit: Ignores diffraction. Cannot model scintillation (speckles) or 
     amplitude variations caused by deep turbulence or low-elevation targets.

2. Full Diffractive (Wrapped Phase + Scintillation)
   - Physics: Full wave optics via Fresnel propagation (Angular Spectrum Method). 
     Properly models diffractive interference causing amplitude fluctuations.
   - Reality: The exact physical electromagnetic field at the pupil plane.
   - Limit: The phase is strictly bounded modulo 2π (wrapped). This makes it 
     unusable for linear modal reconstructors or evaluating the macroscopic 
     wavefront error without complex post-processing.

3. Hybrid: Geometric Phase + Scintillation
   - Physics: Mixes the physically propagated intensity (Fresnel) with the 
     macroscopic geometric phase (Ray tracing). 
   - Reality: Provides the macroscopic unwrapped phase for AO loop stability 
     while retaining the intensity variations on the wavefront sensor.
   - Limit: Ignores the high-frequency diffractive phase residuals (the tiny 
     ripples on the wavefront). It's an approximation, though very efficient.

4. Unwrapped Diffractive Phase + Scintillation
   - Physics: Uses the geometric phase as a guide to 
     perfectly unwrap the true diffractive phase. Contains BOTH the macroscopic 
     turbulence and the diffractive residuals.
   - Reality: The exact unwrapped physical phase.
   - Limit: Computationally heavier. If the scintillation is extremely strong 
     (e.g., extremely low elevation), the residual itself might wrap, breaking 
     the unwrapping process.
"""

from OOPAO.Atmosphere import Atmosphere

# create the Atmosphere object   
atm = Atmosphere(telescope     = tel,                                          
                 r0            = 0.05,                            
                 L0            = 25,                                
                 fractionalR0  = [0.45, 0.2, 0.35],                
                 windSpeed     = [10   ,12   ,11],                
                 windDirection = [0, 72, 288],             
                 altitude      = altitudes)       
               

# --- EXTRACTING THE 4 CASES ---

# CASE 1: Pure Geometric (No Scintillation)
atm.angular_spectrum_propagation = False
atm.initializeAtmosphere(tel)
ngs ** atm * tel
phase_geom = ngs.phase
amp_geom = ngs.scintillation

# CASE 2: Full scintillation and diffractive phase
atm.angular_spectrum_propagation = True
ngs ** atm * tel
phase_diffractive = ngs.phase
amp_diff_scint = ngs.scintillation

# CASE 3: Geometric + Scintillation (OOPAO Optimized)
atm.geometric_phase_backup = True
ngs ** atm * tel
phase_geom_scint = ngs.phase
amp_geom_scint = ngs.scintillation
atm.geometric_phase_backup = False # Reset for next case

# CASE 4: Unwrapped phase + scintillation 
atm.unwrap_diffractive_phase = True
ngs ** atm * tel
phase_unwrapped = ngs.phase
amp_unwrap = ngs.scintillation
atm.unwrap_diffractive_phase = False # Reset

# --- PLOTTING SETUP ---

# Zoom setup
N_padded = phase_diffractive.shape[0]  
N_phys = tel.initial_resolution  
marge_px = 10
idx_min_base = (N_padded - N_phys) // 2
idx_max_base = idx_min_base + N_phys
idx_min = max(0, idx_min_base - marge_px)
idx_max = min(N_padded, idx_max_base + marge_px)

# ---  THE 4 REGIMES COMPARISON ---
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle('Wavefront Regimes Comparison', fontsize=16)

# Row 1: Phase (Corrected colorbars and variables)
im0 = axs[0, 0].imshow(phase_diffractive, cmap='plasma')
axs[0, 0].set_title('Phase: Diffractive (Wrapped)')
fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

im1 = axs[0, 1].imshow(phase_unwrapped, cmap='plasma')
axs[0, 1].set_title('Phase: Unwrapped')
fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

im2 = axs[0, 2].imshow(phase_geom_scint, cmap='plasma')
axs[0, 2].set_title('Phase: Geom + Scint')
fig.colorbar(im2, ax=axs[0, 2], fraction=0.046, pad=0.04)

im3 = axs[0, 3].imshow(phase_geom, cmap='plasma')
axs[0, 3].set_title('Phase: Pure Geometric')
fig.colorbar(im3, ax=axs[0, 3], fraction=0.046, pad=0.04)

# Row 2: Amplitude (Corrected to show proper amplitude vars and colorbars)
im4 = axs[1, 0].imshow(amp_diff_scint, cmap='viridis') 
axs[1, 0].set_title('Amplitude: Diffractive')
fig.colorbar(im4, ax=axs[1, 0], fraction=0.046, pad=0.04)

im5 = axs[1, 1].imshow(amp_unwrap, cmap='viridis')
axs[1, 1].set_title('Amplitude: Unwrapped')
fig.colorbar(im5, ax=axs[1, 1], fraction=0.046, pad=0.04)

im6 = axs[1, 2].imshow(amp_geom_scint, cmap='viridis')
axs[1, 2].set_title('Amplitude: Geom + Scint')
fig.colorbar(im6, ax=axs[1, 2], fraction=0.046, pad=0.04)

im7 = axs[1, 3].imshow(amp_geom, cmap='viridis')
axs[1, 3].set_title('Amplitude: Pure Geom (Flat)')
fig.colorbar(im7, ax=axs[1, 3], fraction=0.046, pad=0.04)


for ax in axs.flat:
    ax.set_xlim(idx_min, idx_max)
    ax.set_ylim(idx_max, idx_min) 

plt.tight_layout()
plt.show()

# --- PHASE DIFFERENCES (RESIDUALS) ---
fig2, axs2 = plt.subplots(1, 3, figsize=(16, 5))
fig2.suptitle('Phase Differences Analysis', fontsize=16)

# 1. Unwrapped vs Geometric -> Reveals the pure diffractive residual (the "ripples")
diff1 = phase_unwrapped - phase_geom
# Use symmetric scaling around 0 for residuals
v_max1 = np.max(np.abs(diff1))
im_d1 = axs2[0].imshow(diff1, cmap='RdBu', vmin=-v_max1, vmax=v_max1)
axs2[0].set_title('Unwrapped - Geometric\n(= Diffractive Residual)')
fig2.colorbar(im_d1, ax=axs2[0], fraction=0.046, pad=0.04)

# 2. Diffractive vs Geometric -> Shows wrapping artifacts + residual
diff2 = phase_diffractive - phase_geom
v_max2 = np.max(np.abs(diff2))
im_d2 = axs2[1].imshow(diff2, cmap='RdBu', vmin=-v_max2, vmax=v_max2)
axs2[1].set_title('Diffractive (Wrapped) - Geometric')
fig2.colorbar(im_d2, ax=axs2[1], fraction=0.046, pad=0.04)

# 3. Unwrapped vs Diffractive -> Shows the 2π macroscopic steps that were restored
diff3 = phase_unwrapped - phase_diffractive
v_max3 = np.max(np.abs(diff3))
im_d3 = axs2[2].imshow(diff3, cmap='RdBu', vmin=-v_max3, vmax=v_max3)
axs2[2].set_title('Unwrapped - Diffractive\n(= Macroscopic Phase Jumps)')
fig2.colorbar(im_d3, ax=axs2[2], fraction=0.046, pad=0.04)

for ax in axs2:
    ax.set_xlim(idx_min, idx_max)
    ax.set_ylim(idx_max, idx_min)

plt.tight_layout()
plt.show()



#%% -----------------------     Propagation to WFS   ----------------------------------
"""
In OOPAO, coupling scintillation with Wavefront Sensors is completely seamless. 
When `atm.angular_spectrum_propagation = True`, the framework automatically propagates the 
full complex electromagnetic field (both phase aberrations and amplitude 
fluctuations) through the optical chain. 
"""
print("\n--- 5. Initializing Wavefront Sensors ---")

pixels_per_subap = tel.initial_resolution / n_subaperture
n_subap_padded_wfs = int(tel.resolution / pixels_per_subap)

# Initialize PWFS
pwfs = Pyramid(nSubap=n_subap_padded_wfs, telescope=tel, lightRatio=0.5, modulation=3, 
               binning=1, n_pix_separation=2, n_pix_edge=1, postProcessing='slopesMaps') 

# Initialize SHWFS
shwfs = ShackHartmann(nSubap=n_subap_padded_wfs, telescope=tel, lightRatio=0.5, is_geometric=False)

# --- Test With Scintillation ---
atm.angular_spectrum_propagation = True
atm.unwrap_diffractive_phase = True
ngs ** atm * tel
ngs * pwfs
ngs * shwfs
pwfs_frame_scint = pwfs.cam.frame.copy()
shwfs_frame_scint = shwfs.cam.frame.copy()

# --- Test Without Scintillation ---
atm.angular_spectrum_propagation = False
atm.unwrap_diffractive_phase = False # Reset for normal propagation
ngs ** atm * tel
ngs * pwfs
ngs * shwfs
pwfs_frame_no_scint = pwfs.cam.frame.copy()
shwfs_frame_no_scint = shwfs.cam.frame.copy()

# --- Compute Differences and Sums ---
diff_pwfs = pwfs_frame_scint - pwfs_frame_no_scint
diff_shwfs = shwfs_frame_scint - shwfs_frame_no_scint

# SHWFS Sums
sum_sh_no_scint = np.sum(shwfs_frame_no_scint)
sum_sh_scint = np.sum(shwfs_frame_scint)
sum_sh_diff = np.sum(diff_shwfs)

# PWFS Sums
sum_pwfs_no_scint = np.sum(pwfs_frame_no_scint)
sum_pwfs_scint = np.sum(pwfs_frame_scint)
sum_pwfs_diff = np.sum(diff_pwfs)

# ============================================================================
# --- PLOTTING SHWFS ---
# ============================================================================
marge_px_shwfs = 10
N_padded_shwfs = shwfs_frame_no_scint.shape[0]
N_phys_shwfs = tel.initial_resolution 

idx_min_shwfs = max(0, (N_padded_shwfs - N_phys_shwfs) // 2 - marge_px_shwfs)
idx_max_shwfs = min(N_padded_shwfs, (N_padded_shwfs - N_phys_shwfs) // 2 + N_phys_shwfs + marge_px_shwfs)

val_max_shwfs = np.max(shwfs_frame_no_scint) * 0.8 

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Impact of Scintillation on SHWFS Camera', fontsize=14)

im0 = axs[0].imshow(shwfs_frame_no_scint, cmap='magma', vmin=0, vmax=val_max_shwfs)
axs[0].set_title(f'Without Scintillation\nSum = {sum_sh_no_scint:.2e}')
fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

im1 = axs[1].imshow(shwfs_frame_scint, cmap='magma', vmin=0, vmax=val_max_shwfs)
axs[1].set_title(f'With Scintillation\nSum = {sum_sh_scint:.2e}')
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

val_max_diff = np.max(np.abs(diff_shwfs)) * 0.8
im2 = axs[2].imshow(diff_shwfs, cmap='RdBu_r', vmin=-val_max_diff, vmax=val_max_diff)
axs[2].set_title(f'Difference (Scint - No Scint)\nSum = {sum_sh_diff:.2e}')
fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

for ax in axs:
    ax.set_xlim(idx_min_shwfs, idx_max_shwfs)
    ax.set_ylim(idx_max_shwfs, idx_min_shwfs)

plt.tight_layout()
plt.show()

# ============================================================================
# --- PLOTTING PWFS ---
# ============================================================================
marge_px_pwfs = 0 
N_padded_pwfs = pwfs_frame_no_scint.shape[0]
N_phys_pwfs = tel.initial_resolution 

idx_min_pwfs = max(0, (N_padded_pwfs - N_phys_pwfs) // 2 - marge_px_pwfs)
idx_max_pwfs = min(N_padded_pwfs, (N_padded_pwfs - N_phys_pwfs) // 2 + N_phys_pwfs + marge_px_pwfs)

val_max_pwfs = max(np.max(pwfs_frame_no_scint), np.max(pwfs_frame_scint))

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Impact of Scintillation on Pyramid WFS Camera', fontsize=14)

im0 = axs[0].imshow(pwfs_frame_no_scint, cmap='plasma', vmin=0, vmax=val_max_pwfs)
axs[0].set_title(f'Without Scintillation\nSum = {sum_pwfs_no_scint:.2e}')
fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

im1 = axs[1].imshow(pwfs_frame_scint, cmap='plasma', vmin=0, vmax=val_max_pwfs)
axs[1].set_title(f'With Scintillation\nSum = {sum_pwfs_scint:.2e}')
fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

val_max_diff_pwfs = np.max(np.abs(diff_pwfs))
im2 = axs[2].imshow(diff_pwfs, cmap='coolwarm', vmin=-val_max_diff_pwfs, vmax=val_max_diff_pwfs)
axs[2].set_title(f'Difference (Scint - No Scint)\nSum = {sum_pwfs_diff:.2e}')
fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

for ax in axs:
    ax.set_xlim(idx_min_pwfs, idx_max_pwfs)
    ax.set_ylim(idx_max_pwfs, idx_min_pwfs)

plt.tight_layout()
plt.show()


