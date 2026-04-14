# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 14:46:43 2026

@author: mpasinetti
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from OOPAO.calibration.InteractionMatrix import InteractionMatrix
from OOPAO.tools.displayTools import cl_plot, displayMap
# Import de ta nouvelle fonction (assure-toi de l'avoir collée dans tools.py)
from OOPAO.tools.tools import compute_fresnel_padding 

# %%
plt.ion()
# number of subaperture for the WFS
n_subaperture = 15
# number of pixels per subaperture 
res_factor = 8 


# %%-----------------------     TELESCOPE   ----------------------------------
from OOPAO.Telescope import Telescope

# create the Telescope object
tel = Telescope(resolution           = res_factor * n_subaperture,             # resolution of the telescope in [pix]
                diameter             = 1.52,                                    # diameter in [m]        
                samplingTime         = 1/1000,                                 # Sampling time in [s] of the AO loop
                centralObstruction   = 0.3,                                    # Central obstruction in [%] of a diameter 
                display_optical_path = False,                                  # Flag to display optical path
                fov                  = 2 )                                     # field of view in [arcsec]


#%% -----------------------     NGS & SRC   ----------------------------------
from OOPAO.Source import Source

# create the Natural Guide Star object (WFS Source)
ngs = Source(optBand     = 'Na',           # Optical band (see photometry.py)
             magnitude   = 8,              # Source Magnitude
             coordinates = [0,0])          # Source coordinated [arcsec,deg]

# create the Scientific Target object
src = Source(optBand     = 'K',            # Optical band (see photometry.py)
             magnitude   = 8,              # Source Magnitude
             coordinates = [1,0])          # Source coordinated [arcsec,deg]

# combine the sources to the telescope
ngs*tel
src*tel


#%% ----------------------- PADDING CHECK (FRESNEL) ---------------------------
# We define the atmosphere parameters here to check the grid BEFORE building the Atmosphere object
r0_500 = 0.05
altitudes = [0     ,1000 ,2000 ]
# 1. Gather all wavelengths in the simulation
wavelengths = [ngs.wavelength, src.wavelength]

# 2. Compute effective r0 for each wavelength
r0_wvl_list = [r0_500 * (wvl / 500e-9)**(6/5) for wvl in wavelengths]

# 3. Compute Max Z and Max Step for ASM validity
z_max = max(altitudes) if altitudes else 0
alts_sorted = sorted(altitudes + [0.0], reverse=True)
max_layer_step = max([alts_sorted[i] - alts_sorted[i+1] for i in range(len(alts_sorted)-1)]) if len(alts_sorted) > 1 else 0

print("\n--- Verifying Fresnel Sampling Grid ---")
# 4. Call the standalone tool
diagnostics = compute_fresnel_padding(
    D_tel=tel.D,
    resolution=tel.resolution,
    wavelengths=wavelengths,
    z_max=z_max,
    r0_wvl_list=r0_wvl_list,
    max_layer_step=max_layer_step,
    res_factor=res_factor # Pass the quantization factor!
)

# 5. Analyze diagnostics and apply the maximum padding required
max_pad_needed = 0
for wvl_str, diag in diagnostics.items():
    print(f"[{wvl_str}] -> Delta: {diag['delta_status']} | ASM: {diag['asm_status']}")
    print(f"           -> Grid Required: {diag['N_quantized']} (Pad needed: {diag['padding_per_side_needed']} px/side)")
    if diag['padding_per_side_needed'] > max_pad_needed:
        max_pad_needed = diag['padding_per_side_needed']

# 6. Apply Padding to the Telescope if necessary
if max_pad_needed > 0:
    print(f"\n=> Automatically padding the telescope by {max_pad_needed} pixels per side...")
    tel.pad(padding_values=max_pad_needed)
else:
    print("\n=> Grid is robust. No padding required.")

print(f"Final Telescope Resolution: {tel.resolution}x{tel.resolution} pixels\n")


ngs**tel
src**tel

#%% -----------------------     ATMOSPHERE   ----------------------------------
from OOPAO.Atmosphere import Atmosphere
            
# # create the Atmosphere object (Now using the securely padded telescope)
# atm = Atmosphere(telescope     = tel,                                      # Telescope                              
#                  r0            = r0_500,                                   # Fried Parameter [m]
#                  L0            = 25,                                       # Outer Scale [m]
#                  fractionalR0  = [0.45 ,0.1  ,0.1  ,0.25  ,0.1   ],        # Cn2 Profile
#                  windSpeed     = [10   ,12   ,11   ,15    ,20    ],        # Wind Speed in [m]
#                  windDirection = [0    ,72   ,144  ,216   ,288   ],        # Wind Direction in [degrees]
#                  altitude      = altitudes)  

# create the Atmosphere object
atm = Atmosphere(telescope     = tel,                               # Telescope                              
                 r0            = 0.05,                              # Fried Parameter [m]
                 L0            = 25,                                # Outer Scale [m]
                 fractionalR0  = [0.45 ,0.2  ,0.35 ], # Cn2 Profile
                 windSpeed     = [2   ,4    ,3    ], # Wind Speed in [m]
                 windDirection = [0    ,72   ,  288   ], # Wind Direction in [degrees]
                 altitude      = [0     ,1000 ,2000 ]) # Altitude Layers in [m]
                              # Altitude Layers in [m]

# atm.scintillation = True
# initialize atmosphere with current Telescope
atm.initializeAtmosphere(tel)

# The phase screen can be updated using atm.update method (Temporal sampling given by tel.samplingTime)
atm.update()

# display the atm.OPD = resulting OPD 
plt.figure()
plt.imshow(atm.OPD*1e9)
plt.title('OPD Turbulence [nm]')
plt.colorbar()

# display the atmosphere layers for the sources specified in list_src: 
atm.display_atm_layers(list_src=[ngs,src])


#%% -----------------------     Scientific Detector   ----------------------------------
from OOPAO.Detector import Detector
ngs**tel
src**tel
# define a detector with its properties (see Detector class for further documentation)
cam = Detector(integrationTime = tel.samplingTime,      # integration time of the detector
               photonNoise     = False,                  # enable photon noise
               readoutNoise    = 0,                     # readout of the detector in [e-/pixel]
               QE              = 1,                   # quantum efficiency
               psf_sampling    = 2,                     # sampling for the PSF computation 2 = Shannon sampling
               binning         = 1)                     # Binning factor of the PSF



# computation of a PSF on the detector using the '*' operator
src**tel*cam

plt.figure()
plt.imshow(cam.frame,extent=[-cam.fov_arcsec/(2*tel.padding_factor_correction),cam.fov_arcsec/(2*tel.padding_factor_correction),-cam.fov_arcsec/(2*tel.padding_factor_correction),cam.fov_arcsec/(2*tel.padding_factor_correction)])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam.pixel_size_arcsec,3))+'"')



#%%         PROPAGATE THE LIGHT THROUGH THE ATMOSPHERE
# Propagation of the light through all the objects using the * operator
ngs**atm*tel*cam
# It is possible to print the optical path: 
ngs.print_optical_path()

plt.figure()
plt.imshow(cam.frame,extent=[-cam.fov_arcsec/(2*tel.padding_factor_correction),cam.fov_arcsec/(2*tel.padding_factor_correction),-cam.fov_arcsec/(2*tel.padding_factor_correction),cam.fov_arcsec/(2*tel.padding_factor_correction)])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam.pixel_size_arcsec,3))+'"')



# Propagation of the light through all the objects using the * operator
ngs**tel*cam

# It is possible to print the optical path: 
ngs.print_optical_path()


plt.figure()
plt.imshow(cam.frame,extent=[-cam.fov_arcsec/2,cam.fov_arcsec/2,-cam.fov_arcsec/2,cam.fov_arcsec/2])
plt.xlabel('Angular separation [arcsec]')
plt.ylabel('Angular separation [arcsec]')
plt.title('Pixel size: '+str(np.round(cam.pixel_size_arcsec,3))+'"')


# %% =========================================================================
# ----------------------- 4. DEFORMABLE MIRROR (DM) --------------------------
# ============================================================================
# Because the telescope grid was padded for Fresnel propagation (scintillation), 
# the total grid size (tel.D) is now larger than the actual mirror (tel.initial_D).
# We must compute the physical pitch of the actuators and scale the DM grid accordingly.
from OOPAO.DeformableMirror import DeformableMirror
from OOPAO.MisRegistration import MisRegistration
# 1. Calculate the ground truth: the physical pitch of the DM actuators
physical_pitch = tel.initial_D / n_subaperture

# 2. Calculate the equivalent number of subapertures across the ENTIRE padded grid
n_subap_simu = int(round(tel.D / physical_pitch))

# ----------------------------------------------------------------------------
# Cas 1: Automatic DM (Method 1 - Recommended)
# Let OOPAO automatically crop the valid actuators based on the padded pupil.
# ----------------------------------------------------------------------------
dm_auto = DeformableMirror(
    telescope=tel,
    nSubap=n_subap_simu,                     # Use the padded grid size
    mechCoupling=0.35,
    pitch=physical_pitch                     # Force the true physical pitch
)

# ----------------------------------------------------------------------------
# Cas 2: Manual DM (Method 2 - Custom Constraints)
# Manually define the inner and outer physical radii to constrain the actuators.
# ----------------------------------------------------------------------------
r_inner_physique = (tel.initial_D / 2) * (tel.centralObstruction)
r_outer_physique = tel.initial_D / 2
dm_manuel = DeformableMirror(
    telescope=tel,
    nSubap=n_subap_simu,                     # Use the padded grid size
    mechCoupling=0.35,
    pitch=physical_pitch,                    # Force the true physical pitch
    r_inner=r_inner_physique,                            # Physical inner radius in meters (e.g., central obscuration)
    r_outer=r_outer_physique              # Physical outer radius in meters
)


# %% =========================================================================
# --------------------- PLOT & VERIFY DM ACTUATORS ---------------------------
# ============================================================================
import matplotlib.pyplot as plt

# On calcule l'étendue physique de la grille totale (paddée) en mètres
extent_m = [-tel.D/2, tel.D/2, -tel.D/2, tel.D/2]

plt.figure(figsize=(12, 5))

# ----------------- Plot Auto DM -----------------
plt.subplot(1, 2, 1)
# 1. On affiche la pupille paddée en fond
plt.imshow(tel.pupil, extent=extent_m, cmap='viridis', origin='lower')

# 2. On trace le cercle rouge représentant le vrai miroir physique (initial_D)
circle_auto = plt.Circle((0, 0), tel.initial_D/2, color='red', fill=False, linestyle='--', label='Physical Mirror')
plt.gca().add_patch(circle_auto)

# 3. On superpose les actuateurs
plt.plot(dm_auto.coordinates[:, 0], dm_auto.coordinates[:, 1], 'rx', label='Actuators')

plt.xlabel('Position [m]')
plt.ylabel('Position [m]')
plt.title('Auto DM: Valid Actuators (Auto-cropped)')
plt.legend(loc='upper right')


# ----------------- Plot Manual DM -----------------
plt.subplot(1, 2, 2)
# 1. On affiche la pupille paddée en fond
plt.imshow(tel.pupil, extent=extent_m, cmap='viridis', origin='lower')

# 2. On trace le cercle rouge représentant le vrai miroir physique
circle_manuel = plt.Circle((0, 0), tel.initial_D/2, color='red', fill=False, linestyle='--', label='Physical Mirror')
plt.gca().add_patch(circle_manuel)

# 3. On superpose les actuateurs
plt.plot(dm_manuel.coordinates[:, 0], dm_manuel.coordinates[:, 1], 'rx', label='Actuators')

plt.xlabel('Position [m]')
plt.ylabel('Position [m]')
plt.title('Manual DM: Valid Actuators (Custom radii)')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()

# %% =========================================================================
# ----------------------- 5. WAVEFRONT SENSOR (PWFS) -------------------------
# ============================================================================
# Similarly, the Pyramid WFS needs to account for the padded grid.
# Modulation is usually defined in terms of lambda/D. Since our simulation 'D' 
# is now larger, we must scale the modulation up using the padding factor.
from OOPAO.Pyramid import Pyramid
# Ensure the source mask is synchronized with the telescope before passing to WFS
ngs *tel
pixels_per_subap = tel.initial_resolution / n_subaperture
n_subap_padded_wfs = int(tel.resolution / pixels_per_subap)
# Compute padded subaperture parameter for the Pyramid
# n_subap_padded_wfs = int(tel.resolution / res_factor) * 4 

# make sure that the ngs is propagated to the wfs
ngs*tel

wfs = Pyramid(nSubap            = n_subap_padded_wfs,                # number of subaperture = number of pixel accros the pupil diameter
              telescope         = tel,                          # telescope object
              lightRatio        = 0.5,                          # flux threshold to select valid sub-subaperture
              modulation        = 3 ,                            # Tip tilt modulation radius * tel.padding_factor_correction
              binning           = 1,                            # binning factor (applied only on the )
              n_pix_separation  = 2,                            # number of pixel separating the different pupils
              n_pix_edge        = 1,                            # number of pixel on the edges of the pupils
              postProcessing    = 'slopesMaps_incidence_flux')  # slopesMap_incidence_flux, fullFrame_incidence_flux (see documentation)

# %%



ngs ** tel * wfs
#  display the Pyramid pupils
plt.figure()
plt.imshow(wfs.cam.frame)

# #  display the Pyramid signals
# plt.figure()
# plt.imshow(wfs.signal_2D)
# plt.colorbar()

# plt.figure()
# plt.imshow(wfs.referenceSignal_2D)
# plt.colorbar()


#  propagate to the focal plane camera to see the modulation path
wfs*wfs.focal_plane_camera
plt.figure()
plt.imshow(wfs.focal_plane_camera.frame)


print(f"=> PWFS initialized with scaled modulation: {wfs.modulation:.2f} lambda/D (padded grid)")




#%% -----------------------     Modal Basis - KL Basis  ----------------------------------


from OOPAO.calibration.compute_KL_modal_basis import compute_KL_basis
# use the default definition of the KL modes with forced Tip and Tilt. For more complex KL modes, consider the use of the compute_KL_basis function. 
M2C_KL = compute_KL_basis(tel,
                          atm,
                          dm_auto,
                          lim = 0) # inversion stability criterion

# apply the 10 first KL modes
dm_auto.coefs = M2C_KL[:,:10]
# propagate through the DM
ngs*tel*dm_auto
# show the first 10 KL modes applied on the DM
displayMap(tel.OPD)
dm_auto.coefs=0

#%%

# compute basis truncatedby the pupil
dm_auto.coefs = M2C_KL[:,:300]
ngs**tel*dm_auto

basis_3D = tel.OPD.copy()
# flatten the npix*npix dimension to compute pseudo inverse
basis = basis_3D.reshape(basis_3D.shape[0]*basis_3D.shape[1],basis_3D.shape[2])
projector = np.linalg.pinv(basis)

from OOPAO.GainSensingCamera import GainSensingCamera
gsc = GainSensingCamera(mask  = wfs.mask,
                        basis = basis_3D)

# make sure that the GSC has the right resolution (it is cropped by default)
wfs.focal_plane_camera.resolution = wfs.nRes

# GSC calibration using flat OPD
ngs** tel * wfs
wfs*wfs.focal_plane_camera
wfs.focal_plane_camera * gsc


#%% Example of the use of the GSC using the fitting error

# compute fitting error
plt.close('all')
from OOPAO.calibration.getFittingError import getFittingError
atm.update()
OPD_fitting_error, OPD_input, OPD_correction = getFittingError(OPD = atm.OPD, 
                                                             proj = projector, 
                                                             basis = basis)
# OPEN LOOP Phase screen
atm.update()
ngs ** atm * tel * wfs
wfs * wfs.focal_plane_camera
wfs.focal_plane_camera * gsc

# plt.figure(),
# plt.plot(gsc.og,label = 'OG -- open loop')
plt.figure()

# On s'assure de rapatrier la donnée sur le CPU (NumPy) pour Matplotlib
og_data = gsc.og.get() if hasattr(gsc.og, 'get') else gsc.og

plt.plot(og_data, label='OG -- open loop')
plt.legend()
# replace atm OPD by fitting error using the static OPD class
from OOPAO.OPD_map import OPD_map

fitting_OPD = OPD_map(OPD_fitting_error)
ngs ** tel * fitting_OPD * wfs
wfs * wfs.focal_plane_camera
wfs.focal_plane_camera * gsc

# plt.plot(gsc.og,label = 'OG -- fitting error')
# plt.legend()
plt.figure()

# On s'assure de rapatrier la donnée sur le CPU (NumPy) pour Matplotlib
og_data = gsc.og.get() if hasattr(gsc.og, 'get') else gsc.og

plt.plot(og_data, label='OG --fitting error')
plt.legend()
#%% -----------------------     Calibration: Interaction Matrix  ----------------------------------

# amplitude of the modes in m
stroke=1e-9
# zonal Interaction Matrix
M2C_zonal = np.eye(dm_auto.nValidAct)

# modal Interaction Matrix for all modes
M2C_modal = M2C_KL[:,:gsc.n_modes]

# swap to geometric WFS for the calibration
ngs*tel*wfs # make sure that the proper source is propagated to the WFS

# zonal interaction matrix
calib_modal = InteractionMatrix(ngs            = ngs,
                                atm            = atm,
                                tel            = tel,
                                dm             = dm_auto,
                                wfs            = wfs,   
                                M2C            = M2C_modal, # M2C matrix used 
                                stroke         = stroke,    # stroke for the push/pull in M2C units
                                nMeasurements  = 8,        # number of simultaneous measurements
                                noise          = 'off',     # disable wfs.cam noise 
                                display        = True,      # display the time using tqdm
                                single_pass    = True)      # only push to compute the interaction matrix instead of push-pull


plt.figure()
plt.plot(np.std(calib_modal.D,axis=0))
plt.xlabel('Mode Number')
plt.ylabel('WFS slopes STD')
#%% Define instrument and WFS path detectors
from OOPAO.Detector import Detector
# instrument path
src_cam = Detector(tel.resolution*2)
src_cam.psf_sampling = 4  # sampling of the PSF
src_cam.integrationTime = tel.samplingTime # exposure time for the PSF

# put the scientific target off-axis to simulate anisoplanetism (set to  [0,0] to remove anisoplanetism)
src.coordinates = [0,0]

# WFS path
ngs_cam = Detector(tel.resolution*2)
ngs_cam.psf_sampling = 4
ngs_cam.integrationTime = tel.samplingTime

ngs**tel*ngs_cam
ngs_psf_ref = ngs_cam.frame.copy()

src**tel*src_cam

src_psf_ref = src_cam.frame.copy()
# %%

# %% =========================================================================
# ----------------------- 7. SCINTILLATION VALIDATION (DSP & RYTOV) ----------
# ============================================================================
print("\n--- Scintillation Validation: Theory vs Simulation ---")

# 1. Update atmosphere and propagate light to get a fresh, uncorrelated frame
atm.update()
ngs ** atm * tel

# 2. Compute Simulated PSD (DSP) for Phase and Scintillation
# The method get_dsp() returns spatial frequencies (kappa) and the 1D PSDs
kappa, dsp_phase_simu, dsp_scint_simu = ngs.get_dsp(tel)

# 3. Compute Theoretical PSD by summing the contribution of each layer
dsp_phase_theo = np.zeros_like(kappa)
dsp_scint_theo = np.zeros_like(kappa)

for i in range(atm.nLayer):
    # Retrieve theoretical DSP for the current layer
    _, dsp_p_layer, dsp_s_layer = ngs.get_theoretical_dsp(
        kappa=kappa, 
        r0=atm.r0_layer[i], 
        L0=atm.L0_layer[i], 
        altitude=atm.altitude[i]
    )
    dsp_phase_theo += dsp_p_layer
    dsp_scint_theo += dsp_s_layer

# 4. Print Rytov Variance (Scalar validation)
print(f"Simulated Rytov Variance : {ngs.rytov_variance:.4f}")
# (Optional) You can print the theoretical one here if you have the Cn2_dh profile ready
# print(f"Theoretical Rytov Variance: {ngs.get_theoretical_rytov(atm.altitude, Cn2_dh):.4f}")

# 5. Plotting the DSPs
plt.figure(figsize=(12, 5))

# --- Phase DSP Plot ---
plt.subplot(1, 2, 1)
plt.loglog(kappa, dsp_phase_simu, 'b-', linewidth=2, label='Simulation (OPD)')
plt.loglog(kappa, dsp_phase_theo, 'r--', linewidth=2, label='Theory (von Kármán)')
plt.axvline(1/tel.D, color='k', linestyle=':', label='1/D (Telescope)')
plt.axvline(1/(tel.D/n_subaperture), color='g', linestyle=':', label='1/d (Subaperture)')
plt.xlabel('Spatial Frequency $\kappa$ [1/m]')
plt.ylabel('Phase PSD [$m^2 m^2$]')
plt.title('Phase Power Spectral Density')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()

# --- Scintillation DSP Plot ---
plt.subplot(1, 2, 2)
plt.loglog(kappa, dsp_scint_simu, 'b-', linewidth=2, label='Simulation (Scintillation)')
plt.loglog(kappa, dsp_scint_theo, 'r--', linewidth=2, label='Theory (Fresnel Filtered)')
plt.axvline(1/tel.D, color='k', linestyle=':', label='1/D')
plt.xlabel('Spatial Frequency $\kappa$ [1/m]')
plt.ylabel('Log-Amplitude PSD')
plt.title('Scintillation Power Spectral Density')
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()


# %% =========================================================================
# ----------------------- 8. ELEVATION IMPACT ON SCINTILLATION ---------------
# ============================================================================
# We simulate the telescope pointing at lower elevations (Zenith Angle > 0).
# Physically, the light travels through more atmosphere (Airmass = sec(Z)).
# This decreases r0 and drastically increases the scintillation.

zenith_angles_deg = np.array([0, 15, 30, 45, 60])
simulated_rytov = []
theoretical_rytov_scaling = []

# Base parameters at Zenith (0 degrees)
base_altitudes = np.array([0, 1000, 2000])
base_r0 = 0.05
base_fractionalR0 = [0.45, 0.2, 0.35]

print("\n--- Simulating Elevation (Zenith Angle) Impact ---")

for z in zenith_angles_deg:
    # 1. Calculate Airmass (secant of zenith angle)
    sec_z = 1.0 / np.cos(np.deg2rad(z))
    
    # 2. Scale Atmospheric Parameters
    # Altitudes increase effectively due to the slant path
    altitudes_z = base_altitudes * sec_z
    # r0 degrades with airmass to the power of -3/5
    r0_z = base_r0 * (sec_z)**(-3.0/5.0)
    
    print(f"Zenith: {z:2d}° | Airmass: {sec_z:.2f} | Eff. r0: {r0_z*100:.1f} cm | Eff. Max Alt: {altitudes_z[-1]/1000:.1f} km")
    
    # 3. Re-initialize Atmosphere with new geometric parameters
    atm_z = Atmosphere(telescope=tel, 
                       r0=r0_z, 
                       L0=25, 
                       fractionalR0=base_fractionalR0, 
                       windSpeed=[2, 4, 3], 
                       windDirection=[0, 72, 288], 
                       altitude=altitudes_z)
    atm_z.scintillation = True
    atm_z.initializeAtmosphere(tel)
    
    # 4. Accumulate frames to get a statistically stable Rytov variance
    n_frames = 10
    rytov_frames = []
    for _ in range(n_frames):
        atm_z.update()
        ngs ** atm_z * tel
        rytov_frames.append(ngs.rytov_variance)
        
    simulated_rytov.append(np.mean(rytov_frames))
    
    # 5. Theoretical scaling
    # Rytov variance scales strictly as sec(Z)^(11/6)
    theoretical_rytov_scaling.append(sec_z**(11.0/6.0))

# Convert to arrays for plotting
simulated_rytov = np.array(simulated_rytov)
theoretical_rytov_scaling = np.array(theoretical_rytov_scaling)

# Normalize theory to match the first simulated point (Zenith = 0)
theoretical_rytov_scaling = theoretical_rytov_scaling * simulated_rytov[0]

# --- Plot the Results ---
plt.figure(figsize=(8, 5))
plt.plot(zenith_angles_deg, simulated_rytov, 'bo-', markersize=8, linewidth=2, label='Simulation (OOPAO)')
plt.plot(zenith_angles_deg, theoretical_rytov_scaling, 'r--', linewidth=2, label='Theory (sec(Z)$^{11/6}$ scaling)')

plt.xlabel('Zenith Angle [Degrees]')
plt.ylabel('Rytov Variance $\sigma_R^2$')
plt.title('Scintillation Strength vs Elevation')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

print("\n=> Tutorial Complete! You are now a master of Scintillation in OOPAO.")