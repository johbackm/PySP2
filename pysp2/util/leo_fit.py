import numpy as np
from scipy.optimize import curve_fit
from .peak_fit import _gaus


def beam_shape(my_binary, beam_position_from='peak maximum', Globals=None):
    """
    
    Calculates the beam shape needed to determine the laser intensity profile
    used for leading-edge-only fits. The beam position is first determined
    from the position of the peak. The beam profile is then calculated by
    positioning all the peaks to that position. The beam shape is then the mean
    of all the normalized profiles arround that position.
    
    Parameters
    ----------
    my_binary : xarray Dataset
            Dataset with gaussian fits. The input data set is retuned by 
            pysp2.util.gaussian_fit().

    Globals: DMTGlobals structure or None
        DMTGlobals structure containing calibration coefficients and detector 
        signal limits.
        
    beam_position_from : str
           'peak maximum' = construct the beam profile arround the maximum peak
           poistion. The maximum peak position is determied from the peak-height
           weighted average peak position.
           'split detector' = construct the beam profile around the split position.
           The split position is taken from the split detector. Not working yet.
           
    Returns
    -------
    coeff : numpy array with gaussian fit [amplitude, peakpos, width, base] 
            of the beam shape profile.
    beam_profile : numpy array with the beam profile calculated from the mean
                   of all the profiles. The array has as many data points as 
                   the input data.

    """
        
    num_base_pts_2_avg = 10
    bins=my_binary['columns']
    
    #boolean array for ok particles in the high gain channel
    scatter_high_gain_accepted = np.logical_and.reduce((
        my_binary['PkFWHM_ch0'].values > Globals.ScatMinWidth,
        my_binary['PkFWHM_ch0'].values < Globals.ScatMaxWidth,
        my_binary['PkHt_ch0'].values > Globals.ScatMinPeakHt1,
        my_binary['PkHt_ch0'].values < Globals.ScatMaxPeakHt1,
        my_binary['FtPos_ch0'].values < Globals.ScatMaxPeakPos,
        my_binary['FtPos_ch0'].values > Globals.ScatMinPeakPos))
    
    #boolean array that is True for particles that have not been triggered by 
    #the high gain incandesence channel
    no_incand_trigged = my_binary['PkHt_ch1'].values < Globals.IncanMinPeakHt1
    
    #find good events that only scatter light
    only_scattering_high_gain = np.logical_and.reduce((scatter_high_gain_accepted,
                                                       no_incand_trigged))
        
    print('High gain scattering particles for beam analysis :: ',
          np.sum(only_scattering_high_gain))
    
    #make an xarray of the purely scattering particles
    my_high_gain_scatterers = my_binary.sel(index = only_scattering_high_gain,
                                            event_index = only_scattering_high_gain)
    
    #numpy array for the normalized beam profiels
    my_high_gain_profiles = np.zeros((my_high_gain_scatterers.dims['index'],
                                    my_high_gain_scatterers.dims['columns'])) \
                                    * np.nan
    
    mean_high_gain_max_peak_pos = np.nanmean(my_high_gain_scatterers['PkPos_ch0'].values)
    
    #weighted mean of beam peak position. Weight is scattering amplitude.
    # high_gain_peak_pos = int(
    #     np.sum(np.multiply(my_high_gain_scatterers['PkPos_ch0'].values,
    #     my_high_gain_scatterers['PkHt_ch0'].values))/ \
    #                         np.sum(my_high_gain_scatterers['PkHt_ch0'].values))
    
    #loop through all particle events
    for i in my_high_gain_scatterers['event_index']:
        data = my_high_gain_scatterers['Data_ch0'].sel(event_index=i).values
        #base level
        base = np.mean(data[0:num_base_pts_2_avg])
        #peak height
        peak_height = data.max() - base
        #peak position
        peak_pos = data.argmax()
        #normalize the profile to range [0,1]
        profile = (data - base) / peak_height
        #distance to the mean beam peak position
        peak_difference = mean_high_gain_max_peak_pos - peak_pos
        #insert so that the peak is at the right position (accounts for 
        #particles travelling at different speeds)
        if peak_difference > 0:
            my_high_gain_profiles[i, peak_difference:] = profile[:len(data) - 
                                                                peak_difference]
        elif peak_difference < 0:
            my_high_gain_profiles[i, :len(data)+peak_difference] = profile[-peak_difference:]
        else:
            my_high_gain_profiles[i, :] = profile

    #get the beam profile
    beam_profile = np.nanmean(my_high_gain_profiles, axis=0)   
    return beam_profile

def leo_fit(my_binary,Globals=None):
    bins = my_binary['columns'].astype('float').values
    #number of points at the beginning to use for base line average
    num_base_pts_2_avg = 10
    num_base_pts_2_avg_2 = 10
    max_amplitude_fraction = 0.03
    
    bl_scattering_ok = my_binary['ScatRejectKey'].values == 0
    bl_only_scattering_particles = np.logical_and(bl_scattering_ok, 
                                                  my_binary['PkHt_ch1'].values < Globals.IncanMinPeakHt1)

    #Particles that only scatter light and ch0 not saturated 
    bl_only_scattering_particles_ch0 = np.logical_and(my_binary['PkHt_ch0'].values < Globals.ScatMaxPeakHt1,
                                                      bl_only_scattering_particles)
    
    #split to peak height difference (in bins) for scattering only particles
    split_to_peak_high_gain = my_binary['PkPos_ch0'].values - my_binary['PkSplitPos_ch3'].values
    #For particles with inandesence signal, set to NaN since the peak needn't 
    #be where the laser intensity is the highest, so se to NaN
    split_to_peak_high_gain[~bl_only_scattering_particles_ch0] = np.nan
    
    #get the information about the gaussian fits
    pos = my_binary['FtPos_ch0'].values
    amplitude = my_binary['PkHt_ch0'].values
    width = my_binary['PkFWHM_ch0'].values / 2.35482 #2*sqrt(2*log(2))
    data_ch0 = my_binary['Data_ch0'].values
    
    #mean of the first num_base_pts_2_avg points
    #leo_base_ch0 = np.mean(data_ch0[:, 0:num_base_pts_2_avg], axis=1)
    #mean of the lowest 3 points
    leo_base_ch0 = np.mean(np.sort(data_ch0[:, 0:num_base_pts_2_avg], 
                                   axis=1)[:,:num_base_pts_2_avg_2], axis=1)
    
    leo_fit_max_pos = np.zeros_like(bl_only_scattering_particles,dtype=np.int64)
    #leo_fit_max_pos_ = np.zeros_like(bl_only_scattering_particles,dtype=np.float64)
    leo_PkHt_ch0 = np.zeros_like(my_binary['PkHt_ch0'].values)*np.nan
    leo_PkHt_ch0_ = np.zeros_like(my_binary['PkHt_ch0'].values)*np.nan

    ilocs = np.argwhere(bl_only_scattering_particles_ch0).flatten()
    for i in ilocs:
        leo_fit_max_pos[i] = np.round(pos[i] - width[i] * np.sqrt(2 * np.log(1. / max_amplitude_fraction)))
        leo_fit_max_pos[i] = 22
        fraction_of_full_power = _gaus(leo_fit_max_pos[i], 1, pos[i], width[i], 0)
        fractional_peak_height_ch0 = data_ch0[i, leo_fit_max_pos[i]] - leo_base_ch0[i]
        if leo_fit_max_pos[i] > 10:
            bins_ = bins[:leo_fit_max_pos[i]]
            #signals
            data_ch0_ = data_ch0[i, :leo_fit_max_pos[i]]
            leo_coeff, var_matrix = curve_fit(
                lambda x, a: _gaus(x, a, pos[i], width[i], leo_base_ch0[i]), 
                bins_, data_ch0_, p0=[amplitude[i]], maxfev=40, 
                ftol=1e-3, method='lm') #bounds=(0, 1e6), method='dogbox'
            leo_PkHt_ch0[i] = leo_coeff
            estimated_peak_height_ch0 = fractional_peak_height_ch0 / fraction_of_full_power
            leo_PkHt_ch0_[i] = estimated_peak_height_ch0
    
    output_ds = my_binary.copy()
    output_ds['leo_FtAmp_ch0'] = (('index'), leo_PkHt_ch0)
    output_ds['leo_FtAmp_ch0_'] = (('index'), leo_PkHt_ch0_)
    output_ds['leo_FtMaxPos_ch0'] = (('index'), leo_fit_max_pos)
    output_ds['leo_Base_ch0'] = (('index'), leo_base_ch0)
    
    return output_ds
