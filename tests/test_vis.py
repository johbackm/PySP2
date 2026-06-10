import matplotlib
import pytest
import xarray as xr
import numpy as np

import pysp2
from pysp2.util.normalized_derivative_method import plot_normalized_derivative
from pysp2.vis.plot_wave import plot_wave

matplotlib.use("Agg")

@pytest.mark.mpl_image_compare(tolerance=10)
def test_plot_normalized_derivative():

    my_sp2b = pysp2.io.read_sp2(pysp2.testing.EXAMPLE_SP2B)
    my_ini = pysp2.io.read_config(pysp2.testing.EXAMPLE_INI)
    my_binary = pysp2.util.gaussian_fit(my_sp2b, my_ini, parallel=False, baseline_to_zero=True)
    dSdt_norm = pysp2.util.central_difference(my_binary, normalize=True, baseline_to_zero=True)

    # Test the plotting function for channel 0 and record number 499
    ax = plot_normalized_derivative(my_sp2b, dSdt_norm, record_no=499, chn=0, plot_scattering_signal=True)
    fig = ax.figure
    
    return fig

@pytest.mark.mpl_image_compare(tolerance=10)
def test_plot_wave():

    my_sp2b = pysp2.io.read_sp2(pysp2.testing.EXAMPLE_SP2B)
    my_ini = pysp2.io.read_config(pysp2.testing.EXAMPLE_INI)
    my_binary = pysp2.util.gaussian_fit(my_sp2b, my_ini, parallel=False, baseline_to_zero=True)

    # Test the plotting function for channel 0 and record number 499
    display = plot_wave(my_binary, record_no=499, chn=0)
    fig = display.axes[0].figure
    return fig