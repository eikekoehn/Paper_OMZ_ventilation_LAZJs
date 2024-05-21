"""
author: Eike E. Köhn
date: Jan 29, 2024
Description: Script containing fuctions to be loaded. 
"""

#%%
import numpy as np
import numpy.ma as ma
import xarray as xr
import glob

#%%

def load_data_directories():
    output_path = 'https://data.geomar.de/thredds/dodsC/20.500.12085/dd331654-413c-4157-8796-6edf4c4be207/output/'
    return output_path

def get_topograpy():
    input_forcing_path = 'https://data.geomar.de/thredds/dodsC/20.500.12085/dd331654-413c-4157-8796-6edf4c4be207/input/forcing/'
    ds_topo = xr.open_dataset('{}topo_atl_close_gib_50N.nc'.format(input_forcing_path))
    topo_mask = ds_topo.H.values
    topo_mask[topo_mask==500.]=np.NaN
    land = ma.array(topo_mask,mask=np.isnan(topo_mask)) 
    return land, ds_topo

def get_glorys_climatology():
    glorys_path = '/nfs/kryo/work/datasets/grd/ocean/3d/ra/glorys/glorys12v1/monthly_climatology/'
    ds_glorys = xr.open_mfdataset(glob.glob('{}*.nc'.format(glorys_path)))
    return ds_glorys

def get_oxygen_data():
    woa_path = '/nfs/kryo/work/updata/woa2018/oxygen/all/1.00/'
    woa_file = 'woa18_all_o_annual.nc'
    ds_oxy = xr.open_dataset(woa_path+woa_file,decode_times=False)
    return ds_oxy

# %%
