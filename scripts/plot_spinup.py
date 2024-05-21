"""
author: Eike E. Köhn
date: Jan 22, 2024
description: Plot the interface displacement and the tracer concentration during the spinup of the SWM using snapshots from the model output. 
"""

#%% Load the packages
print('Load packages.')
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean as cmo
from matplotlib import colors as c
import os
#% Load further utilities
import swm_utilities

#%% Get the data directories 
print('Get data directories.')
url_root_output = swm_utilities.load_data_directories()

#%% Get the land mask
print('Get land mask for plotting later on.')
land, ds_topo = swm_utilities.get_topograpy()

#%% Open the datasets obtained from the GEOMAR THREDDS Server
print('Open the required datasets.')
#
def preprocess(ds):
    # Drop the last time slice to avoid time overlaps when opening using open_mfdataset
    ds = ds.isel(TIME=slice(None, -1))
    return ds
#
T1_C_snapshot_files = ['{}run001000000000001_T1_C_fl.nc'.format(url_root_output),
                      '{}run002000000000001_T1_C_fl.nc'.format(url_root_output)]
ds_T1_C = xr.open_mfdataset(T1_C_snapshot_files,concat_dim='TIME',combine='nested',data_vars='minimal',preprocess=preprocess)
#
eta_snapshot_files = ['{}run001000000000001_eta_fl.nc'.format(url_root_output),
                      '{}run002000000000001_eta_fl.nc'.format(url_root_output)]
ds_eta = xr.open_mfdataset(eta_snapshot_files,concat_dim='TIME',combine='nested',data_vars='minimal',preprocess=preprocess)

#%% Select eta and the tracer concentration during the spinup at months 6, 60 and 600 - and load the data
print('Select and load snapshots.')
months = [6,60,600] 
eta_sels = [ds_eta.eta.isel(TIME=month).load() for month in months]
T1_C_sels = [ds_T1_C.T1_C.isel(TIME=month).load() for month in months]
h0 = 500. # undisturbed layer depth

#%% Plot the snapshots
print('Set plotting parameters.')
savefig = True
panel_labs = ['a)','b)','c)','d)','e)','f)']
props = dict(boxstyle='round', facecolor='#EEEEEE', alpha=0.9)
#
print('Create the plot.')
fontsize=20
plt.rcParams['font.size']=fontsize
fig, ax = plt.subplots(2,3,figsize=(17,8.5),sharex=True,sharey=True)
for mdx,month in enumerate(months):
    c0 = ax[0,mdx].contourf(ds_eta.LONGITUDE,ds_eta.LATITUDE,eta_sels[mdx]+h0,levels=np.linspace(350,570,23),cmap='cmo.delta',extend='both')
    c1 = ax[1,mdx].contourf(ds_T1_C.LONGITUDE,ds_T1_C.LATITUDE,T1_C_sels[mdx],levels=np.linspace(0,1,21),cmap='cmo.haline',extend='min')
plt.tight_layout()
plt.subplots_adjust(right=0.9,left=0.23,top=0.93)
cbax0 = fig.add_axes((0.91,0.525,0.02,0.4))
cbar0 = plt.colorbar(c0,cax=cbax0)
cbax1 = fig.add_axes((0.91,0.08,0.02,0.4))
cbar0 = plt.colorbar(c1,cax=cbax1,ticks=np.linspace(0,1,6))
for idx,axi in enumerate(ax.flatten()):
    axi.pcolormesh(ds_topo.LONGITUDE1,ds_topo.LATITUDE,land,cmap=c.ListedColormap(['#777777']))
    axi.text(0.04,0.04,panel_labs[idx],transform=axi.transAxes,va='bottom',ha='left',bbox=props)
    for hlineval in [0,20,40]:
        axi.axhline(hlineval,linestyle='--',color='k',alpha=0.1,linewidth=0.75)
    for vlineval in [-60,-40,-20,0]:
        axi.axvline(vlineval,linestyle='--',color='k',alpha=0.1,linewidth=0.75)
for axi in ax[-1,:]:
    axi.set_xticks([-60,-40,-20,0])
    axi.set_xticklabels(['60°W','40°W','20°W','0°'],fontsize=fontsize-2)
for axi in ax[:,0]:
    axi.set_yticks([-20,0,20,40])
    axi.set_yticklabels(['20°S','Eq.','20°N','40°N'],fontsize=fontsize-2)
for idx,axi in enumerate(ax[0,:]):
    axi.set_title('month {}'.format(int(months[idx])),fontsize=fontsize+2)
fig.text(0.1,0.66,'Layer\nthickness\n'+r"$h$ in m",fontweight='bold',ha='center',fontsize=fontsize+2)
fig.text(0.1,0.22,'Tracer\nconcentration\n'+r'$C$',fontweight='bold',ha='center',fontsize=fontsize+2)
if savefig==True:
    figname = '../plots/fig_spinup'
    plt.savefig('{}.png'.format(figname),dpi=250)
    os.system('convert {}.png {}.pdf'.format(figname,figname))
plt.show()


# %%
