"""
author: Eike E. Köhn
date: Jan 29, 2024
Description: Calculate the tracer mean (thickness-weighted) and variability along 23°W.
"""

#%% Load the required packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean as cmo
from scipy.ndimage import convolve1d

#%% Load further utilities
import swm_utilities

#%% Get the data directories 
url_root_output = swm_utilities.load_data_directories()

#%% Open the datasets and put into a dictionary for common treatment afterwards
ds_eta = xr.open_dataset('{}eta.nc'.format(url_root_output),decode_times=False)
ds_T1_CH = xr.open_dataset('{}T1_CH.nc'.format(url_root_output),decode_times=False)
ds_T1_C = xr.open_dataset('{}T1_C.nc'.format(url_root_output),decode_times=False)

datasets_to_load = dict()
datasets_to_load['eta'] = dict() 
datasets_to_load['eta']['ds'] = ds_eta
datasets_to_load['eta']['varname'] = 'eta'
datasets_to_load['CH'] = dict() 
datasets_to_load['CH']['ds'] = ds_T1_CH
datasets_to_load['CH']['varname'] = 'T1_CH'
datasets_to_load['C'] = dict() 
datasets_to_load['C']['ds'] = ds_T1_C
datasets_to_load['C']['varname'] = 'T1_C'

#%% Load the last 240 years of data

number_of_years = 240
last_year = 400
data_choice = dict()
data_choice['lat_min'] = 0
data_choice['lat_max'] = 30
data_choice['lon'] = -23
data_choice['time_minidx'] = -number_of_years*12 # last 240 years

variable_dict = dict()
for var2load in datasets_to_load.keys():
    variable_dict[var2load] = dict()
    ds_dum       = datasets_to_load[var2load]['ds']
    varname_dum  = datasets_to_load[var2load]['varname']
    # select subset
    selected_t   = ds_dum.isel(TIME=slice(data_choice['time_minidx'],None))
    selected_tx  = selected_t.sel(LONGITUDE=data_choice['lon'],method="nearest")
    selected_txy = selected_tx.sel(LATITUDE=slice(data_choice['lat_min'],data_choice['lat_max']))
    # load the subset
    variable_dict[var2load]['variable'] = selected_txy.variables[varname_dum].load()
    variable_dict[var2load]['dataarray'] = selected_txy
    print('{} done'.format(varname_dum))

#%% Calculate thickness-weighted tracer concentration, the anomalies (C'', i.e. Cdd) and the relative C anom in percent
C_hat = variable_dict['CH']['variable'].mean(dim='TIME')/(variable_dict['eta']['variable'].mean(dim='TIME')+500.)
Cdd = variable_dict['C']['variable']-C_hat
C_anom_percent = Cdd/C_hat*100.

#%% Smooth the variability field by convolving with a 49-month box kernel 
kernel_length = 49 # 4 years = 4*12 (+1 to have an odd kernel)
half_kernel = int(kernel_length/2)
conv_kernel = np.ones(kernel_length)
C_anom_percent_convolved = convolve1d(C_anom_percent,conv_kernel,axis=0,mode='nearest')/kernel_length

#%% Plot the variability 
savefig = False
timevec = np.arange(last_year-number_of_years,last_year,1/12.)

plt.rcParams['font.size']=20
fig,ax = plt.subplots(1,2,gridspec_kw={'width_ratios':[1,4]},figsize=(17,8))
#% Plot the thickness-weighted mean tracer distribution
ax[0].plot(C_hat,variable_dict['C']['dataarray'].LATITUDE,color='k',linewidth=3,zorder=2)
#% Plot the temporal variability in tracer concentration (convoluted with a 49-month box kernel)
c1 = ax[1].contourf(timevec[half_kernel:-half_kernel],variable_dict['C']['dataarray'].LATITUDE,C_anom_percent_convolved[half_kernel:-half_kernel,:].T,cmap='cmo.delta',levels=np.linspace(-5,5,21),extend='both')
#% Add titles, labels, ticklabels etc.
ax[0].set_title(r"a) $\hat{C}$ at "+"{}°W".format(int(np.abs(data_choice['lon']))),loc='left')
ax[1].set_title(r"b) $C''/\hat{C}$ at "+"{}°W".format(int(np.abs(data_choice['lon']))),loc='left')
ax[0].set_xlabel('Thickness-weighted\nmean tracer\nconcentration')
ax[1].set_xlabel('Years')
ax[1].set_facecolor("#BBBBBB")
ax[1].set_xlim([last_year-number_of_years,last_year])
for idx,axi in enumerate(ax):
    axi.set_ylim([data_choice['lat_min'],data_choice['lat_max']])
    axi.set_yticks([0,10,20,30])
    axi.set_yticklabels(['Eq.','10°N','20°N','30°N'])
#% Add grid lines
for idx,axi in enumerate(ax):
    for hlineval in [0,10,20,30]:
        axi.axhline(hlineval,linestyle='--',color='k',alpha=0.5,linewidth=0.75)
for vlineval in [0.4,0.6]:
    ax[0].axvline(vlineval,linestyle='--',color='k',alpha=0.5,linewidth=0.75)
for vlineval in [200,250,300,350]:
    ax[1].axvline(vlineval,linestyle='--',color='k',alpha=0.5,linewidth=0.75)
#% Add colorbar
plt.tight_layout()
plt.colorbar(c1,ax=ax[1],pad=0.01,label='Tracer anomaly in %',ticks=[-4,-2,0,2,4])
plt.subplots_adjust(wspace=0.15,right=1.02)
if savefig == True:
    figname = 'fig_23W_SWM'
    plt.savefig('{}.png'.format(figname),dpi=250)
    plt.savefig('{}.pdf'.format(figname),dpi=250)
plt.show()

# %%
