#%% Load packages
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean as cmo
import numpy.ma as ma
from matplotlib import colors as c
import os

#%% Load further utilities and define function

import swm_utilities

def add_arrows_to_plots(fig,start_axi,end_axi,connectionstyle):
    from matplotlib import patches
    arrow = patches.ConnectionPatch(
        [-25,59],
        [-25,-21],
        coordsA=start_axi.transData,
        coordsB=end_axi.transData,
        color="#777777",
        arrowstyle="-|>",  # "normal" arrow
        mutation_scale=30,  # controls arrow head size
        connectionstyle=connectionstyle,
        linewidth=3,
    )
    fig.patches.append(arrow)

def set_up_gradient_arrays(variable_dict):
    A = 6371000
    lon_e = 15
    lon_s = -65
    lat_e = 50
    lat_s = -20
    nx = 801
    ny = 701
    d2r = (2*np.pi)/360
    dLambda = (lon_e - lon_s)/(nx-1)*d2r # longitudinal grid spacing in radians
    dTheta = (lat_e - lat_s)/(ny-1)*d2r # latitudinal grid spacing in radians
    coslat_v = np.cos(variable_dict['vhc']['dataarray'].LATITUDE.values*np.pi/180)
    coslat_h = np.cos(variable_dict['eta']['dataarray'].LATITUDE.values*np.pi/180)
    return dLambda, dTheta, coslat_v, coslat_h, A

def calc_advective_flux_convergence(variable_dict,A,dLambda,dTheta,coslat_h,coslat_v):
    uhc_advflux_convergence = np.zeros_like(variable_dict['eta']['varmean'])  + np.NaN
    vhc_advflux_convergence = np.zeros_like(variable_dict['eta']['varmean'])  + np.NaN
    uhc_advflux_convergence[:,:-1] = (-1)*(variable_dict['uhc']['varmean'][:,1:].values - variable_dict['uhc']['varmean'][:,:-1].values) / dLambda / A / coslat_h[:,None]
    vhc_advflux_convergence[:-1,:] = (-1)*(variable_dict['vhc']['varmean'][1:,:].values * coslat_v[1:,None] - variable_dict['vhc']['varmean'][:-1,:].values * coslat_v[:-1,None]) / dTheta / A / coslat_h[:-1,None]
    advective_flux_convergence = uhc_advflux_convergence + vhc_advflux_convergence
    return uhc_advflux_convergence, vhc_advflux_convergence, advective_flux_convergence

def calc_mean_advective_flux_convergence(variable_dict,A,dLambda,dTheta,coslat_h,coslat_v):
    adv_zonal_mean_flux = np.zeros_like(variable_dict['eta']['varmean'])  + np.NaN
    adv_merid_mean_flux = np.zeros_like(variable_dict['eta']['varmean'])  + np.NaN
    adv_total_mean_flux = np.zeros_like(variable_dict['eta']['varmean'])  + np.NaN
    CH = variable_dict['CH']['varmean']
    uvel = variable_dict['u']['varmean']
    vvel = variable_dict['v']['varmean']
    #
    advmx = \
    (-1)*(CH[:,2:].values   * uvel[:,2:].values)    /2/A/dLambda/coslat_h[:,None] + \
         (CH[:,:-2].values  * uvel[:,1:-1].values)  /2/A/dLambda/coslat_h[:,None] + \
    (-1)*(CH[:,1:-1].values * uvel[:,2:].values)    /2/A/dLambda/coslat_h[:,None] + \
         (CH[:,1:-1].values * uvel[:,1:-1].values)  /2/A/dLambda/coslat_h[:,None]
    #
    advmy = \
    (-1)*(CH[2:,:].values   * vvel[2:,:].values)    /2/A/dTheta/coslat_h[1:-1,None] * coslat_v[2:,None]  + \
         (CH[:-2,:].values  * vvel[1:-1,:].values)  /2/A/dTheta/coslat_h[1:-1,None] * coslat_v[1:-1,None]  + \
    (-1)*(CH[1:-1,:].values * vvel[2:,:].values)    /2/A/dTheta/coslat_h[1:-1,None] * coslat_v[2:,None]  + \
         (CH[1:-1,:].values * vvel[1:-1,:].values)  /2/A/dTheta/coslat_h[1:-1,None] * coslat_v[1:-1,None]
    #
    totadvmflux = advmx[1:-1,:] + advmy[:,1:-1]   
    adv_zonal_mean_flux[:,1:-1] = advmx
    adv_merid_mean_flux[1:-1,:] = advmy
    adv_total_mean_flux[1:-1,1:-1] = totadvmflux
    return adv_zonal_mean_flux, adv_merid_mean_flux, adv_total_mean_flux

def calc_eddy_advective_flux_convergence(variable_dict,A,dLambda,dTheta,coslat_h,coslat_v):
    adv_zonal_eddy_flux = np.zeros_like(variable_dict['eta']['varmean'])  + np.NaN
    adv_merid_eddy_flux = np.zeros_like(variable_dict['eta']['varmean'])  + np.NaN
    adv_total_eddy_flux = np.zeros_like(variable_dict['eta']['varmean'])  + np.NaN
    CH = variable_dict['CH']['varmean']
    #
    advex = \
    (-1)*(CH[:,2:].values    *u_star[:,2:].values  )/2/A/dLambda/coslat_h[:,None]     + \
         (CH[:,:-2].values   *u_star[:,1:-1].values)/2/A/dLambda/coslat_h[:,None]   + \
    (-1)*(CH[:,1:-1].values  *u_star[:,2:].values  )/2/A/dLambda/coslat_h[:,None]   + \
         (CH[:,1:-1].values  *u_star[:,1:-1].values)/2/A/dLambda/coslat_h[:,None]
    #
    advey = \
    (-1)*(CH[2:,:].values    *v_star[2:,:].values)  /2/A/dTheta/coslat_h[1:-1,None] * coslat_v[2:,None]   + \
         (CH[:-2,:].values   *v_star[1:-1,:].values)/2/A/dTheta/coslat_h[1:-1,None] * coslat_v[1:-1,None]   + \
    (-1)*(CH[1:-1,:].values  *v_star[2:,:].values)  /2/A/dTheta/coslat_h[1:-1,None] * coslat_v[2:,None]   + \
         (CH[1:-1,:].values  *v_star[1:-1,:].values)/2/A/dTheta/coslat_h[1:-1,None] * coslat_v[1:-1,None]
    #
    totadveflux = advex[1:-1,:] + advey[:,1:-1]   
    adv_zonal_eddy_flux[:,1:-1] = advex
    adv_merid_eddy_flux[1:-1,:] = advey
    adv_total_eddy_flux[1:-1,1:-1] = totadveflux
    return adv_zonal_eddy_flux, adv_merid_eddy_flux, adv_total_eddy_flux

#%% Get the data directories 
url_root_output = swm_utilities.load_data_directories()

#%% Get the land mask
land, ds_topo = swm_utilities.get_topograpy()

#%% Open the datasets
ds_eta = xr.open_dataset('{}eta.nc'.format(url_root_output),decode_times=False)
ds_T1_CH = xr.open_dataset('{}T1_CH.nc'.format(url_root_output),decode_times=False)
ds_T1_C = xr.open_dataset('{}T1_C.nc'.format(url_root_output),decode_times=False)
ds_T1_C0 = xr.open_dataset('{}T1_C0.nc'.format(url_root_output),decode_times=False)
ds_cons = xr.open_dataset('{}T1_CONS.nc'.format(url_root_output),decode_times=False)
ds_forcing = xr.open_dataset('{}T1_FORCING.nc'.format(url_root_output),decode_times=False)
ds_gamma = xr.open_dataset('{}T1_GAMMA_C.nc'.format(url_root_output),decode_times=False)
ds_diff = xr.open_dataset('{}T1_DIFF.nc'.format(url_root_output),decode_times=False)
ds_uhc = xr.open_dataset('{}T1_UHC.nc'.format(url_root_output),decode_times=False)
ds_vhc = xr.open_dataset('{}T1_VHC.nc'.format(url_root_output),decode_times=False)
ds_u = xr.open_dataset('{}u.nc'.format(url_root_output),decode_times=False)
ds_v = xr.open_dataset('{}v.nc'.format(url_root_output),decode_times=False)
ds_mu = xr.open_dataset('{}mu.nc'.format(url_root_output),decode_times=False)
ds_mv = xr.open_dataset('{}mv.nc'.format(url_root_output),decode_times=False)
ds_du = xr.open_dataset('{}du.nc'.format(url_root_output),decode_times=False)
ds_dv = xr.open_dataset('{}dv.nc'.format(url_root_output),decode_times=False)

#%%
datasets_to_load = dict()
# ETA
datasets_to_load['eta'] = dict() 
datasets_to_load['eta']['ds'] = ds_eta
datasets_to_load['eta']['varname'] = 'eta'
# CH
datasets_to_load['CH'] = dict() 
datasets_to_load['CH']['ds'] = ds_T1_CH
datasets_to_load['CH']['varname'] = 'T1_CH'
# C
datasets_to_load['C'] = dict() 
datasets_to_load['C']['ds'] = ds_T1_C
datasets_to_load['C']['varname'] = 'T1_C'
# C0
datasets_to_load['C0'] = dict() 
datasets_to_load['C0']['ds'] = ds_T1_C0
datasets_to_load['C0']['varname'] = 'T1_C0'
# MU
datasets_to_load['mu'] = dict() 
datasets_to_load['mu']['ds'] = ds_mu
datasets_to_load['mu']['varname'] = 'SWM_MU'
# MV
datasets_to_load['mv'] = dict() 
datasets_to_load['mv']['ds'] = ds_mv
datasets_to_load['mv']['varname'] = 'SWM_MV'
# DU
datasets_to_load['du'] = dict() 
datasets_to_load['du']['ds'] = ds_du
datasets_to_load['du']['varname'] = 'SWM_DU'
# DV
datasets_to_load['dv'] = dict() 
datasets_to_load['dv']['ds'] = ds_dv
datasets_to_load['dv']['varname'] = 'SWM_DV'
# U
datasets_to_load['u'] = dict() 
datasets_to_load['u']['ds'] = ds_u
datasets_to_load['u']['varname'] = 'u'
# V
datasets_to_load['v'] = dict() 
datasets_to_load['v']['ds'] = ds_v
datasets_to_load['v']['varname'] = 'v'
# UHC
datasets_to_load['uhc'] = dict() 
datasets_to_load['uhc']['ds'] = ds_uhc
datasets_to_load['uhc']['varname'] = 'T1_uhc'
# VHC
datasets_to_load['vhc'] = dict() 
datasets_to_load['vhc']['ds'] = ds_vhc
datasets_to_load['vhc']['varname'] = 'T1_vhc'
# CONS
datasets_to_load['cons'] = dict() 
datasets_to_load['cons']['ds'] = ds_cons
datasets_to_load['cons']['varname'] = 'T1_cons'
# FORC
datasets_to_load['forc'] = dict() 
datasets_to_load['forc']['ds'] = ds_forcing
datasets_to_load['forc']['varname'] = 'T1_forcing'
# GAMMA
datasets_to_load['gamma'] = dict() 
datasets_to_load['gamma']['ds'] = ds_gamma
datasets_to_load['gamma']['varname'] = 'T1_gamma_C'
# DIFF
datasets_to_load['diff'] = dict() 
datasets_to_load['diff']['ds'] = ds_diff
datasets_to_load['diff']['varname'] = 'T1_diff'

#%% Load the last 80 years of the data
print('Load the last 80 years of the data')
number_of_years = 80
data_choice = dict()
data_choice['time_minidx'] = -number_of_years*12 # last 240 years

variable_dict = dict()
for var2load in datasets_to_load.keys():
    print(var2load)
    variable_dict[var2load] = dict()
    ds_dum       = datasets_to_load[var2load]['ds']
    varname_dum  = datasets_to_load[var2load]['varname']
    # select subset
    if var2load != 'cons' and var2load != 'gamma' and var2load != 'C0':
        selected_t = ds_dum.isel(TIME=slice(data_choice['time_minidx'],None))
    else:
        selected_t = ds_dum.isel(TIME=-1)
    # load the subset and the mean
    if var2load != 'cons' and var2load != 'gamma' and var2load != 'C0':
        variable_dict[var2load]['varmean'] = selected_t.variables[varname_dum].mean(dim='TIME').load()
    else:
        variable_dict[var2load]['varmean'] = selected_t.variables[varname_dum].load()
    variable_dict[var2load]['dataarray'] = selected_t
    print('Calculation for mean of {} done'.format(var2load))

#%% Calculate values for the calculations of gradients etc.
print('Calculate values for the calculations of gradients etc.')
dLambda, dTheta, coslat_v, coslat_h, A = set_up_gradient_arrays(variable_dict)

#%% Calculate thickness weighted averages and bolus velocities
print('Calculate thickness weighted averages and bolus velcoities')
undisturbed_depth = 500.
C_hat = variable_dict['CH']['varmean']/(variable_dict['eta']['varmean']+undisturbed_depth)
u_hat = variable_dict['mu']['varmean']/variable_dict['du']['varmean']
v_hat = variable_dict['mv']['varmean']/variable_dict['dv']['varmean']
u_star = u_hat - variable_dict['u']['varmean']
v_star = v_hat - variable_dict['v']['varmean']

#%% Set up the budget dictionary (Part 1)
print('Set up the budget dictionary (Part 1)')

budget_dict = dict()
# Consumption
consumption = -1*variable_dict['CH']['varmean']*variable_dict['cons']['varmean']
budget_dict['consumption'] = dict()
budget_dict['consumption']['data'] = consumption
budget_dict['consumption']['varname'] = 'Consumption: '+r'$- J\hat{C}\bar{h}$'
# Forcing
forcing = variable_dict['forc']['varmean']
budget_dict['forcing'] = dict()
budget_dict['forcing']['data'] = forcing
budget_dict['forcing']['varname'] = 'Forcing: '+r'$\overline{CF_{\eta}}$'
# Relaxation
relaxation = -1*variable_dict['gamma']['varmean']*(variable_dict['eta']['varmean']+undisturbed_depth)*(C_hat-variable_dict['C0']['varmean'])
budget_dict['relaxation'] = dict()
budget_dict['relaxation']['data'] = relaxation
budget_dict['relaxation']['varname'] = 'Relaxation: '+r'$- \gamma \bar{h} \left(\hat{C}-C_0\right)$'
# Diffusive flux convergence
diffusive_flux_convergence = variable_dict['diff']['varmean']
budget_dict['diffusion'] = dict()
budget_dict['diffusion']['data'] = diffusive_flux_convergence
budget_dict['diffusion']['varname'] = 'Diffusive flux conv.: '+r'$\nabla \cdot \left(\overline{h\kappa_h \nabla C}\right)$'
# Sum of forcing, relaxation and diffusive_flux_convergence
budget_dict['sum_of_forcing_relaxation_diffusion'] = dict()
budget_dict['sum_of_forcing_relaxation_diffusion']['data'] = forcing+relaxation+diffusive_flux_convergence
budget_dict['sum_of_forcing_relaxation_diffusion']['varname'] = 'Sum of other budget terms'

#%% Calculate the advective flux convergence and put into budget_dictionary
print('Calculate the advective flux convergence term.')

uhc_advflux_convergence, vhc_advflux_convergence, advective_flux_convergence = calc_advective_flux_convergence(variable_dict,
                                                                                                               A,
                                                                                                               dLambda,
                                                                                                               dTheta,
                                                                                                               coslat_h,
                                                                                                               coslat_v)
budget_dict['advection'] = dict()
budget_dict['advection']['data'] = advective_flux_convergence
budget_dict['advection']['varname'] = 'Advective flux conv.: '+r'$- \nabla \cdot \left(\overline{h\vec{u}C}\right)$'

#%% Calculate the sum of all terms
print('Calculate the sum of all terms')
sum_of_all = advective_flux_convergence + consumption + relaxation + forcing + diffusive_flux_convergence

budget_dict['sum_of_all'] = dict()
budget_dict['sum_of_all']['data'] = sum_of_all
budget_dict['sum_of_all']['varname'] = 'Sum of all terms'

#%% Now separate the total advective flux into the eddy flux conv., mean flux conv. and the eddy mixing.
print('Separate the total advective flux into the eddy flux conv., mean flux conv. and the eddy mixing.')

#######################################################################
# 1st Term: Mean advection [ = - \nabla \cdot (\overbar{Ch} \bar{u}) ]
#######################################################################
print('-> Mean advection')

adv_zonal_mean_flux, adv_merid_mean_flux, adv_total_mean_flux = calc_mean_advective_flux_convergence(variable_dict,
                                                                                                     A,
                                                                                                     dLambda,
                                                                                                     dTheta,
                                                                                                     coslat_h,
                                                                                                     coslat_v)
budget_dict['adv_meanflux_total'] = dict()
budget_dict['adv_meanflux_total']['data'] = adv_total_mean_flux
budget_dict['adv_meanflux_total']['varname'] = 'Total mean flux conv.: '+r'$- \nabla \cdot \left(\bar{h}\bar{\vec{u}}\hat{C}\right)$'


#####################################################################
# 2nd Term: Eddy advection [ = - \nabla \cdot (\overbar{Ch} u_star)  ]
#####################################################################
print('-> Eddy advection')

adv_zonal_eddy_flux, adv_merid_eddy_flux, adv_total_eddy_flux = calc_eddy_advective_flux_convergence(variable_dict,
                                                                                                     A,
                                                                                                     dLambda,
                                                                                                     dTheta,
                                                                                                     coslat_h,
                                                                                                     coslat_v)
budget_dict['adv_eddyflux_total'] = dict()
budget_dict['adv_eddyflux_total']['data'] = adv_total_eddy_flux
budget_dict['adv_eddyflux_total']['varname'] = 'Total eddy flux conv.: '+r'$- \nabla \cdot \left(\bar{h}\vec{u}^{*}\hat{C}\right)$'

#######################
# 3rd Term: Eddy mixing
#######################
print('-> Eddy mixing')
zonal_eddymix = uhc_advflux_convergence - adv_zonal_mean_flux - adv_zonal_eddy_flux
merid_eddymix = vhc_advflux_convergence - adv_merid_mean_flux - adv_merid_eddy_flux
total_eddymix = zonal_eddymix+merid_eddymix
#
budget_dict['eddymix_total'] = dict()
budget_dict['eddymix_total']['data'] = total_eddymix
budget_dict['eddymix_total']['varname'] = 'Total eddy mixing: '+r'$- \nabla \cdot \left(\overline{h\vec{u}^{\prime\prime}C^{\prime\prime}}\right)$'


# %% Plot tracer budget for simulation
print('Generate the tracer budget plot.')

savefig = False
factor = 1e6
fontsize=18
plt.rcParams['font.size']=fontsize

numrows = 2
numcols = 3
plot_array = np.empty((numrows,numcols), dtype=object)
plot_array[0,0] = budget_dict['advection']
plot_array[0,1] = budget_dict['consumption']
plot_array[0,2] = budget_dict['sum_of_forcing_relaxation_diffusion']
plot_array[1,0] = budget_dict['adv_meanflux_total']
plot_array[1,1] = budget_dict['adv_eddyflux_total']
plot_array[1,2] = budget_dict['eddymix_total']
#

fig, ax = plt.subplots(numrows,numcols,figsize=(18,12),sharex=True,sharey=True)
panel_labs = ['a)','b)','c)','d)','e)','f)']
props = dict(boxstyle='round', facecolor='#EEEEEE', alpha=0.9)

for row in range(np.shape(plot_array)[0]):
    for col in range(np.shape(plot_array)[1]):
        c0 = ax[row,col].contourf(ds_eta.LONGITUDE,ds_eta.LATITUDE,plot_array[row,col]['data']*factor,levels=np.linspace(-1e-6*factor,1e-6*factor,33),cmap='seismic',extend='both')
        ax[row,col].set_title(plot_array[row,col]['varname'],loc='left')
for adx,axi in enumerate(ax.flatten()):
    axi.pcolormesh(ds_topo.LONGITUDE1,ds_topo.LATITUDE,land,cmap=c.ListedColormap(['#777777']))
    for hlineval in [0,20,40]:
        axi.axhline(hlineval,linestyle='--',color='k',alpha=0.2,linewidth=1)
    for vlineval in [-60,-40,-20,0]:
        axi.axvline(vlineval,linestyle='--',color='k',alpha=0.2,linewidth=1)
    axi.set_xticks([-60,-40,-20,0])
    axi.set_xticklabels(['60°W','40°W','20°W','0°'],fontsize=fontsize-2)
    axi.set_yticks([-20,0,20,40])
    axi.set_yticklabels(['20°S','Eq.','20°N','40°N'],fontsize=fontsize-2)
    axi.text(0.04,0.04,panel_labs[adx],transform=axi.transAxes,va='bottom',ha='left',bbox=props,fontsize=fontsize+4)
plt.tight_layout()
plt.subplots_adjust(right=0.9,hspace=0.25,wspace=0.1)
cbax = fig.add_axes([0.91,0.2,0.02,0.6])
cbar = plt.colorbar(c0,cax=cbax,label='ms$^{-1}$')
cbar.ax.set_title(r"$\times$ {:1.0e}".format(1/factor),loc='left',pad=30,fontsize=fontsize)
add_arrows_to_plots(fig,ax[1,0],ax[0,0],"arc3")
add_arrows_to_plots(fig,ax[1,1],ax[0,0],"bar,angle={},fraction={}".format(180,-0.06))
add_arrows_to_plots(fig,ax[1,2],ax[0,0],"bar,angle={},fraction={}".format(180,-0.06/2))
if savefig == True:
    figname = '../plots/fig_tracer_budget'
    plt.savefig('{}.png'.format(figname),dpi=250)
    os.system('convert {}.png {}.pdf'.format(figname,figname))
plt.show()

#%%