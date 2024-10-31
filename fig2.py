#%%
import numpy as np
from netCDF4 import Dataset
import xarray as xr
from datetime import datetime
import pandas as pd
import pickle
import pymannkendall as mk
import scipy.stats as stats
from sklearn.linear_model import LinearRegression


# plot libaray
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from cartopy.io.shapereader import Reader
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import cmaps
import seaborn as sns
import matplotlib.transforms as mtransforms
from matplotlib.ticker import ScalarFormatter

# 
def mk_trend_ve(x):
    if np.isnan(x).sum() > 25:
        return (np.nan ,np.nan)
    else :
        mk_result = mk.original_test(x)
        slope = mk_result.slope
        p = mk_result.p
        return (slope ,p)

def xarray_weighted_mean(data_array):

    weights = np.cos(np.deg2rad(data_array.lat))
    weights.name = "weights"
    data_weighted = data_array.weighted(weights)
    weighted_mean = data_weighted.mean(("lon", "lat"))

    return weighted_mean

def add_map_nanhai(ax,cf_china,cf_9line):

    ax.add_feature(cf_china, edgecolor='black')
    ax.add_feature(cf_9line, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), edgecolor='#626063')
    ax.set_extent([105.8,122,2,25], crs=ccrs.PlateCarree())  # nanhai

def add_map_feature(ax, cf_china, cf_9line):
    # ax.add_feature(cfeature.COASTLINE.with_scale('110m'), edgecolor='#626063')
    # # ax.add_feature(cf_coastial_land_200km)
    # ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
    # ax.set_yticks(np.arange(-60, 90, 30), crs=ccrs.PlateCarree())
    # ax.xaxis.set_major_formatter(LongitudeFormatter())
    # ax.yaxis.set_major_formatter(LatitudeFormatter())
    # ax.set_extent([-180, 180, -65, 90], crs=ccrs.PlateCarree())
    # ax.axes.set_xlabel('')
    # ax.axes.set_ylabel('')

    ax.add_feature(cf_china, edgecolor='black')
    ax.add_feature(cf_9line, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), edgecolor='#626063')
    lb = ax.gridlines(draw_labels=False, xlocs=range(0, 180, 10), ylocs=range(0, 90, 10), linestyle=(0, (10, 10)),
                        linewidth=0.25, color='dimgrey', alpha=0.8, zorder=4)
    lb = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False, xlocs=range(80, 135, 10),
                        ylocs=range(0, 51, 10), linewidth=0.1, color='gray', alpha=0.8, linestyle='--')
    lb.top_labels = None
    lb.right_labels = None
    lb.left_labels = None
    lb.rotate_labels = False

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = '19'

cf_china = cfeature.ShapelyFeature(
    Reader(r'/home/zq2/wg/code/shp/china.shp').geometries(),
    ccrs.PlateCarree(),
    edgecolor='gray',
    facecolor='none'
)
cf_9line = cfeature.ShapelyFeature(
    Reader(r'/home/zq2/wg/code/shp/9lines_hainan_taiwan.shp').geometries(),
    ccrs.PlateCarree(),
    edgecolor='gray',
    facecolor='none'
)


def xarray_weighted_mean(data_array):

    weights = np.cos(np.deg2rad(data_array.lat))
    weights.name = "weights"
    data_weighted = data_array.weighted(weights)
    weighted_mean = data_weighted.mean(("lon", "lat"))

    return weighted_mean
#%%
# Trend of landfalling hws
data_landfalling_exposure_trend = xr.apply_ufunc(
    mk_trend_ve,
    ds_ocean_onto_land_hws['exposure'].where(da_china_mask, drop=True).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)

data_landfalling_frequency_trend = xr.apply_ufunc(
    mk_trend_ve,
    ds_ocean_onto_land_hws['frequency'].where(da_china_mask, drop=True).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)
data_landfalling_extent_trend = xr.apply_ufunc(
    mk_trend_ve,
    ds_ocean_onto_land_hws['extent'].where(da_china_mask, drop=True).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)
data_landfalling_cum_heat_trend = xr.apply_ufunc(
    mk_trend_ve,
    ds_ocean_onto_land_hws['cum_heat_grid'].where(da_china_mask, drop=True).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)
data_landfalling_mean_intensity_trend = xr.apply_ufunc(
    mk_trend_ve,
    (ds_ocean_onto_land_hws['cum_heat_grid'].where(da_china_mask, drop=True) / \
     ds_ocean_onto_land_hws['exposure'].where(da_china_mask, drop=True)).rolling(time=10, center=True, min_periods=1).mean(),
    input_core_dims = [['time']],
    output_core_dims = [[],[]],
    vectorize=True
)

lons = data_landfalling_exposure_trend[0].lon.data
lats = data_landfalling_exposure_trend[0].lat.data
meshgrid_lon, meshgrid_lat = np.meshgrid(lons, lats)
#%% plot
fig = plt.figure(figsize=[18,18])
proj_china=ccrs.LambertConformal(central_longitude=107.5,central_latitude=30.0,standard_parallels=(20,50))
# ax1 = fig.add_axes([0.075, 0.75, 0.24, 0.14], frameon=True)
# ax2 = fig.add_axes([0.4, 0.75, 0.58, 0.14], frameon=True)

# ax3 = fig.add_axes([0.075, 0.45, 0.88, 0.31], frameon=True,projection=ccrs.PlateCarree(central_longitude =150.0))#上1

ax1 = fig.add_axes([0.05, 0.43, 0.29, 0.4], frameon=True,projection=proj_china) #中1
ax1_si = fig.add_axes([0.271, 0.494, 0.086, 0.07], frameon=True,projection=proj_china)
ax2 = fig.add_axes([0.37, 0.43, 0.29, 0.4], frameon=True,projection=proj_china) #中2
ax2_si = fig.add_axes([0.591, 0.494, 0.086, 0.07], frameon=True,projection=proj_china)
ax3 = fig.add_axes([0.69, 0.43, 0.29, 0.4], frameon=True,projection=proj_china) #中2
ax3_si = fig.add_axes([0.911, 0.494, 0.086, 0.07], frameon=True,projection=proj_china)

ax4 = fig.add_axes([0.245, 0.1, 0.29, 0.4], frameon=True,projection=proj_china) #中1
ax4_si = fig.add_axes([0.466, 0.164, 0.086, 0.07], frameon=True,projection=proj_china)
ax5 = fig.add_axes([0.565, 0.1, 0.29, 0.4], frameon=True,projection=proj_china) #中2
ax5_si = fig.add_axes([0.786, 0.164, 0.086, 0.07], frameon=True,projection=proj_china)
# ax6 = fig.add_axes([0.69, 0.1, 0.29, 0.4], frameon=True,projection=proj_china) #中2
# ax6_si = fig.add_axes([0.911, 0.164, 0.086, 0.08], frameon=True,projection=proj_china)

#ax1
ax1.cla()
(data_landfalling_exposure_trend[0]*10).plot(
    ax=ax1,levels=np.arange(-2,2.2,0.2),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.85,orientation='horizontal',label='Exposure (days/10yr)',pad=0.1)
)
ax1.add_feature(cf_china,edgecolor='black')
ax1.add_feature(cf_9line,edgecolor='black')
ax1.add_feature(cfeature.COASTLINE.with_scale('110m'),edgecolor='#626063')
# ax4.add_feature(cfeature.LAND)
lb1=ax1.gridlines(draw_labels=False, xlocs=range(0,180,10), ylocs=range(0,90,10), linestyle=(0,(10,10)), linewidth=0.25, color='dimgrey', alpha=0.8, zorder=4)
lb1=ax1.gridlines(draw_labels=True,x_inline=False, y_inline=False,xlocs=range(80,135,10), ylocs=range(20,51,10),linewidth=0.1, color='gray', alpha=0.8, linestyle='--' )
lb1.top_labels = None
lb1.right_labels = None
lb1.rotate_labels = False

dot_area = np.where(data_landfalling_exposure_trend[1] < 0.05)
ax1.scatter(meshgrid_lon[dot_area], meshgrid_lat[dot_area],color='k',s=0.2,linewidths=0.2,transform=ccrs.PlateCarree()) #绘制散点图(即打点)

(data_landfalling_exposure_trend[0]*10).plot(
    ax=ax1_si,levels=np.arange(-1,1.1,0.1),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    add_colorbar=False)
add_map_nanhai(ax1_si, cf_china, cf_9line)

# ax2
ax2.cla()
(data_landfalling_frequency_trend[0]*10).plot(
    ax=ax2,levels=np.arange(-1,1.1,0.1),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.85,orientation='horizontal',label='Frequency (events/10yr)',ticks=np.arange(-1,1.1,0.4),pad=0.1)
)
add_map_feature(ax2, cf_china, cf_9line)

dot_area = np.where(data_landfalling_frequency_trend[1] < 0.05)
ax2.scatter(meshgrid_lon[dot_area], meshgrid_lat[dot_area],color='k',s=0.2,linewidths=0.2,transform=ccrs.PlateCarree()) #绘制散点图(即打点)

(data_landfalling_frequency_trend[0]*10).plot(
    ax=ax2_si,levels=np.arange(-1,1.1,0.1),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    add_colorbar=False)
add_map_nanhai(ax2_si, cf_china, cf_9line)

# ax3
ax3.cla()
formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((0, 0))
(data_landfalling_extent_trend[0]*10).plot(
    ax=ax3,levels=np.arange(-1000000,1100000,100000),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.85,orientation='horizontal',label='Extent (km$^2$/10yr)',format=formatter,pad=0.1)
)
dot_area = np.where(data_landfalling_extent_trend[1] < 0.05)
ax3.scatter(meshgrid_lon[dot_area], meshgrid_lat[dot_area],color='k',s=0.2,linewidths=0.2,transform=ccrs.PlateCarree()) #绘制散点图(即打点)

add_map_feature(ax3, cf_china, cf_9line)
(data_landfalling_extent_trend[0]*10).plot(
    ax=ax3_si,levels=np.arange(-1000000,1100000,100000),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    add_colorbar=False)
add_map_nanhai(ax3_si, cf_china, cf_9line)

# ax4
ax4.cla()
(data_landfalling_cum_heat_trend[0]*10).plot(
    ax=ax4,levels=np.arange(-2,2.1,0.2),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.85,orientation='horizontal',label='Cumulative Heat (°C/10yr)',ticks=np.arange(-2,2.1,1),pad=0.1)
)
ax4.add_feature(cf_china,edgecolor='black')
ax4.add_feature(cf_9line,edgecolor='black')
ax4.add_feature(cfeature.COASTLINE.with_scale('110m'),edgecolor='#626063')
# ax4.add_feature(cfeature.LAND)
lb4=ax4.gridlines(draw_labels=False, xlocs=range(0,180,10), ylocs=range(0,90,10), linestyle=(0,(10,10)), linewidth=0.25, color='dimgrey', alpha=0.8, zorder=4)
lb4=ax4.gridlines(draw_labels=True,x_inline=False, y_inline=False,xlocs=range(80,135,10), ylocs=range(20,51,10),linewidth=0.1, color='gray', alpha=0.8, linestyle='--' )
lb4.top_labels = None
lb4.right_labels = None
lb4.rotate_labels = False

dot_area = np.where(data_landfalling_cum_heat_trend[1] < 0.05)
ax4.scatter(meshgrid_lon[dot_area], meshgrid_lat[dot_area],color='k',s=0.2,linewidths=0.2,transform=ccrs.PlateCarree()) #绘制散点图(即打点)

(data_landfalling_cum_heat_trend[0]*10).plot(
    ax=ax4_si,levels=np.arange(-2,2.1,0.2),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    add_colorbar=False)
add_map_nanhai(ax4_si, cf_china, cf_9line)

# ax4
ax5.cla()
(data_landfalling_mean_intensity_trend[0]*10).plot(
    ax=ax5,levels=np.arange(-0.5,0.55,0.05),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    cbar_kwargs=dict(fraction=0.06, shrink=0.85,orientation='horizontal',label='Intensity (°C*day$^-1$/10yr)',ticks=np.arange(-0.4,0.44,0.2),pad=0.1)
)
add_map_feature(ax5, cf_china, cf_9line)

dot_area = np.where(data_landfalling_mean_intensity_trend[1] < 0.05)
ax5.scatter(meshgrid_lon[dot_area], meshgrid_lat[dot_area],color='k',s=0.2,linewidths=0.2,transform=ccrs.PlateCarree()) #绘制散点图(即打点)

(data_landfalling_mean_intensity_trend[0]*10).plot(
    ax=ax5_si,levels=np.arange(-0.5,0.55,0.05),cmap='RdYlBu_r',transform = ccrs.PlateCarree(),
    add_colorbar=False)
add_map_nanhai(ax5_si, cf_china, cf_9line)

ax1.text(0,1.03,'a',fontweight='bold',transform=ax1.transAxes,fontsize=25)
ax2.text(0,1.03,'b',fontweight='bold',transform=ax2.transAxes,fontsize=25)
ax3.text(0,1.03,'c',fontweight='bold',transform=ax3.transAxes,fontsize=25)
ax4.text(0,1.03,'d',fontweight='bold',transform=ax4.transAxes,fontsize=25)
ax5.text(0,1.03,'e',fontweight='bold',transform=ax5.transAxes,fontsize=25)

ax1.text(0.02,0.04,'Mean:1.40*',transform=ax1.transAxes,fontsize=20)
ax2.text(0.02,0.04,'Mean:0.60*',transform=ax2.transAxes,fontsize=20)
ax3.text(0.02,0.04,'Mean:5.32*',transform=ax3.transAxes,fontsize=20)
ax4.text(0.02,0.04,'Mean:1.39*',transform=ax4.transAxes,fontsize=20)
ax5.text(0.02,0.04,'Mean:0.03*',transform=ax5.transAxes,fontsize=20)
fig.show()

fig_savepath = '/home/zq2/wg/code/src/plot/china/part1/figure/'

fig.savefig(fig_savepath+'Fig2_Trend_of_hws.png',dpi=600)