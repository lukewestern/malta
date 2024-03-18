# Derive transport for 2D model from 3D GEOS-Chem fields.
# This requires the user to output GEOS-Chem fields following the appropriate 
# formatting.
# Users should change the directories below to the user's own directory locations/structure.
# Required packages are not installed during the build of the main package - these must be done 
# manually to require fewer package installs for main users of the model itself.
#
# Approach to take:
# 1) Run GEOS-Chem tracer experiments for desired years
# 2) Run 'run_derive_transport.sh' (or equivelent scheduler) for all months/years. 
#    This runs derive_transport __main__, the main function for deriving monthly transport.
# 3) Run function 'make_2D_yearly_files' in this script to save yearly transport files. 
#
# TODO: Interpolation in __main__ can be main neater using interpolate_T function. 
#
import xarray as xr
import numpy as np
import pandas as pd
import xesmf as xe
from xgcm import Grid
import sys
import glob
from malta import utils
from scipy.interpolate import interp2d
from scipy.interpolate import pchip
import os

model_trans_dir = utils.get_transport_directory() # Main directory for yearly transport files to be saved to
weight_folder = "/user/home/lw13938/work/TwoDmodel/weights" # Directory to save xesmf weights (stops conflicts)
monthly_transport = "/user/home/lw13938/work/TwoDmodel/TransportParameters" # Directory to save temporary monthly netcdf transport files
filepath = "/user/home/lw13938/work/GCClassic.13.3.4/rundirs/Tracer_outputs/" # Directory containing GEOS-Chem outputs
gc_metpath = "/user/home/lw13938/shared/GEOS_CHEM/data/ExtData/GEOS_4x5/MERRA2/" # Directory contain GEOS-Chem met inputs

def interpolate_T(ds_met, var, zb3D, lon, zx2D, latx2D):
    """
    Interpolate dataset variable from 3D to 2D grid using xesmf

    Args:
        ds_met (dataset): xarray dataset containing variable to interpolate
        var (string): variable to interpolate
        zb3D (array): Scale height of 3D model
        lon (array): Longitude of 3D model
        zx2D (array): Scale height of 2D model
        latx2D (array): Latitude of 2D model

    Returns:
        DataArray: xarray DataArray of interpolated variable
    """
    dsT_ = ds_met[var]
    dsT_.coords['z'] = zb3D  
    dsT = dsT_.swap_dims({'lev': 'z'})
    ds_out = xr.Dataset({"lat": (["lat"], latx2D),
            "lon": (["lon"], lon),
            "z":  (["z"], zx2D),})
    regridder = xe.Regridder(dsT, ds_out, "bilinear", filename=f"{weight_folder}/bilinear_{str(int(np.random.random()*100))}_{month}.nc") 
    dsT_out = regridder(dsT).to_dataset(name="Met_T").drop('lev')
    grid = Grid(dsT_out, coords={'z': {'center':'z'}}, periodic=False)
    dsTt = grid.transform(dsT_out[var], 'z', zx2D, target_data=dsT_out.z)

    return dsTt


def interpolate_cflux(ds_met, var, zb3D, lon, zx2D, latx2D):
    """
    Interpolate dataset variable from 3D to 2D grid using xesmf.
    Basically the same as interpolate_T but need some slight modifications.

    Args:
        ds_met (dataset): xarray dataset containing variable to interpolate
        var (string): variable to interpolate
        zb3D (array): Scale height of 3D model
        lon (array): Longitude of 3D model
        zx2D (array): Scale height of 2D model
        latx2D (array): Latitude of 2D model

    Returns:
        DataArray: xarray DataArray of interpolated variable
    """
    dsT_ = ds_met[var]
    dsT_ = dsT_.assign_coords(z=('lev',zb3D))
    dsT = dsT_.swap_dims({'lev': 'z'})
    ds_out = xr.Dataset({"lat": (["lat"], latx2D),
            "lon": (["lon"], lon),
            "z":  (["z"], zx2D),})
    regridder = xe.Regridder(dsT, ds_out, "bilinear", filename=f"{weight_folder}/bilinear_{str(int(np.random.random()*100))}_{month}.nc") 
    dsT_out = regridder(dsT).to_dataset(name="cflux").drop('lev')
    grid = Grid(dsT_out, coords={'z': {'center':'z'}}, periodic=False)
    dsTt = grid.transform(dsT_out["cflux"], 'z', zx2D, target_data=dsT_out.z)

    return dsTt


def pchip2(x, y, z, xi, yi):
    """
    P-chip interpolation on 2-D. x, y, xi, yi should be 1-D
    and z.shape == (len(x), len(y))

    Taken from: https://scipy-user.scipy.narkive.com/FG5DVM1l/2-d-data-interpolation
    This ensures monoticity for interpolation (splines can over/undershoot)
    """

    z2 = np.empty([len(xi), len(y)], dtype=z.dtype)
    for k in range(len(y)):
        z2[:,k] = pchip(x, z[:,k])(xi)

    z3 = np.empty([len(xi), len(yi)], dtype=z.dtype)
    for k in range(len(xi)):
        z3[k,:] = pchip(y, z2[k,:])(yi)

    return z3


def infer_eddy_agrid(dsvqt, dsvt,dswqt, dswt, y2D, ym2D):
    """    
    This infers the eddy trasport tensor following the approach of Bachmann et al. (2015)
    Input are data arrays of v and w winds, and dataset of transport tracers all interpolated
    to the 2D model resolution.

    Args:
        dsvqt (dataset): Concentrations on same grid as v-field
        dsvt (dataset): Met data on same grid as v-field
        dswqt (dataset): Concentrations on same grid as w-field
        dswt (dataset): Met data on same grid as w-field
        y2D (array): y-coordinate at grid cell edges
        ym2D (array): y-coordinate at grid cell mid-points

    Returns:
        list: Lists containing the eddy flux tensor and gradient flux fields
    """
    v = dsvt
    w = dswt

    qvb_arr = []
    qvbp_arr = []
    dqvb_dz = []
    dqvb_dy = []
    qwb_arr = []
    qwbp_arr = []
    dqwb_dz = []
    dqwb_dy = []
    for i in range(1,7):
        qv = dsvqt[f"SpeciesConc_COInfTracer{i}"]
        qw = dswqt[f"SpeciesConc_COInfTracer{i}"]
        qvb_arr.append(qv.mean(dim=("lon","time")))
        qwb_arr.append(qw.mean(dim=("lon","time")))
        qvbp_arr.append(  ( (v - v.mean(dim=("lon","time")))*(qv - qv.mean(dim=("lon","time")))).mean(dim=("lon","time")))
        qwbp_arr.append(  ( (w - w.mean(dim=("lon","time")))*(qw - qw.mean(dim=("lon","time")))).mean(dim=("lon","time")))
        # Work out dq/dy and dq/dz
        delqvb = np.gradient(qv.mean(dim=("lon","time")).values, qv.z.values, y2D,edge_order=2)
        dqvb_dz.append(delqvb[0])
        dqvb_dy.append(delqvb[1])
        delqwb = np.gradient(qw.mean(dim=("lon","time")).values, qw.z.values, ym2D,edge_order=2)
        dqwb_dz.append(delqwb[0])
        dqwb_dy.append(delqwb[1])

    Fy = np.array([Fv.values for Fv in qvbp_arr])*cosphi
    Fz = np.array([Fw.values for Fw in qwbp_arr])
    Dvyz =  np.array([ np.array([dqdy for dqdy in dqvb_dy]) ,
                    np.array([dqdz for dqdz in dqvb_dz])])
    Dwyz =  np.array([ np.array([dqdy for dqdy in dqwb_dy]) ,
                    np.array([dqdz for dqdz in dqwb_dz])])
                    
    Kyy = np.zeros_like(dqvb_dz[0])
    Kyz = np.zeros_like(dqvb_dz[0])
    Kzz = np.zeros_like(dqwb_dz[0])
    Kzy = np.zeros_like(dqwb_dz[0])
    qvbbarr = np.c_[[qi for qi in qvb_arr]]
    qwbbarr = np.c_[[qi for qi in qwb_arr]]
    for i in range(len(qv.z.values)):
        for j in range(len(y2D)):
            isigv = np.diag(1/(qvbbarr[:,i,j]/qvbbarr[:,i,j].sum())**2)
            K_y = np.linalg.inv(Dvyz[:,:,i,j] @ isigv @ Dvyz[:,:,i,j].T) @ Dvyz[:,:,i,j] @ isigv @ np.expand_dims(-Fy[:,i,j],1)
            Kyy[i,j] = K_y[0]
            Kyz[i,j] = K_y[1]
    for i in range(len(qw.z.values)):
        for j in range(len(ym2D)):
            isigw = np.diag(1/(qwbbarr[:,i,j]/qwbbarr[:,i,j].sum())**2)
            K_z = np.linalg.inv(Dwyz[:,:,i,j] @ isigw @ Dwyz[:,:,i,j].T) @ Dwyz[:,:,i,j] @ isigw @ np.expand_dims(-Fz[:,i,j],1)
            Kzz[i,j] = K_z[1]
            Kzy[i,j] = K_z[0]

    return [Kzz, Kyy, Kyz, Kzy], [Fz,Fy], [Dwyz,Dvyz]

def validation_tracer_agrid(dsvqt, dsvt,dswqt, dswt, tn, y2D, ym2D):
    """
    Flux-gradient calculation if wishing to validate the eddy-flux derivation.
    Note, this is not currently used.

    Args:
        dsvqt (dataset): Concentrations on same grid as v-field
        dsvt (dataset): Met data on same grid as v-field
        dswqt (dataset): Concentrations on same grid as w-field
        dswt (dataset): Met data on same grid as w-field
        tn (int): Tracer number used for validation
        y2D (array): y-coordinate at grid cell edges
        ym2D (array): y-coordinate at grid cell mid-points

    Returns:
        arrays: Gradient flux fields.
    """
    v = dsvt
    w = dswt

    qv = dsvqt[f"SpeciesConc_COInfTracer{tn}"]
    qw = dswqt[f"SpeciesConc_COInfTracer{tn}"]
    Fy = ( (v - v.mean(dim=("lon","time")))*(qv - qv.mean(dim=("lon","time")))).mean(dim=("lon","time"))
    Fz = ( (w - w.mean(dim=("lon","time")))*(qw - qw.mean(dim=("lon","time")))).mean(dim=("lon","time"))

    # Work out dq/dy and dq/dz
    delqvb = np.gradient(qv.mean(dim=("lon","time")).values, qv.z.values, y2D, edge_order=2)
    delqwb = np.gradient(qw.mean(dim=("lon","time")).values, qw.z.values, ym2D, edge_order=2)
    Dvyz = [delqvb[1], delqvb[0]]
    Dwyz = [delqwb[1], delqwb[0]]

    return [Fz.values,Fy.values*cosphi], [Dwyz,Dvyz]

def smoothKzz(Kz, limit=50.0, lower=False):
    """    
    Kzz is hard to derive so smooth over unrealistically high diffusion.
    Kzz is generally higher in the lower atmosphere. Set upper Kzz limit
    to 50 m2/s (Plumb and Mahlman max is around 20) as upper limit, 
    and reset as the average of surrounding grid squares.

    Args:
        Kz (array): Kzz component of eddy-transport tensor
        limit (float, optional): Upper limit of Kzz. Defaults to 50.
        lower (bool, optional): True if setting a lower limit. Defaults to False.

    Returns:
        array: Smoothed Kzz tensor component.
    """
    if lower:
        Kz[Kz < limit] = limit
    else:
        Kz[Kz > limit] = limit
    for i in range(Kz.shape[0]):
        for j in range(Kz.shape[1]):
            if np.isclose(Kz[i,j], limit):
                Kzav = 0
                avdn = 0
                for cnti in [1,-1, 0]:
                    for cntj in [1,-1, 0]:
                        if cnti == 0 and cntj == 0:
                            continue
                        if (i + cnti) < 0 or (i + cnti) >= (Kz.shape[0]):
                            continue 
                        if (j + cntj) < 0 or (j + cntj) >= (Kz.shape[1]):
                            continue 
                        if np.isclose(Kz[i+cnti, j+cntj], limit):
                            continue
                        Kzav += Kz[i+cnti, j+cntj]
                        avdn += 1
                Kz[i,j] = Kzav/avdn
    return Kz

def correct_eddy(K, zm2D, z2D, ym2D, y2D):
    """
    Kyy can't be negative. Put in check/correction as Plumb & Mahlman 1986

    Args:
        K (list): Eddy-transport tensor
        zm2D (array): z-coordinate mid-points in 2D model
        z2D (array): z-coordinate edges in 2D model
        ym2D (array): y-coordinate mid-points in 2D model
        y2D (array): y-coordinate edges in 2D model

    Returns:
        list, arrays: Corrected Eddy-transport tensor, the Kzy tensor component 
                      interpolated to other edges, diffusivity on different grids
    """
    kyycondflag = 0
    for i in range(len(zm2D)):
        kyycond = K[1][i,:] < 0 
        K[1][i, kyycond] = 1e4 * np.cos(lat2D[kyycond])**2
        kyycondflag += kyycond.sum()
    if kyycondflag > 0:
        print("Kyy had at least one negative value")
    # Limit Dyy for large values than can appear at the poles
    K[1] = smoothKzz(K[1], limit=1e7)
    # Smooth Kzy as can also have anomalous points so slightly smooth
    K[3] = smoothKzz(K[3], limit=np.percentile(K[3], 99.))
    K[3] = smoothKzz(K[3], limit=np.percentile(K[3], 1.), lower=True)
    K[2] = smoothKzz(K[2], limit=np.percentile(K[2], 99.))
    K[2] = smoothKzz(K[2], limit=np.percentile(K[2], 1.), lower=True)
    # This condition must be satisfied (diffusion - or symmetric - tensor component 
    # must be positive definite) so we'll modify accordingly.
    # Need Kyy and Kyz on Kzz's grid to evaluate, so just do a 2D interpolation
    Kyy_zgrid = pchip2(zm2D,y2D, K[1], z2D,ym2D )
    Kzy_ygrid_f = interp2d(ym2D, z2D, K[3], kind='linear')
    Kzy_ygrid = Kzy_ygrid_f(y2D, zm2D)
    Kyz_zgrid_f = interp2d(y2D, zm2D, K[2], kind='linear')
    Kyz_zgrid = Kyz_zgrid_f(ym2D, z2D)
    Dzy = (Kyz_zgrid+K[3])/2 
    Dyz = (Kzy_ygrid+K[2])/2 
    for i in [0,-1]:
        Kyz_zgrid[i,:] = 0.
        Dzy[i,:] = 0.
        Kzy_ygrid[:,i] = 0.
        Dyz[:,i] = 0.
    cond = Dzy**2 > (K[0]*Kyy_zgrid) 
    K[0][cond] = (Dzy[cond]**2 / Kyy_zgrid[cond])
    # Kzy and Kzz are jointly inferred. If Kzz is still unrealistic, estimate
    # from surrounding grid cells and then correct Kzy accordingly.
    # Then need to recalculate Dyz Dzy.
    K[0] = smoothKzz(K[0])
    # Ensure checks still hold and adjust Kzy if necessary 
    kyzcond = Dzy**2 > (K[0]*Kyy_zgrid)
    K[3][kyzcond] = ( ( K[0][kyzcond] * Kyy_zgrid[kyzcond] )**0.5 - abs(Kyz_zgrid[kyzcond]) )*np.sign(K[3][kyzcond]) 
    Kzy_ygrid_f = interp2d(ym2D, z2D, K[3], kind='linear')
    Kzy_ygrid = Kzy_ygrid_f(y2D, zm2D)
    Dzy = (Kyz_zgrid+K[3])/2 
    Dyz = (Kzy_ygrid+K[2])/2 
    # Ensure edge values are zero.
    for i in [0,-1]:
        Kyz_zgrid[i,:] = 0.
        Dzy[i,:] = 0.
        Kzy_ygrid[:,i] = 0.
        Dyz[:,i] = 0.
    kyzcond = Dzy**2 > (K[0]*Kyy_zgrid)
    if kyzcond.sum() > 0:
        print("Warning: Dzy is too large for at least one grid cell!")
    else:
        print("Dzy has been corrected")

    for i in [0,-1]:
        K[0][i,:] = 0.
        K[3][i,:] = 0.
        K[1][:,i] = 0.
        K[2][:,i] = 0.
    
    # Under current set-up, only Dzy is needed and defined on a c-grid
    Dzy_cgrid_f = interp2d(ym2D, z2D, Dzy, kind='linear')
    Dzy = Dzy_cgrid_f(ym2D, zm2D)

    return K, Kyz_zgrid, Kzy_ygrid, Dzy, Dyz

def derive_ustar(K, Kzy_ygrid, Kyz_zgrid, zm2D, ym2D):
    """
    Take anti-symmetric part of eddy transport tensor to derive 
    residual advective transport. 

    Args:
        K (list): Eddy-transport tensor.
        Kzy_ygrid (array): Kzy component on v-wind grid. 
        Kyz_zgrid (array): Kzy component on w-wind grid. 
        zm2D (array): z-coordinate mid-points in 2D model.
        ym2D (array): y-coordinate mid-points in 2D model.

    Returns:
        array: residual v and w advection.
    """
    # 
    Psi1_y = (K[2]-Kzy_ygrid)/2
    Psi1_z = (K[3]-Kyz_zgrid)/2 
    v_star = -np.gradient(Psi1_y, zm2D, axis=0, edge_order=2)
    w_star = np.gradient(Psi1_z, ym2D, axis=1, edge_order=2)
    for i in [0,-1]:
        v_star[:,i] = 0.
        w_star[i,:] = 0
    
    return v_star, w_star

def wstar_potentialtemp(dsTwt,dsvwt, H):
    """Calculate w* (eddy advective component) using potential temperature (not used)"""
    kt = 2./7.
    dsThetaw = dsTwt*np.exp(kt*dsTwt.z.values/H)
    dThetawdz = np.gradient(dsThetaw.mean(("time","lon")).values, dsThetaw.z.values, axis=1, edge_order=2)
    vw = dsvwt
    vTb =  ( (vw - vw.mean(dim=("lon","time")))*(dsThetaw - dsThetaw.mean(dim=("lon","time")))).mean(dim=("lon","time"))
    costvtbdtdz = np.expand_dims(np.cos(dsThetaw.lat.values*np.pi/180),1)*vTb.values/dThetawdz
    w_star = np.transpose(np.expand_dims(1/(R*np.cos(dsThetaw.lat.values*np.pi/180)),1) * np.gradient(costvtbdtdz, dsThetaw.lat.values, axis=0, edge_order=2))
    w_star[0,:] = 0.
    w_star[-1,:] = 0.

    return w_star

def make_nondivergent(w_in, v_in, cosc, cose, H, z, zm, y, dz, dy):
    """
    Ensure residual wind fields are non-divergence. 
    This is lost through interpolation etc.

    Args:
        w_in (array): Divergent w-wind.
        v_in (array): Divergent v-wind.
        cosc (_type_): Cosine of latitude at grid centres. 
        cose (_type_): Cosine of latitude at grid edges.
        H (float): Scale height
        z (array): z-coordinate edge points in 2D model.
        zm (array): z-coordinate mid-points in 2D model.
        y (array): y-coordinate edge points in 2D model.
        dz (array): Grid cell heights.
        dy (array): Grid cell widths.

    Returns:
        arrays: Non-divergent w and v wind fields.
    """
    # Adjust residual winds using Shine 1989 approach  
    w = (w_in - np.expand_dims(np.sum(w_in*cosc,1)/np.sum(cosc),1))
    # Then take a mass-conservation approach to derive appropriate horizontal winds:
    v = v_in.copy()
    for k in range(len(zm)):
        for j in range(1,len(y)-1):
            v[k,j] = ( v[k,j-1]*cose[j-1] - (w[k+1,j-1]*np.exp(-z[k+1]/H) - w[k,j-1]*np.exp(-z[k]/H)) / (np.exp(-zm[k]/H)*dz[k]) *cosc[j-1]*dy[0] ) / cose[j] 
        
    return w, v

def add_convection(ds2d):
    """
    Add convective parameters straight from MERRA2 met data
    
    Args:
        ds2d (xarray dataset): The monthly dataset of 2D transport to which
                               the convection will be added
                               
    Returns:
        xarray dataarray: A Data Array of the 2D monthly convective flux
    """
    tdtime = pd.to_datetime(ds2d.time.values)
    year = tdtime.year
    m = tdtime.month
    adpd = os.path.dirname(os.path.realpath(__file__))
    dfzp = pd.read_csv(f"{adpd}/merra2_zpressure.dat", delim_whitespace=True)
    zmerra = 7200*np.log(1000./dfzp["pressure"][1::2].values)[::-1]
  
    fns = sorted(glob.glob(f"{gc_metpath}/{year}/{str(m).zfill(2)}/MERRA2.*.A3dyn.4x5.nc4"))
    dsconv = xr.open_mfdataset(fns,concat_dim="time", combine='nested')
    cvmi = interpolate_cflux(dsconv, "DTRAIN", zmerra, dsconv.lon.values, ds2d.zm.values, ds2d.latm.values).transpose("time","z","lat","lon").mean(("lon")).resample(time="MS").mean()
    ds_cv = cvmi/(1e-3*28.9647)
    return xr.DataArray(ds_cv.assign_coords({"lat":ds2d.ym.values}).rename(lat="ym").rename(z="zm"))



def make_2D_yearly_files(start_year, end_year = None):
    """Take monthly 2D transport and make CF compliant 3D netcdf files"""
    coord_dict = {
    "y" : {"long_name":"distance from equator at cell boundary","units":"m", "standard_name":"none" },
    "ym": {"long_name":"distance from equator at cell midpoint","units":"m", "standard_name":"none" },
    "z" : {"long_name":"height at cell boundary","units":"m", "standard_name":"height"},
    "zm": {"long_name":"height at cell midpoint","units":"m", "standard_name":"height"},
    "time":{"long_name":"time","standard_name":"time"}
    }
    vars_dict = {
        'Dzz'  : {"long_name":"vertical eddy diffusivity","units":"m2 s-1", "standard_name":"none"},
        'Dyy'  : {"long_name":"horizontal eddy diffusivity","units":"m2 s-1", "standard_name":"none"},
        'Dzy'  : {"long_name":"off-diagonal eddy diffusivity","units":"m2 s-1", "standard_name":"none"},
        'v'    : {"long_name":"horizontal wind component","units":"m s-1", "standard_name":"northward_wind"},
        'w'    : {"long_name":"vertical wind component","units":"m s-1", "standard_name":"upward_air_velocity"},
        'temp' : {"long_name":"air temperature","units":"K", "standard_name":"air_temperature"},
        'press': {"long_name":"pressure at cell midpoint","units":"hPa", "standard_name":"air_pressure"},
        'cose' : {"long_name":"cosine latitude at cell boundary","units":"none", "standard_name":"none"},
        'cosc' : {"long_name":"cosine latitude at cell midpoint","units":"none", "standard_name":"none"},
        'latm' : {"long_name":"latitude at cell midpoint","units":"degree_north", "standard_name":"latitude"},
        'lat'  : {"long_name":"latitude at cell boundary","units":"degree_north", "standard_name":"latitude"},
        'dy'   : {"long_name":"cell horizontal width","units":"m", "standard_name":"none"},
        'dz'   : {"long_name":"cell vertical height","units":"m", "standard_name":"none"},
        'mva'  : {"long_name":"number density of air at cell midpoint","units":"m-3", "standard_name":"none"},
        'mvae' : {"long_name":"number density of air at cell boundary","units":"m-3", "standard_name":"none"},
        'z_trop': {"long_name":"layer which contains the tropopause","units":"none", "standard_name":"none"},
        'cflux': {"long_name":"detraining_molar_flux","units":"mol m-2 s-1", "standard_name":"detraining_molar_flux"}
    }

    file_attrs = {
        "Title" : "Two dimensional transport derived from MERRA2.",
        "Contact" : "Luke Western, luke.western@bristol.ac.uk",
    }

    static_vars = ['press', 'cose', 'cosc', 'latm', 'lat', 'dy', 'dz', 'mva', 'mvae']

    start_year = int(start_year)
    if end_year is None:
        end_year = start_year+1
    else:
        end_year = int(end_year)

    for year in range(start_year, end_year):
        yrfns = sorted(glob.glob(f"{monthly_transport}/transport2D_{year}*"))
        if not len(yrfns):
            continue
        yrds = xr.open_mfdataset(yrfns, combine='nested', concat_dim="time")

        for st_var in static_vars:
            yrds[st_var] = yrds[st_var].mean("time")

        var_list = list(yrds.keys())
        coord_list = list(yrds.coords)

        for var in var_list:
            yrds[var].attrs = vars_dict[var]
        for coord in coord_list:
            yrds[coord].attrs = coord_dict[coord]
        yrds.attrs = file_attrs
        yrds.attrs["Filename"] = f"transport2D_{year}.nc"
        yrds.attrs["History"] = f"File created on {pd.to_datetime('today')}"
        yrds.to_netcdf(f"{model_trans_dir}/transport2D_{year}.nc")    

    # yrfns = sorted(glob.glob(f"{monthly_transport}/transport2D_{1900}*"))
    # yrds = xr.open_mfdataset(yrfns, combine='nested', concat_dim="time")

    # for st_var in static_vars:
    #     yrds[st_var] = yrds[st_var].mean("time")

    # var_list = list(yrds.keys())
    # coord_list = list(yrds.coords)

    # for var in var_list:
    #     yrds[var].attrs = vars_dict[var]
    # for coord in coord_list:
    #     yrds[coord].attrs = coord_dict[coord]
    # yrds.attrs = file_attrs
    # yrds.attrs["Filename"] = f"transport2D_{year}.nc"
    # yrds.attrs["History"] = f"File created on {pd.to_datetime('today')}"
    # yrds.to_netcdf(f"{model_trans_dir}/transport2D_{1900}.nc")

if __name__ == "__main__":

    # Get month to work with
    # Format is yyyymmdd
    month = sys.argv[1].replace("-","")

    # Constants
    R = 6.371*1e6 # radius of Earth
    H = 7200. # Scale height in m 
    # Begin by defining the desired 2D grid
    # and defining some needed inputs 
    press2D = np.logspace(3,1,30)
    z2D = H*np.log(1000/press2D)
    dz2D = z2D[1:] - z2D[:-1]
    zm2D = z2D[:-1] + dz2D/2 
    pressm2D = np.exp(-zm2D/H)*1000.
    lat2D = np.linspace(-90,90, 19)
    y2D = lat2D*np.pi/180 * R # 
    dlat2D = (lat2D[1]-lat2D[0])/2.
    latm2D = (lat2D[:-1]+dlat2D)
    dy2D = y2D[1:] - y2D[:-1]
    ym2D = y2D[:-1] + dy2D/2
    cose =  np.cos(lat2D*np.pi/180)
    cosc = np.cos(latm2D*np.pi/180)
    cosphi = 1 #np.cos(lat2D*np.pi/180) # Only needed if using sine-latitude coordinates.
    mvae = np.expand_dims(1e-6 * 100*press2D*1.e3/(28.97*2.87e6*239.27*1.66e-24)/(6.022e23*1e-10),1)
    mva = np.expand_dims(1e-6 * 100*pressm2D*1.e3/(28.97*2.87e6*239.27*1.66e-24)/(6.022e23*1e-10),1)

    # Read in file for this month and get variables
    sfn = f"GEOSChem.SpeciesConc.{month}_0000z.nc4"  
    mfn = f"GEOSChem.StateMet.{month}_0000z.nc4"
    ds_q = utils.opends(filepath + sfn)
    ds_met = utils.opends(filepath + mfn)

    # z-coordinate for height of tracer run 
    pressb3D = ds_met.lev*1e3
    zb3D = H*np.log(ds_met.P0.values/pressb3D)
    press3D = ds_met.ilev*1e3
    z3D = H*np.log(ds_met.P0.values/press3D)

    # Interpolate v-winds onto 2D model a-grid
    dsv_ = ds_met.Met_V
    dsv_.coords['z'] = zb3D  
    dsv = dsv_.swap_dims({'lev': 'z'})
    ds_out = xr.Dataset({"lat": (["lat"], lat2D),
            "lon": (["lon"], ds_met.lon.values),
            "z":  (["z"], zm2D),})
    regridder = xe.Regridder(dsv, ds_out, "bilinear", filename=f"{weight_folder}/bilinear_{str(int(np.random.random()*100))}_{month}.nc") 
    dsv_out = regridder(dsv).to_dataset(name="Met_V").drop('lev')
    grid = Grid(dsv_out, coords={'z': {'center':'z'}}, periodic=False)
    dsvt = grid.transform(dsv_out.Met_V, 'z', zm2D, target_data=dsv_out.z)
    dsvt[:,-1,:,:] = 0
    dsvt[:,0,:,:] = 0 
    # Interpolate q onto v-wind's 2D model a-grid
    dsqv_ = ds_q.copy()
    dsqv_.coords['z'] = zb3D  
    dsqv = dsqv_.swap_dims({'lev': 'z'})
    for var in list(dsqv.keys()):
        if "SpeciesConc_COInfTracer" not in var:
            dsqv = dsqv.drop(var)
    ds_out = xr.Dataset({"lat": (["lat"], lat2D),
            "lon": (["lon"], ds_met.lon.values),
            "z":  (["z"], zm2D),})
    regridder = xe.Regridder(dsqv, ds_out, "bilinear", filename=f"{weight_folder}/bilinear_{str(int(np.random.random()*100))}_{month}.nc") 
    dsqv_out = regridder(dsqv).drop(('lev','ilev'))
    grid = Grid(dsqv_out, coords={'z': {'center':'z'}}, periodic=False)
    dsvqt = xr.Dataset()
    for var in list(dsqv.keys()):
            dsvqt[var] = grid.transform(dsqv_out[var], 'z', zm2D, target_data=dsv_out.z)
            dsvqt[var][:,-1,:,:] = dsvqt[var][:,-2,:,:]
            dsvqt[var][:,0,:,:] = dsvqt[var][:,1,:,:]
    # Set dimension order as input 
    dsvt = dsvt.transpose("time","z","lat","lon")
    dsvqt = dsvqt.transpose("time","z","lat","lon")

    # Interpolate w-winds onto 2D model a-grid
    dsw_ = -ds_met.Met_OMEGA / (ds_met.Met_AIRDEN*9.81) 
    dsw_.coords['z'] = zb3D  
    dsw = dsw_.swap_dims({'lev': 'z'})
    ds_out = xr.Dataset({"lat": (["lat"], latm2D),
            "lon": (["lon"], ds_met.lon.values),})
    regridder = xe.Regridder(dsw, ds_out, "bilinear", filename=f"{weight_folder}/bilinear_{str(int(np.random.random()*100))}_{month}.nc") 
    dsw_out = regridder(dsw).to_dataset(name='Met_W').drop('lev')
    grid = Grid(dsw_out, coords={'z': {'center':'z'}}, periodic=False)
    dswt = grid.transform(dsw_out.Met_W, 'z', z2D, target_data=dsw_out.z)
    dswt[:,:,:,0] = 0
    dswt[:,:,:,-1] = 0
    # # Interpolate q onto w-wind's 2D model a-grid
    dsqw_ = ds_q.copy()
    dsqw_.coords['z'] = zb3D  
    dsqw = dsqw_.swap_dims({'lev': 'z'})
    for var in list(dsqw.keys()):
        if "SpeciesConc_COInfTracer" not in var:
            dsqw = dsqw.drop(var)
    regridder = xe.Regridder(dsqw, ds_out, "bilinear", filename=f"{weight_folder}/bilinear_{str(int(np.random.random()*100))}_{month}.nc") 
    dsqw_out = regridder(dsqw).drop(('lev','ilev'))
    grid = Grid(dsqw_out, coords={'z': {'center':'z'}}, periodic=False)
    dswqt = xr.Dataset()
    for vi, var in enumerate(list(dsqw.keys())):
            dswqt[var] = grid.transform(dsqw_out[var], 'z', z2D, target_data=dsw_out.z)
            dswqt[var][:,:,:,0] = dswqt[var][:,:,:,1]
    # Set dimension order as input 
    dswt = dswt.transpose("time","z","lat","lon")
    dswqt = dswqt.transpose("time","z","lat","lon")

    # Tropopause height
    z_trop = H*np.log(1000/ds_met.Met_TropP).mean(("time", "lon")).interp(lat=latm2D)
    z_trop_index = [abs(zt-z2D).argmin() for zt in z_trop.values]
    z_trop_mask = np.zeros((len(z2D), len(ym2D)), dtype=bool) 
    for i in range(len(ym2D)):
        z_trop_mask[z_trop_index[i]:,i] = True 

    # Infer eddy transport
    K, F, D = infer_eddy_agrid(dsvqt, dsvt,dswqt, dswt, y2D, ym2D)
    # vF, vD = validation_tracer_agrid(dsvqt, dsvt,dswqt, dswt,7, y2D, ym2D) # Can be used to validate
    K, Kyz_zgrid, Kzy_ygrid, Dzy, Dyz = correct_eddy(K, zm2D, z2D, ym2D, y2D)
    K, Kyz_zgrid, Kzy_ygrid, Dzy, Dyz = correct_eddy(K, zm2D, z2D, ym2D, y2D)

    # Advection terms made non-divergent
    # First, derive residual transport (eddy component)
    # dsTwt = interpolate_T(ds_met, "Met_T", zb3D, ds_met.lon.values, z2D, latm2D)
    # dsTwt[:,:,:,0] = 0
    # dsvwt = interpolate_T(ds_met, "Met_V", zb3D, ds_met.lon.values, z2D, latm2D)
    # dsvwt[:,:,:,0] = 0
    # dsvwt[:,:,:,-1] = 0
    # Derive w_star only as v is derived from w forcing non-divergence  
    # w_star = wstar_potentialtemp(dsTwt,dsvwt, H)
    v_star, w_star = derive_ustar(K, Kzy_ygrid, Kyz_zgrid, zm2D, ym2D) 
    w_res = dswt.mean(("time", "lon")).values + w_star # NB this was - w_star if using derive_ustar, but think it should be +ve
    v_res = dsvt.mean(("time", "lon")).values #- v_star
    w,v = make_nondivergent(w_res, v_res, cosc, cose, H, z2D, zm2D, y2D, dz2D, dy2D)

    # Temperature at cell centres
    dsT = interpolate_T(ds_met, "Met_T", zb3D, ds_met.lon.values, zm2D, latm2D).transpose("time","z","lat","lon").mean(("lon","time"))

    # Tests against tracers show that Dzz is underestimated and Dzy is overestimated. 
    # The impact is negligible on Dyy (which dominates) but scaling Dzz and Dzy helps
    # improve agreement. Dzy also needs to be negative with current way the Dzy dispersion 
    # is done.  
    K[0] = K[0] * 1.5
    Dzy = Dzy * -0.75

    # Place all transport in to a netcdf file
    ds_2Dtransport = xr.Dataset(
        data_vars=dict(
            Dzz=(["z", "ym"], K[0]),
            Dyy=(["zm", "y"], K[1]),
            # Dyz=(["zm", "y"], Dyz),
            Dzy=(["zm", "ym"], Dzy),
            v=(["zm", "y"], v),
            w=(["z", "ym"], w),
            temp=(["zm", "ym"], dsT.values),
            press=(["z"], press2D),
            cose=(["y"], cose),
            cosc=(["ym"], cosc),
            latm=(["ym"], latm2D),
            lat=(["y"], lat2D),
            dy=(["ym"], dy2D),
            dz=(["zm"], dz2D),
            mva=(["zm"], np.squeeze(mva)),
            mvae=(["z"], np.squeeze(mvae)),
            z_trop=(["ym"], z_trop.values),
        ),
        coords=dict(
            y=(["y"], y2D),
            ym=(["ym"], ym2D),
            z = (["z"], z2D),
            zm=(["zm"], zm2D),
            time=pd.to_datetime(month),
        ),
        attrs=dict(description="Input transport."),
    )
    
    # Now awkwardly add in the convective parameters
    ds_2Dtransport["cflux"] = add_convection(ds_2Dtransport)
    
    # comp = dict(zlib=True, complevel=5)
    # encoding = {var: comp for var in ds_2Dtransport.data_vars}
    ds_2Dtransport.to_netcdf(f"{monthly_transport}/transport2D_{month[:-2]}.nc")#, encoding=encoding, mode="w")
