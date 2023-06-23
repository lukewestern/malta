# Core to set-up and run 2D model of atmospheric transport.

import numpy as np
from malta import transport
from malta import utils
import xarray as xr
import pandas as pd


def run_model(years_in, dt, emissions, sink, ics=None, trans_dir=None):
    """
    Runs 2D model for time period specified. 

    Args:
        years_in (array/list): Whole years for which to run model.
        dt (float): Time step in seconds (recommended 8 hrs or less)
        emissions (class): Class containing emissions.
        sink (sink): Class containing sinks.
        ics (array, optional): Initial conditions of model. Defaults to None (zero initial conditions).
        trans_dir (str, optional): Path to directory containing transport files. Defaults to None.

    Returns:
        dataset: xarray dataset containing outputs from model.
    """
    nyears = int(len(years_in))
    # days in each month. Ignore leap years for now.
    dom = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    sid = 3600*24  # seconds in day
    dtpd = int(sid/dt)  # dts per day

    if not trans_dir:
        trans_dir = utils.get_transport_directory()

    # Get the necessary model constants
    ds_t = utils.opends(trans_dir + "transport2D_1900.nc")
    cosc = ds_t.cosc.values
    cose = ds_t.cose.values
    dz = ds_t.dz.values
    dy = ds_t.dy.values
    zm = ds_t.zm.values
    ym = ds_t.ym.values
    z = ds_t.z.values
    y = ds_t.y.values
    mva = np.expand_dims(ds_t.mva.values, 1)
    mvae = np.expand_dims(ds_t.mvae.values, 1)
    nzm = len(ds_t.zm)
    nym = len(ds_t.ym)

    z_horiz = np.tile(zm, (nym, 1)).T

    # If no initial conditions, set to zero
    if ics is None:
        ics = np.ones((len(ds_t.zm), len(ds_t.ym)))*1e-16

    # Make output dataset and storage arrays
    dts = pd.date_range(f"{years_in[0]}-01-01",
                        f"{int(years_in[-1])}-12-01", freq="MS")
    ds_out = xr.Dataset(
        data_vars=dict(),
        coords=dict(
            time=pd.to_datetime(dts),
            z=(["z"], ds_t.zm.values),
            lat=(["lat"], ds_t.latm.values),

        ),
        attrs=dict(description="2D model output."),
    )

    c_arr_mm = np.zeros((12*nyears, nzm, nym))
    c_arr_mm_trop = np.zeros((12*nyears, nzm, nym))
    loss_mm = np.zeros((12*nyears, nzm, nym))
    cend = np.zeros((12*nyears, nzm, nym))

    # Use climatological transport for years with no met
    years = utils.set_climatological_years(years_in, trans_dir)

    for yr, year in enumerate(years):
        mt = 0
        ds_t = utils.opends(trans_dir + f"transport2D_{str(year)}.nc")
        Dyy = ds_t.Dyy.values[:, :, :-1]
        Dzz = ds_t.Dzz.values[:, :-1, :]
        Dyz = ds_t.Dzy.values
        w = ds_t.w.values
        v = ds_t.v.values
        temp = ds_t.temp.values

        for mnth in dom:
            t = 0
            A = transport.eulerian_diffusion_yy(Dyy[mt, :, :], dy[0], cosc, cose) + \
                transport.eulerian_diffusion_zz(Dzz[mt, :, :],  dz[0], mva)
            c_arr = np.zeros((mnth*dtpd, nzm, nym))
            loss_arr = np.zeros((mnth*dtpd, nzm, nym))
            c_arr[-1, :, :] = ics
            for _ in range(mnth):
                for __ in range(dtpd):
                    c_arr[t, :, :] = c_arr[t-1, :, :] + emissions.emit(yr)
                    c_arr[t, :, :] = transport.linrood_advection(
                        c_arr[t, :, :], v[mt, :, :], w[mt, :, :], dy, dz, dt, mva, mvae, cosc, cose)
                    c_arr[t, :, :] = transport.rk4(c_arr[t, :, :], mva,  A, dt)
                    wdiff, vdiff = transport.u_diffusive(
                        c_arr[t, :, :], Dyz[mt, :, :], zm, ym, z, y, dz, dy)
                    c_arr[t, :, :] = transport.linrood_advection(
                        c_arr[t, :, :], vdiff, wdiff, dy, dz, dt, mva, mvae, cosc, cose)
                    c_arr[t, :, :], loss_arr[t, :, :] = sink.losses(
                        c_arr[t, :, :], mt, temp[mt, :, :])
                    t += 1

            ics = c_arr[-1, :, :]
            cend[yr*12 + mt, :, :] = c_arr[-1, :, :]
            c_arr_mm[yr*12 + mt, :, :] = np.mean(c_arr, axis=0)
            loss_mm[yr*12 + mt, :, :] = np.mean(loss_arr, axis=0)

            c_arr_mm_trop[yr*12 + mt, :, :] = np.mean(c_arr, axis=0)
            strat_mask = z_horiz >= np.expand_dims(
                ds_t.z_trop.values[mt, :], 0)
            c_arr_mm_trop[yr*12 + mt, strat_mask] = 0.

            mt += 1

    # Get climatological tropopause height
    trop_mask = np.zeros((12, nzm, nym), dtype=bool)
    strat_mask = np.zeros((12, nzm, nym), dtype=bool)
    for i in range(12):
        trop_mask[i, :, :] = z_horiz < np.expand_dims(
            ds_t.z_trop.values[i, :], 0)
        strat_mask[i, :, :] = z_horiz >= np.expand_dims(
            ds_t.z_trop.values[i, :], 0)

    B = np.sum(c_arr_mm*(1/emissions.unit_scale)*emissions.mol_air_bx,
               (1, 2))*emissions.molmass/emissions.emis_scale
    L = np.sum(loss_mm*(1/emissions.unit_scale)*emissions.mol_air_bx,
               (1, 2))*emissions.molmass/emissions.emis_scale
    loss_trop = loss_mm.copy()
    loss_trop[np.tile(strat_mask, (nyears, 1, 1))] = 0.
    loss_strat = loss_mm.copy()
    loss_strat[np.tile(trop_mask, (nyears, 1, 1))] = 0.
    L_trop = np.sum(loss_trop*(1/emissions.unit_scale)*emissions.mol_air_bx,
                    (1, 2))*emissions.molmass/emissions.emis_scale
    L_strat = np.sum(loss_strat*(1/emissions.unit_scale) *
                     emissions.mol_air_bx, (1, 2))*emissions.molmass/emissions.emis_scale
    B_trop = np.sum(c_arr_mm_trop*(1/emissions.unit_scale) *
                    emissions.mol_air_bx, (1, 2))*emissions.molmass/emissions.emis_scale
    if (L == 0).all():
        lifetime = np.repeat(np.inf, len(L))
        lifetime_trop = np.repeat(np.inf, len(L))
        lifetime_strat = np.repeat(np.inf, len(L))
    else:
        lifetime = B/L/(3600*24*365)
        lifetime_trop = B/L_trop/(3600*24*365)
        lifetime_strat = B/L_strat/(3600*24*365)

    return create_output(ds_out, c_arr_mm, cend, loss_mm, lifetime, lifetime_trop, lifetime_strat, B, B_trop, emissions)


def create_output(ds_out, c_arr_mm, cend, loss_mm, lifetime, lifetime_trop, lifetime_strat, B, B_trop, emissions):
    """Put variables into xarray dataset"""

    ds_out[emissions.species] = (('time', 'z', "lat"), c_arr_mm)
    ds_out[f"{emissions.species}_end"] = (('time', 'z', "lat"), cend)
    ds_out['loss'] = (('time', 'z', "lat"), loss_mm)
    ds_out['lifetime'] = (('time'), lifetime)
    ds_out['lifetime_trop'] = (('time'), lifetime_trop)
    ds_out['lifetime_strat'] = (('time'), lifetime_strat)
    ds_out['burden'] = (('time'), B)
    ds_out['burden_trop'] = (('time'), B_trop)
    ds_out[emissions.species].attrs = {"standard_name": f"{emissions.species}_mole_fraction_in_air",
                                       "long_name": f"Mean {emissions.species} for each month",
                                       "units": f"{1/emissions.unit_scale} mol mol-1"}
    ds_out[f"{emissions.species}_end"].attrs = {"standard_name": f"{emissions.species}_mole_fraction_in_air_end",
                                                "units": f"{1/emissions.unit_scale} mol mol-1",
                                                "long_name": "Mole fraction at end time of month"}
    ds_out['loss'].attrs = {"standard_name": f"{emissions.species}_loss",
                            "units": f"{1/emissions.unit_scale} mol mol-1 s-1",
                            "long_name": f"Loss of {emissions.species}"}
    ds_out['lifetime'].attrs = {"standard_name": f"{emissions.species}_lifetime",
                                "units": f"years",
                                "long_name": "Total atmospheric lifetime"}
    ds_out['lifetime_trop'].attrs = {"standard_name": f"{emissions.species}_tropospheric_lifetime",
                                     "units": f"years",
                                     "long_name": "Tropospheric atmospheric lifetime"}
    ds_out['lifetime_strat'].attrs = {"standard_name": f"{emissions.species}_stratospheric_lifetime",
                                      "units": f"years",
                                      "long_name": "Stratospheric atmospheric lifetime"}
    ds_out['burden'].attrs = {"standard_name": f"{emissions.species}_burden",
                              "units": f"{emissions.emis_scale} kg",
                              "long_name": "Total atmospheric burden"}
    ds_out['burden_trop'].attrs = {"standard_name": f"{emissions.species}_tropospheric_burden",
                                   "units": f"{emissions.emis_scale} kg",
                                   "long_name": "Tropospheric atmospheric burden"}
    
    return ds_out

class sink:
    """
    Class to build single container for sink processes without passing loss fields
    into model or having to load them during run time.

    Args:
        dt (float): Time step in seconds.
        strat_loss (array, optional): Stratospheric loss rate in each grid cell. Defaults to None.
        OH_field (array, optional): OH loss rate in each grid cell. Defaults to None.
        A_OH (float, optional): Arrhenius A constant for OH. Defaults to None.
        ER_OH (float, optional): Arrhenius E/R constant for OH. Defaults to None.
        Cl_field (float, optional): Cl loss rate in each grid cell. Defaults to None.
        A_Cl (float, optional): Arrhenius A constant for Cl. Defaults to None.
        ER_Cl (float, optional): Arrhenius E/R constant for Cl. Defaults to None.
    """

    def __init__(self, dt, strat_loss=None,
                 OH_field=None, A_OH=None, ER_OH=None,
                 Cl_field=None, A_Cl=None, ER_Cl=None):

        self.strat_loss = strat_loss
        self.OH_field = OH_field
        self.Cl_field = Cl_field
        self.A_OH = A_OH
        self.ER_OH = ER_OH
        self.A_Cl = A_Cl
        self.ER_Cl = ER_Cl
        self.dt = dt

    def losses(self, cin, month, Temp=None):
        """Computes losses in mole fraction field"""
        loss_out = np.zeros_like(cin)
        if self.strat_loss is not None:
            cin, strat_loss = self.first_order_loss(
                cin, self.strat_loss[month, :, :])
            loss_out += strat_loss
        if self.OH_field is not None:
            cin, OH_loss = self.arrhenius_loss(
                cin, self.A_OH, self.ER_OH, self.OH_field[month, :, :], Temp)
            loss_out += OH_loss
        if self.Cl_field is not None:
            cin, Cl_loss = self.arrhenius_loss(
                cin, self.A_Cl, self.ER_Cl, self.Cl_field, Temp)
            loss_out += Cl_loss
        return cin, loss_out

    def first_order_loss(self, c_in, sink):
        """Compute first order loss in mol/mol/s"""
        c_out = c_in * np.exp(-self.dt*sink)
        return c_out, (c_in - c_out)/self.dt

    def arrhenius_loss(self, c_in, A, ER, field, Temp):
        """Compute loss using arrhenius rate constant"""
        c_loss = self.arrhenius_rate_constant(
            A, ER, Temp) * field * self.dt * c_in
        if (c_loss > c_in).any():
            c_loss[c_loss > c_in] = c_in[c_loss > c_in] - 1e-16
        return c_in - c_loss,  c_loss/self.dt

    def arrhenius_rate_constant(self, A, ER, Temp):
        """Compute arrhenius rate constant"""
        return A*np.exp(-ER/Temp)


class emissions:
    """
    Class to build single container for emissive processes.

    Args:
        emis (array): Emissions for each year for each latitude.
        dt (float): Time step in seconds.
        species (string): Species to emit.
        lat (array): Latitude at grid cell edges.
        mva (array): Molar density of air (mol/m3) for each vertical layer. 
        dz (array): Uniform vertical grid spacing.
        R (float, optional): Radius of Earth in metres. Defaults to 6378.1e3.
        emis_units (str, optional): Units of emissions (g, kg, Mg, Gg, Tg). Defaults to "Gg".
        mf_units (str, optional): Units of mole fractions (ppq, ppt, ppb, ppm). Defaults to "ppt".
    """

    def __init__(self, emis, dt, species, lat, mva, dz, R=6378.1e3, emis_units="Gg", mf_units="ppt"):

        if emis.shape[1] != (len(lat)-1):
            print("Warning: shape of emissions array do not match latitude boxes")

        self.dt = dt
        self.species = species
        self.molmass = utils.get_molmass(species)
        mfunit_dict = {"ppq": 1e15, "ppt": 1e12, "ppb": 1e9, "ppm": 1e6}
        emisunit_dict = {"g": 1, "kg": 1e3, "Mg": 1e6, "Gg": 1e9, "Tg": 1e12}
        self.unit_scale = mfunit_dict[mf_units]
        self.emis_scale = emisunit_dict[emis_units]
        sindbx = np.sin(lat[1:]*np.pi/180.)-np.sin(lat[:-1]*np.pi/180.)
        self.mol_air_bx = mva*2*np.pi*R**2*sindbx*dz  # mol of air per box
        self.ny = int(len(lat)-1)
        self.nz = len(dz)
        self.emis = emis

    def emit(self, year):
        """Function to emit emissions"""
        mol_emitted = np.zeros((self.nz, self.ny))
        g_emitted = self.emis[year, :]*self.emis_scale*self.dt / (3600*24*365)
        mol_emitted[0, :] = g_emitted/self.molmass/self.mol_air_bx[0, :]
        return mol_emitted*self.unit_scale


def create_sink(species, dt):
    """Function to create sink class from species"""
    species_info = utils.get_speciesinfo(species)
    if "A_OH" in species_info.keys() and "ER_OH" in species_info.keys():
        A_OH = species_info["A_OH"]
        ER_OH = species_info["ER_OH"]
        OH_field = utils.get_OHfield()
    else:
        A_OH = None
        ER_OH = None
        OH_field = None
    strat_loss = utils.get_stratfield(species)
    return sink(dt, strat_loss, OH_field, A_OH, ER_OH)


def create_emissions(species, emis, dt, distribute="uniform", weights=None, R=6378.1e3, emis_units="Gg", mf_units="ppt", trans_dir=None):
    """
    Wrapper to create emissions class. Provides a convenient way to distribute the global total of emissions. Emissions distributions have been interpolated
    from multiple data sources: 
    "land"/"ocean" Olsen et al., 2001 (https://doi.org/10.1641/0006-3568(2001)051[0933:TEOTWA]2.0.CO;2)
    "population" GPWv4, average of 2000-2020 (https://doi.org/10.7927/H4JW8BX5)
    "gdp" Kammu et al., 2020, 1990-2015 average (https://doi.org/10.5061/dryad.dk1j0)

    Args:
        species (str): Species to emit.
        emis (array): Array of total global emissions for each year of run.
        dt (float): Time step in seconds.
        distribute (str, optional): How to distribute global total emissions. Options are: "uniform", "land", "ocean", "population" and "gpb. Defaults to "uniform".
        weights (array, optional): Option to provide custom weights to distribute emissions in each latitude box. Can either be of length of latitudes or different weights for each year. Defaults to None.
        R (float, optional): Radius of Earth. Defaults to 6378.1e3.
        emis_units (str, optional): Units of emissions (g, kg, Mg, Gg, Tg). Defaults to "Gg".
        mf_units (str, optional): Units of mole fractions (ppq, ppt, ppb, ppm). Defaults to "ppt".
        trans_dir (str, optional): Path to directory containing transport files. Defaults to None.

    Returns:
        class: Class containing emissoins information.
    """

    weights_dict = {"uniform": np.array([0.00759612, 0.02255757, 0.03683361, 0.04999048, 0.06162842,
                                         0.0713938, 0.07898993, 0.08418598, 0.08682409, 0.08682409,
                                         0.08418598, 0.07898993, 0.0713938, 0.06162842, 0.04999048,
                                         0.03683361, 0.02255757, 0.00759612]),
                    "land":  np.array([7.0636270e-09, 6.5525919e-02, 5.2834521e-03, 2.2426052e-03,
                                       5.8303396e-03, 4.0336054e-02, 5.3951893e-02, 7.8690864e-02,
                                       5.8018744e-02, 8.3756834e-02, 6.4553551e-02, 1.2939535e-01,
                                       8.4416285e-02, 1.3299182e-01, 7.6903336e-02, 1.0026484e-01,
                                       1.5211911e-02, 2.6261979e-03]),
                    "ocean": np.array([2.7425755e-09, 1.4863277e-02, 4.0725686e-02, 8.5077688e-02,
                                       6.7930795e-02, 1.0556417e-01, 6.8197727e-02, 1.1132761e-01,
                                       7.4817389e-02, 1.1290311e-01, 6.8737686e-02, 8.1185445e-02,
                                       4.6169315e-02, 4.9939472e-02, 2.4708044e-02, 2.0545173e-02,
                                       1.7700905e-02, 9.6065011e-03]),
                    "population": np.array([1.4483384e-09, 1.4483384e-09, 1.4483384e-09, 5.9895810e-05,
                                            6.0846534e-04, 1.2203984e-02, 2.5620667e-02, 2.3779465e-02,
                                            6.5588579e-02, 8.0703236e-02, 1.3553666e-01, 2.5726408e-01,
                                            2.3727001e-01, 1.0700166e-01, 5.1544670e-02, 2.7892306e-03,
                                            2.9307219e-05, 8.4769439e-08]),
                    "gdp": np.array([1.39981629e-11, 1.39981629e-11, 1.39981629e-11, 1.57431961e-04,
                                     1.30003423e-03, 2.15199850e-02, 3.08055561e-02, 1.06130345e-02,
                                     2.80869063e-02, 3.84262614e-02, 6.47760555e-02, 1.35033742e-01,
                                     2.93563217e-01, 2.33442307e-01, 1.32013127e-01, 1.02106240e-02,
                                     5.18115994e-05, 1.39981629e-11])
                    }

    if trans_dir == None:
        trans_dir = utils.get_transport_directory()
    ds_t = utils.opends(trans_dir + "transport2D_1900.nc")

    if weights is None:
        weights = weights_dict[distribute]

    if len(np.squeeze(weights).shape) == 1:
        weights = np.expand_dims(np.squeeze(weights), 0)
    if len(np.squeeze(emis).shape) == 1:
        emis = np.expand_dims(np.squeeze(emis), 1)

    emis_lat = weights*emis
    return emissions(emis_lat, dt, species, ds_t.lat.values, np.expand_dims(ds_t.mva.values, 1), np.expand_dims(ds_t.dz.values, 1), R=R, emis_units=emis_units, mf_units=mf_units)
