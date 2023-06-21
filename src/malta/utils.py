# Auxillary routines for 2D model
import xarray as xr
import json
import os
import glob
import datetime
import pandas as pd

rel_path = f"{os.path.dirname(__file__)}"


def opends(fn):
    """Open an xarray dataset"""
    with xr.open_dataset(fn) as load:
        ds = load.load()
    return ds


def get_molmass(species):
    """Gets the molar mass from species_info file"""
    with open(f"{rel_path}/aux_data/species_info.json") as f:
        species_info = json.load(f)
    return float(species_info[species]["mol_mass"])


def get_speciesinfo(species):
    """Return all info for species in species_info file"""
    with open(f"{rel_path}/aux_data/species_info.json") as f:
        species_info = json.load(f)
    return species_info[species]


def get_siteinfo(site):
    """Return info about a site from site_info"""
    with open(f"{rel_path}/aux_data/site_info.json") as f:
        site_info = json.load(f)
    return site_info[site]


def get_OHfield():
    """Returns OH field as array"""
    OH_ds = opends(f"{rel_path}/model_data/sinks/OH_fields.nc")
    return OH_ds.OH.values


def get_stratfield(species):
    """Returns stratospheric sink for species as array"""
    ds_stratloss = opends(f"{rel_path}/model_data/sinks/strat_loss.nc")
    if species in ds_stratloss.keys():
        return ds_stratloss[species].values
    else:
        print(f"No stratospheric sink field found for {species}")
        return None


def get_transport_directory():
    """Returns default directory containing transport files"""
    return f"{rel_path}/model_data/transport/"


def set_climatological_years(years, trans_dir):
    """
    Finds years where no transport is available and set to climatological trasport.
    This changes the year to read to 1900.

    Args:
        years (list/array): Years for which transport is needed.
        trans_dir (string): Directory containing the transport.

    Returns:
        array/list: years input with years with no transport set to 1900.
    """
    tpfns = glob.glob(f"{trans_dir}/*")
    tpyrs = [tpfn.split(".")[-2].split("_")[-1] for tpfn in tpfns]
    for i, yr in enumerate(years):
        if yr not in tpyrs:
            years[i] = "1900"
    return years


def decimal_to_pandas(dec, offset_days=0):
    """
    Convert decimal date to pandas datetime

    Args:
        dec (float): Decimal dates to convert
    Returns:
        List: List of pandas datetimes
    """
    dates = []
    for f in dec:
        year = int(f)
        yeardatetime = datetime.datetime(year, 1, 1)
        daysPerYear = 365
        days = int(daysPerYear*(f - year))
        dates.append(pd.Timestamp(
            yeardatetime + datetime.timedelta(days=days) + datetime.timedelta(days=offset_days)))

    return dates
