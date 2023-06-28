import numpy as np
from malta import model

# Note that JIT compiled transport functions cannot be tested using pytest.")

def test_model_run():
    """Test run of model over 2 years of data, 1 climatological"""
    start_year = 1979
    end_year = 1981
    dt = 8*3600
    species = "CFC11"

    # Get model years to run
    years = np.array([str(yr) for yr in range(start_year,end_year)])

    emissions = model.create_emissions(species, np.array([10,10]), dt)
    sink = model.create_sink(species, dt)

    ds_out = model.run_model(years, dt, emissions, sink)

    correct_B = np.array([ 0.42922289,  1.23741207,  2.0454774 ,  2.88075264,  3.7158526 ,
        4.55077974,  5.38544662,  6.23339043,  7.06719568,  7.90047437,
        8.73300646,  9.56457015, 10.40864785, 11.21081906, 12.01189346,
       12.83952341, 13.66703514, 14.49435361, 15.32120355, 16.16060161,
       16.98574529, 17.80947086, 18.63087828, 19.45076816])

    assert np.allclose(ds_out.burden.values, correct_B)