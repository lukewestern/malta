import numpy as np
from malta import model

# Note that JIT compiled transport functions cannot be tested using pytest.

def test_model_run():
    """Test run of model over 2 years of data, 1 climatological"""
    start_year = 1979
    end_year = 1981
    species = "CFC11"

    # Get model years to run
    years = np.array([str(yr) for yr in range(start_year,end_year)])

    emissions = model.create_emissions(species, np.array([10,10]))
    sink = model.create_sink(species)

    ds_out = model.run_model(years, emissions, sink)

    correct_B = np.array([ 0.42921938,  1.23736878,  2.04533306,  2.88045993,  3.71539281,
        4.55013761,  5.38458236,  6.23223737,  7.06569712,  7.89856356,
        8.73061827,  9.56164556, 10.40511349, 11.20664537, 12.00700733,
       12.83387794, 13.66066753, 14.48731327, 15.31350999, 16.15221892,
       16.97667243, 17.79964106, 18.62018707, 19.43920262])

    assert np.allclose(ds_out.burden.values, correct_B)