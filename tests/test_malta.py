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

    correct_B = np.array([ 0.42922225,  1.2374124 ,  2.04550423,  2.88083018,  3.7159529 ,
        4.55083699,  5.38541681,  6.23327298,  7.06696631,  7.90004446,
        8.73231873,  9.56374079, 10.40801024, 11.21072916, 12.01226389,
       12.83979265, 13.66657479, 14.4930417 , 15.31930158, 16.15840346,
       16.98337378, 17.8070728 , 18.62828538, 19.44823996])

    assert np.allclose(ds_out.burden.values, correct_B)