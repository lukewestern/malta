# malta

MALTA: Model of Averaged in Longitude Transport in the Atmosphere.\
Latest release [![DOI](https://zenodo.org/badge/656881397.svg)](https://zenodo.org/badge/latestdoi/656881397)\
A description of the model is available here: https://doi.org/10.22541/essoar.168890012.27918585/v1

## Installation
To install the latest release:
```bash
$ pip install malta
```

## Usage

`malta` is a two-dimensional (longitudinally-averaged) model of atmospheric transport.
A simple run of constant emissons of 10 Gg of CFC-11 with zero initial conditions
from 2010-2020 inclusive could be:
```python
import numpy as np
from malta import model

start_year = 2010
end_year = 2021
species = "CFC11"

# Set up and run model
years = np.array([str(yr) for yr in range(start_year,end_year)])
emistot = np.repeat(10, len(years))
emissions = model.create_emissions(species, emistot)
sink = model.create_sink(species)
ds_out = model.run_model(years, emissions, sink) 
```
The returned ds_out is an xarray dataset containing monthly mean output variables from 
the 2D model run.
See docs/example.ipynb or the [readthedocs](https://malta.readthedocs.io/en/latest/index.html) page for more information on running the model.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`malta` was created by Luke Western. It is licensed under the terms of the MIT license.

## Credits

`malta` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
