# malta

A 2D model of atmospheric transport

## Installation

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
dt = 8*3600
species = "CFC-11"

# Set up and run model
years = np.array([str(yr) for yr in range(start_year,end_year)])
emistot = np.repeat(10, len(years))
emissions = model.create_emissions(species, emistot, dt)
sink = model.create_sink(species, dt)
ds_out = model.run_model(years, dt, emissions, sink) 
```
The returned ds_out is an xarray dataset containing monthly mean output variables from 
the 2D model run.

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`malta` was created by Luke Western. It is licensed under the terms of the MIT license.

## Credits

`malta` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
