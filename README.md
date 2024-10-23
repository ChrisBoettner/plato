# PLATO Exoplanet Yield Predictions

This repository contains the code and data used for the paper "Exoplanets Across Galactic Stellar Populations with PLATO: Estimating Exoplanet Yields Around FGK Stars for the Thin Disk, Thick Disk and Stellar Halo".  The project predicts the exoplanet detection yields of the PLATO mission for different Galactic stellar populations (thin disk, thick disk, and halo). This is achieved by combining the all-sky PLATO input catalog (asPIC) with a kinematic classification scheme and the New Generation Planet Population Synthesis (NGPPS) models.  We estimate PLATO's exoplanet detection efficiency based on instrumental, planetary, and stellar properties.

## Data Availability

The NGPPS data used in this project is available from the Data & Analysis Center for Exoplanets (dace.unige.ch). The asPIC catalog can retrieved from Montalto2021. The Gaia DR3 data can be obtained through the Gaia Archive.  Due to size constraints, these datasets are not included in the repository.  See the paper for details on data acquisition and processing, or send me a message. The processed target lists for the LOPS2 and LOPN1 fields and the halo special target list are available via Zenodo: [https://doi.org/10.5281/zenodo.11428968](https://doi.org/10.5281/zenodo.11428968).

## Running the Code

The code is written in Python and relies on several scientific computing libraries. The notebooks provide a step-by-step guide through the analysis.  To run the notebooks or scripts, you need to:

1.  **Install dependencies:**  The required packages are listed in the `requirements.txt` file. You can install them using `pip install -r requirements.txt`.
2.  **Download the data:** Download the NGPPS, asPIC, and Gaia DR3 data as described in the Data Availability section. Update the paths in the scripts and notebooks to point to the downloaded data.
3.  **Run the scripts:**  The `scripts` directory contains several Python scripts to perform specific tasks, such as creating mock observations and submitting Slurm jobs for the notebooks. See the individual script files for usage instructions.
4.  **Run the notebooks:**  The notebooks provide a more interactive way to explore the data and reproduce the results and figures of the paper.  You can run the notebooks using Jupyter.  Please adjust the paths, if necessary.

## Scripts

- `create_mock_observations.py`: This script generates and saves mock exoplanet observations based on the NGPPS data and PLATO detection efficiency model. It takes the number of embryos and an optional metallicity limit as input arguments.
-   `run_notebook.py`: This script allows executing Jupyter notebooks programmatically. It takes the notebook name and an optional path to the notebook directory as input arguments.
- `slurm_create_mocks.py`: This script submits Slurm jobs for the `create_mock_observations.py` script. It automatically generates Slurm batch scripts for different numbers of embryos and metallicity limits. See `mock_obs_out` for outputs.
-   `slurm_run_notebooks.py`:  This script submits Slurm jobs for running specified Jupyter notebooks. See `notebooks_out` for outputs.

## Notebooks

The Jupyter notebooks (`notebooks` directory) provide a detailed walkthrough of the analysis. Please run notebooks in their numerical order, and replace paths, if necessary.

## Contributing

Contributions are welcome.  Please open an issue to discuss any proposed changes or bug reports.


## License

The MIT License (MIT)

Copyright (c) 2024 Christopher Boettner

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
