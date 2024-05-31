import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from tqdm import tqdm

from plato.planets.populations import PopulationModel
from plato.planets.observation import ObservationModel
from plato.stars import filter_valid_targets
from plato.visualisation import FigureProcessor, get_palette, set_plot_defaults

set_plot_defaults()

figure_directory = f"../figures/07_heatmaps/"
save = True

LOPS2 = pd.read_csv(f"../data/processed/LOPS2_targets.csv")
LOPN1 = pd.read_csv(f"../data/processed/LOPN1_targets.csv")

fields = pd.concat([LOPS2, LOPN1])
fields = filter_valid_targets(fields)
fields = fields[
    [
        "Radius",
        "Mass",
        "[Fe/H]",
        "u1",
        "u2",
        "gaiaV",
        "n_cameras",
        "Population",
    ]
]

fields["cos_i"] = 0
fields["sigma_star"] = 10e-6
fields = fields.rename(
    columns={
        "Radius": "R_star",
        "Mass": "M_star",
        "gaiaV": "Magnitude_V",
    }
)

num_mocks = 300
obs_model = ObservationModel(load_mocks=False)

obs_model.save_mocks(
    fields,
    num_embryos=[100],
    metallicity_limit=-0.6,
    num_mocks=num_mocks,
)
