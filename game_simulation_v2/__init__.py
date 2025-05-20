# ────────────────────────────────────────────────
# Classes
# ────────────────────────────────────────────────
from .agent import Agent
from .environnement import Environnement
from .model import TimeSeriesDataset, PositionalEncoding, TransformerClassifier, SimpleCNN1D

# ────────────────────────────────────────────────
# Fonctions utilitaires
# ────────────────────────────────────────────────
from .visualisation import replicator_dynamics_2x2

# ────────────────────────────────────────────────
# Jeux 2×2 prédéfinis
# ────────────────────────────────────────────────
from .games import (
    A_PD, B_PD,   # Prisoner’s Dilemma
    A_SH, B_SH,   # Stag Hunt
    A_MP, B_MP,   # Matching Pennies
    A_A,  B_A,    # Exemple 1
    A_A2, B_A2,   # Exemple 2
)

__all__ = [
    # classes
    "Agent", "Environnement",
    # fonctions
    "replicator_dynamics_2x2",
    # matrices de jeux
    "A_PD", "B_PD", "A_SH", "B_SH",
    "A_MP", "B_MP", "A_A", "B_A", "A_A2", "B_A2",
]

# Version du package (optionnel)
__version__ = "0.1.0"

