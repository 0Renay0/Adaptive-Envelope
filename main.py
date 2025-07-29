import numpy as np
from dynamique import flow_two_tanks
from core import simulate_system, apply_to_all_variables

# === Paramètres du système ===
params = {"A1": 1.0, "A2": 1.0, "k12": 0.1, "k2": 0.05, "q_in": 0.3}

# === Simulation ===

t = np.linspace(0, 50, 200)  # Vecteur de temps
x0 = [40, 31]  # Conditions initiales
X = simulate_system(flow_two_tanks, x0, t, params)  # X[:, 0] = h1, X[:, 1] = h2

# === Noms des variables ===
var_names = [f"x{i + 1}" for i in range(len(x0))]

# === Calcul et tracé de l'enveloppe et régression ===
apply_to_all_variables(t, X, var_names, window_size=10, iterations=50)
