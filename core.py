from scipy.integrate import odeint
import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib.pyplot as plt
from pysr import PySRRegressor


# === Simulation ===


def simulate_system(flow_func, x0, t, params):
    """
    Simule le système dynamique défini par flow_func.
    :param flow_func: fonction de dynamique
    :param x0: conditions initiales
    :param t: vecteur de temps
    :param params: dictionnaire des paramètres
    :return: matrice des états simulés
    """
    return odeint(flow_func, x0, t, args=(params,))


# === Calcul enveloppe et regression ===


def compute_envelope_and_regression(t, signal, name, window_size=10, iterations=50):
    # Filtrage local
    centered_mean = uniform_filter1d(signal, size=window_size, mode="nearest")
    squared_diff = (signal - centered_mean) ** 2
    local_std = np.sqrt(
        uniform_filter1d(squared_diff, size=window_size, mode="nearest")
    )

    upper_data = signal + local_std
    lower_data = signal - local_std

    # Régression symbolique
    X_reg = signal.reshape(-1, 1)
    model_upper = PySRRegressor(
        niterations=iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        model_selection="best",
    )
    model_lower = PySRRegressor(
        niterations=iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[],
        model_selection="best",
    )

    model_upper.fit(X_reg, upper_data)
    model_lower.fit(X_reg, lower_data)

    upper_expr = model_upper.get_best()["sympy_format"]
    lower_expr = model_lower.get_best()["sympy_format"]

    print(f"\n=== {name} ===")
    print(f"Upper({name}) = {upper_expr}")
    print(f"Lower({name}) = {lower_expr}")

    # Evaluation
    def eval_expr(expr, x):
        return eval(str(expr).replace("x0", str(x)))

    upper_pred = np.array([eval_expr(upper_expr, val) for val in signal])
    lower_pred = np.array([eval_expr(lower_expr, val) for val in signal])

    # Tracer les résultats
    plt.figure(figsize=(10, 5))
    plt.plot(t, signal, "k", label=f"{name} (réel)")
    plt.plot(t, upper_data, "r--", alpha=0.4, label="Upper (data)")
    plt.plot(t, lower_data, "b--", alpha=0.4, label="Lower (data)")
    plt.plot(t, upper_pred, "r", label="Upper (modèle)")
    plt.plot(t, lower_pred, "b", label="Lower (modèle)")
    plt.fill_between(
        t,
        lower_pred,
        upper_pred,
        where=upper_pred > lower_pred,
        color="gray",
        alpha=0.2,
        label="Enveloppe apprise",
    )
    plt.title(f"Enveloppe dynamique pour {name}")
    plt.xlabel("Temps")
    plt.ylabel(f"{name}")
    plt.legend()
    plt.grid(True)
    plt.show()


# === Application Enveloppe aux variables ===
def apply_to_all_variables(t, X, var_names, window_size=10, iterations=50):
    for i, name in enumerate(var_names):
        compute_envelope_and_regression(X[:, i], name, window_size, iterations)
