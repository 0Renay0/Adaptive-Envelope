from scipy.integrate import odeint

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
