import numpy as np

# Exemple : dynamique à deux bacs interconnectés
def flow_two_tanks(x, t, params):
    """
    Dynamique générique à partir des paramètres.
    :param x: vecteur des états [h1, h2]
    :param t: temps
    :param params: dictionnaire des paramètres {A1, A2, k12, k2, q_in}
    :return: dérivées des états
    """
    A1, A2 = params['A1'], params['A2']
    k12, k2 = params['k12'], params['k2']
    q_in = params['q_in']

    h1, h2 = x
    q12 = k12 * np.sqrt(abs(h1 - h2))
    qout = 5 * k2 * np.sqrt(abs(h2))

    return [(q_in - q12) / A1, (q12 - qout) / A2]

