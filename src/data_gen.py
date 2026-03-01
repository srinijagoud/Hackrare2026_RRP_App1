import numpy as np


# src/data_gen.py 
rng = np.random.default_rng()
def bern(p):
    return rng.binomial(1, np.clip(p, 0.01, 0.99), size=n)

# HPO-like symptom flags correlated with severity
HP_0006536 = bern(0.08 + 0.60 * sigmoid(1.4 * sev))  # airway obstruction
HP_0002094 = bern(0.12 + 0.55 * sigmoid(1.2 * sev))  # dyspnea
HP_0010307 = bern(0.08 + 0.50 * sigmoid(1.2 * sev))  # stridor
HP_0002205 = bern(0.18 + 0.25 * sigmoid(1.0 * sev))  # recurrent infections
HP_0012735 = bern(0.35 + 0.15 * sigmoid(0.7 * sev))  # cough
HP_0001609 = bern(0.65 + 0.20 * sigmoid(0.6 * sev))  # hoarse voice