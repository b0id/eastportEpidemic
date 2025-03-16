import numpy as np
import pandas as pd
import geopandas as gpd
import logging
from scipy.integrate import solve_ivp
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============== Model Constants ===============
INCUBATION_PERIOD = 10
INFECTIOUS_PERIOD = 14
SIGMA = 1 / INCUBATION_PERIOD
GAMMA = 1 / INFECTIOUS_PERIOD
BETA_BASE = 0.5
BETA_FLOOD = 1.2
INTERVENTION_EFFECTS = {5: 0.90, 10: 0.70, 15: 0.40, 20: 0.20}
CASE_FATALITY_RATES = {"young": 0.02, "adult": 0.04, "senior": 0.07, "elderly": 0.12}
NEIGHBOR_RADIUS = 150  # meters, adjusted for clearer spread

class LeptoModel:
    def __init__(self, centroids, response_day=10):
        self.centroids = centroids
        self.response_day = response_day
        self.tree = cKDTree(np.column_stack([centroids['x_coord'], centroids['y_coord']]))
        self.neighbors = self.tree.query_ball_tree(self.tree, NEIGHBOR_RADIUS)

    def transmission_rate(self, t, household, neighbor_factor):
        base_rate = BETA_BASE + (household['FloodRisk'] * (BETA_FLOOD - BETA_BASE))
        density_factor = 0.8 + (household['density_no'] * 0.4)
        time_factor = 1.0 if t < 5 else 1.2 if t < 15 else 1.0

        beta = base_rate * density_factor * time_factor * neighbor_factor

        if t >= self.response_day:
            days_since_response = min(t - self.response_day, 7)
            effect = INTERVENTION_EFFECTS.get(self.response_day, 0.5)
            beta *= (1 - days_since_response / 7 * effect)

        return beta

    def determine_initial_infected(self, household):
        risk = household['lepto_risk']
        probs = {risk >= 3: 0.75, risk >= 2: 0.4, risk >= 1: 0.1}
        return 1 if np.random.rand() < probs.get(True, 0.01) else 0

    def seir_model(self, t, y, household, neighbor_infectious):
        S, E, I, R = y
        neighbor_factor = 1 + (neighbor_infectious * 0.05)
        beta = self.transmission_rate(t, household, neighbor_factor)
        dSdt = -beta * S * I
        dEdt = beta * S * I - SIGMA * E
        dIdt = SIGMA * E - GAMMA * I
        dRdt = GAMMA * I
        return [dSdt, dEdt, dIdt, dRdt]

    def run_household_simulation(self, idx, days=60):
        household = self.centroids.iloc[idx]
        if household['vacancy']:
            return None
        initial_infected = self.determine_initial_infected(household)
        if initial_infected == 0:
            return None

        y0 = [household['house_size'] - initial_infected, 0, initial_infected, 0]
        neighbors_idx = self.neighbors[idx]

        t_eval = np.arange(days + 1)
        neighbor_infectious = np.random.rand() * len(neighbors_idx)

        solution = solve_ivp(
            lambda t, y: self.seir_model(t, y, household, neighbor_infectious),
            [0, days], y0, t_eval=t_eval, method='RK45', rtol=1e-5, atol=1e-7
        )

        df = pd.DataFrame({
            "day": t_eval,
            "susceptible": solution.y[0],
            "exposed": solution.y[1],
            "infectious": solution.y[2],
            "recovered": solution.y[3]
        })

        df['cumulative_cases'] = initial_infected + SIGMA * df['exposed'].cumsum()
        cfr = CASE_FATALITY_RATES.get(household['age_cat'], 0.05)
        cfr *= 1.25 if household['disability'] else 1
        df['deaths'] = df['cumulative_cases'] * cfr

        return df

    def run_simulation(self, days=60, n_jobs=4):
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.run_household_simulation)(idx, days)
            for idx in range(len(self.centroids))
        )
        results = [r for r in results if r is not None]
        daily_totals = pd.DataFrame({"day": np.arange(days + 1)})
        for key in ['susceptible', 'exposed', 'infectious', 'recovered', 'cumulative_cases', 'deaths']:
            daily_totals[key] = sum(df[key] for df in results)
        logger.info("Simulation completed. Cases: %d, Deaths: %d",
                    daily_totals['cumulative_cases'].iloc[-1], daily_totals['deaths'].iloc[-1])
        return daily_totals, results
