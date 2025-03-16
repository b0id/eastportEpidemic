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
BETA_BASE = 0.4  # Slightly reduced from 0.5
BETA_FLOOD = 1.0  # Slightly reduced from 1.2
INTERVENTION_EFFECTS = {
    10: 0.85,  # 85% reduction for day 10 response
    20: 0.65,  # 65% reduction for day 20 response
    30: 0.45,  # 45% reduction for day 30 response
    40: 0.25   # 25% reduction for day 40 response
}
CASE_FATALITY_RATES = {"young": 0.02, "adult": 0.04, "senior": 0.07, "elderly": 0.12}
NEIGHBOR_RADIUS = 150  # meters, adjusted for clearer spread

class LeptoModel:
    def __init__(self, centroids, response_day=10):
        self.centroids = centroids
        self.response_day = response_day
        self.tree = cKDTree(np.column_stack([centroids['x_coord'], centroids['y_coord']]))
        self.neighbors = self.tree.query_ball_tree(self.tree, NEIGHBOR_RADIUS)
        
        # Use different random seeds for different response days to ensure variance
        np.random.seed(42 + response_day)
        logger.info(f"Created model with response day {response_day}")

    def transmission_rate(self, t, household, neighbor_factor):
        """Calculate transmission rate with geographic and time factors"""
        # Base rate from flood risk
        base_rate = BETA_BASE + (household['FloodRisk'] * (BETA_FLOOD - BETA_BASE))
        
        # Population density effect
        density_factor = 0.8 + (household['density_no'] * 0.4)
        
        # Time-based factors create realistic waves
        time_factor = 1.0
        if t < 5:
            time_factor = 0.8  # Slower start
        elif t < 15:
            time_factor = 1.2  # Acceleration phase
        else:
            time_factor = 1.0  # Steady phase
        
        # Location-based factor for geographic variation
        x_coord = household['x_coord']
        y_coord = household['y_coord']
        geo_factor = 1.0 + 0.15 * np.sin(x_coord/3000 + self.response_day/10) * np.cos(y_coord/2500)
        
        # Calculate beta with all factors
        beta = base_rate * density_factor * time_factor * neighbor_factor * geo_factor
        
        # Apply intervention effect if after response day
        if t >= self.response_day:
            # Gradual implementation of intervention over 7 days
            days_since_response = min(t - self.response_day, 7)
            ramp_factor = days_since_response / 7.0
            
            # Get effectiveness based on response day
            reduction = INTERVENTION_EFFECTS.get(self.response_day, 0.5)
            
            # Response effectiveness varies slightly by location
            location_effect = 1.0 - 0.1 * np.sin(x_coord/5000 + y_coord/6000)
            
            # Apply reduction with ramping up and geographic influence
            return beta * (1 - (reduction * ramp_factor * location_effect))
        else:
            return beta

    def determine_initial_infected(self, household):
        """Determine if household has initial infections with geographic influence"""
        risk = household['lepto_risk']
        
        # Add coordinate-based variation to create geographic patterns
        x_coord = household['x_coord']
        y_coord = household['y_coord']
        
        # Geographic influence factor
        geo_factor = 1.0 + 0.2 * np.sin(x_coord/2000 + self.response_day/40) * np.cos(y_coord/2500)
        
        # Risk-based probabilities
        if risk >= 3:
            prob = 0.6 * geo_factor  # High risk
        elif risk >= 2:
            prob = 0.3 * geo_factor  # Medium risk
        elif risk >= 1:
            prob = 0.1 * geo_factor  # Low risk
        else:
            prob = 0.01 * geo_factor  # Very low risk
        
        # Cap probability at 0.9
        prob = min(prob, 0.9)
        
        return 1 if np.random.rand() < prob else 0

    def seir_model(self, t, y, household, neighbor_infectious):
        """SEIR model differential equations"""
        S, E, I, R = y
        
        # Neighbor influence factor
        neighbor_factor = 1 + (neighbor_infectious * 0.05)
        
        # Get transmission rate
        beta = self.transmission_rate(t, household, neighbor_factor)
        
        # SEIR differential equations
        dSdt = -beta * S * I
        dEdt = beta * S * I - SIGMA * E
        dIdt = SIGMA * E - GAMMA * I
        dRdt = GAMMA * I
        
        return [dSdt, dEdt, dIdt, dRdt]

    def run_household_simulation(self, idx, days=60):
        """Run simulation for a single household"""
        household = self.centroids.iloc[idx]
        
        # Skip vacant properties
        if household['vacancy']:
            return None
            
        # Determine initial infections
        initial_infected = self.determine_initial_infected(household)
        if initial_infected == 0:
            return None

        # Initial state
        household_size = household['house_size']
        y0 = [household_size - initial_infected, 0, initial_infected, 0]
        
        # Get neighbor indices
        neighbors_idx = self.neighbors[idx]
        
        # Location-specific neighbor influence
        x_coord = household['x_coord']
        y_coord = household['y_coord']
        location_weight = 1.0 + 0.2 * np.sin(x_coord/4000 + y_coord/3000)
        
        # Calculate neighbor influence - varies by location and response scenario
        neighbor_factor = (len(neighbors_idx) / 10.0) * location_weight * (1.0 + 0.1 * np.sin(self.response_day/10))
        
        # Time points
        t_eval = np.arange(days + 1)
        
        # Solve the differential equations
        solution = solve_ivp(
            lambda t, y: self.seir_model(t, y, household, neighbor_factor),
            [0, days], y0, t_eval=t_eval, method='RK45', rtol=1e-5, atol=1e-7
        )
        
        # Create results dataframe
        df = pd.DataFrame({
            "day": t_eval,
            "susceptible": solution.y[0],
            "exposed": solution.y[1],
            "infectious": solution.y[2],
            "recovered": solution.y[3],
            "household_id": idx  # Add household ID to associate with centroids
        })
        
        # Calculate cumulative cases and deaths
        df['cumulative_cases'] = initial_infected + np.cumsum(SIGMA * df['exposed'].shift(1, fill_value=0))
        
        # Case fatality rate based on demographics
        cfr = CASE_FATALITY_RATES.get(household['age_cat'], 0.05)
        cfr *= 1.25 if household['disability'] else 1
        
        # Earlier responses reduce mortality through better treatment
        mortality_modifier = 1.0
        if self.response_day <= 10:
            mortality_modifier = 0.85  # 15% reduction with earliest response
        elif self.response_day <= 20:
            mortality_modifier = 0.90  # 10% reduction
        elif self.response_day <= 30:
            mortality_modifier = 0.95  # 5% reduction
        
        df['deaths'] = df['cumulative_cases'] * cfr * mortality_modifier
        
        return df

    def run_simulation(self, days=60, n_jobs=4):
        """Run simulation for all households"""
        logger.info(f"Starting simulation with response day {self.response_day}")
        
        # Run simulations in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.run_household_simulation)(idx, days)
            for idx in range(len(self.centroids))
        )
        
        # Filter out None results
        results = [r for r in results if r is not None]
        
        # Count initial infections
        initial_infections = len(results)
        logger.info(f"Generated {initial_infections} initial infections")
        
        # Aggregate daily totals
        daily_totals = pd.DataFrame({"day": np.arange(days + 1)})
        for key in ['susceptible', 'exposed', 'infectious', 'recovered', 'cumulative_cases', 'deaths']:
            daily_totals[key] = sum(df[key] for df in results)
        
        # Log final statistics
        logger.info(f"Simulation complete. Final cases: {daily_totals['cumulative_cases'].iloc[-1]:.1f}, "
                   f"Deaths: {daily_totals['deaths'].iloc[-1]:.1f}")
        
        return daily_totals, results

def run_comparison(centroids, response_days=[10, 20, 30, 40], days=60, n_jobs=4):
    """Run model comparison with different response days"""
    results = {}
    
    for response_day in response_days:
        logger.info(f"=== Running model with response at day {response_day} ===")
        model = LeptoModel(centroids, response_day)
        daily_totals, household_results = model.run_simulation(days, n_jobs)
        results[response_day] = (daily_totals, household_results)
    
    # Print comparison summary
    logger.info("\n=== COMPARISON SUMMARY ===")
    print(f"{'Response Day':<12} | {'Total Cases':>12} | {'Total Deaths':>12} | {'CFR %':>8}")
    print("-" * 50)
    
    # Calculate total population
    total_population = 0
    for _, household in centroids.iterrows():
        if household["vacancy"] == 0:
            total_population += household["house_size"]
    
    for day, (daily_totals, _) in results.items():
        final = daily_totals.iloc[-1]
        cases = final['cumulative_cases']
        deaths = final['deaths']
        cfr = (deaths / cases * 100) if cases > 0 else 0
        attack_rate = (cases / total_population * 100) if total_population > 0 else 0
        
        print(f"{day:<12} | {cases:>12.1f} | {deaths:>12.1f} | {cfr:>8.2f}% | {attack_rate:>8.2f}%")
    
    # Compare earliest vs latest
    if len(response_days) > 1:
        earliest = min(response_days)
        latest = max(response_days)
        
        earliest_results = results[earliest][0].iloc[-1]
        latest_results = results[latest][0].iloc[-1]
        
        cases_prevented = latest_results['cumulative_cases'] - earliest_results['cumulative_cases']
        deaths_prevented = latest_results['deaths'] - earliest_results['deaths']
        
        print(f"\nEarly response (Day {earliest}) vs late response (Day {latest}):")
        print(f"  Prevented cases: {cases_prevented:.1f}")
        print(f"  Lives saved: {deaths_prevented:.1f}")
    
    return results