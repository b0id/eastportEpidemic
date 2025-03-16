import os
import sys
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging
from data_preparation import prepare_centroids, plot_statistics
from lepto_model import LeptoModel, run_comparison
from results_visualization import create_hexgrid_animation, create_response_comparison_charts, create_risk_factor_visualization, generate_report_data
from distribution_visualization import create_lepto_distribution_maps, plot_spatial_result

# Setup logging for diagnostics
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_directory(directory):
    """Ensure a directory exists, create if not."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def run_simulation(centroids, response_day=10, days=60):
    """Run simulation with the LeptoModel class"""
    logger.info(f"Creating model with response day {response_day} (unique seed: {42 + response_day})")
    
    # Create model with response day
    model = LeptoModel(centroids, response_day)
    
    # Run simulation
    daily_totals, household_results = model.run_simulation(days=days, n_jobs=4)
    
    return daily_totals, household_results

def plot_outbreak_curve(daily_totals, response_day, output_dir="outputs"):
    """Plot the outbreak curve"""
    # Create output directory if it doesn't exist
    ensure_directory(output_dir)
    
    # Plot SEIR curves
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(daily_totals["day"], daily_totals["susceptible"], label="Susceptible", color="blue")
    ax.plot(daily_totals["day"], daily_totals["exposed"], label="Exposed", color="orange")
    ax.plot(daily_totals["day"], daily_totals["infectious"], label="Infectious", color="red")
    ax.plot(daily_totals["day"], daily_totals["recovered"], label="Recovered", color="green")
    
    # Add vertical line for intervention day
    ax.axvline(x=response_day, linestyle="--", color="black", label=f"Intervention (Day {response_day})")
    
    # Add labels and title
    ax.set_xlabel("Days")
    ax.set_ylabel("Number of People")
    ax.set_title(f"Leptospirosis Outbreak - Response at Day {response_day}")
    ax.legend()
    ax.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/outbreak_curve_response_{response_day}.png", dpi=300)
    plt.close()
    
    # Plot cumulative cases and deaths
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    color = 'tab:red'
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Cumulative Cases', color=color)
    ax1.plot(daily_totals["day"], daily_totals["cumulative_cases"], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create second y-axis for deaths
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Deaths', color=color)
    ax2.plot(daily_totals["day"], daily_totals["deaths"], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add vertical line for intervention day
    ax1.axvline(x=response_day, linestyle="--", color="black", label=f"Intervention (Day {response_day})")
    
    # Add title
    plt.title(f"Leptospirosis Outbreak - Cumulative Impact (Response at Day {response_day})")
    fig.tight_layout()
    plt.savefig(f"{output_dir}/cumulative_impact_response_{response_day}.png", dpi=300)
    plt.close()

def create_animation_data(centroids, household_results, grid, days=60, output_dir="outputs"):
    """Create GeoJSON files for animation"""
    # Create output directory if it doesn't exist
    ensure_directory(output_dir)
    
    try:
        # Join household centroids with grid
        centroids_with_grid = gpd.sjoin(centroids, grid, how="left", predicate="within")
        
        # For each day, create a grid file with aggregated statistics
        for day in range(0, days+1, 10):  # Creating every 10th day to save processing time
            logger.info(f"Creating animation data for day {day}")
            
            # Get data for this day - this is different for each result format
            grid_data = grid.copy()
            
            # Initialize SEIR columns with float type to avoid dtype warnings
            for column in ["current_susceptible", "current_exposed", "current_infectious", 
                         "current_recovered", "cumulative_cases", "deaths", "house_size"]:
                grid_data[column] = 0.0  # Using float instead of int
            
            # Create a mapping from centroids to grid cells
            centroids_grid_map = {}
            for idx, row in centroids_with_grid.iterrows():
                if "index_right" in row and not pd.isna(row["index_right"]):
                    centroids_grid_map[idx] = int(row["index_right"])  # Convert to int to avoid NaN issues
            
            # Process each household result
            for result in household_results:
                if result is None:
                    continue
                
                # Get data for current day
                day_data = result[result["day"] == day]
                if day_data.empty:
                    continue
                
                try:
                    # Get household index - try different approaches based on data format
                    if "household_id" in day_data.columns:
                        household_idx = int(day_data["household_id"].iloc[0])
                    else:
                        # If household_id isn't in columns, try using the first index
                        # This assumes household indices match with centroids indices
                        household_idx = result.index[0]
                    
                    # Skip if household is not mapped to a grid cell
                    if household_idx not in centroids_grid_map:
                        continue
                    
                    # Get grid cell for this household
                    grid_idx = centroids_grid_map[household_idx]
                    
                    # Skip if grid index is not valid
                    if grid_idx not in grid_data.index:
                        continue
                    
                    # Add household data to grid cell (explicitly convert to float)
                    grid_data.loc[grid_idx, "current_susceptible"] += float(day_data["susceptible"].iloc[0])
                    grid_data.loc[grid_idx, "current_exposed"] += float(day_data["exposed"].iloc[0])
                    grid_data.loc[grid_idx, "current_infectious"] += float(day_data["infectious"].iloc[0])
                    grid_data.loc[grid_idx, "current_recovered"] += float(day_data["recovered"].iloc[0])
                    
                    # For cases and deaths, make sure these columns exist
                    if "cumulative_cases" in day_data.columns:
                        grid_data.loc[grid_idx, "cumulative_cases"] += float(day_data["cumulative_cases"].iloc[0])
                    
                    if "deaths" in day_data.columns:
                        grid_data.loc[grid_idx, "deaths"] += float(day_data["deaths"].iloc[0])
                    
                    # Add household size
                    if household_idx in centroids.index:
                        grid_data.loc[grid_idx, "house_size"] += float(centroids.loc[household_idx, "house_size"])
                except (KeyError, TypeError, ValueError) as e:
                    # Skip this result if there's an issue with indexing
                    continue
            
            # Calculate percent of population infected - avoid division by zero
            grid_data["total_population"] = grid_data["house_size"].replace(0, 1)  # Prevent division by zero
            grid_data["percent_infected"] = (grid_data["cumulative_cases"] / grid_data["total_population"]) * 100
            grid_data["percent_infected"] = grid_data["percent_infected"].fillna(0)
            
            # Save to GeoJSON
            grid_data.to_file(f"{output_dir}/grid_day_{day}.geojson", driver="GeoJSON")
        
        logger.info(f"Created animation data files in {output_dir}")
        return grid_data  # Return the last day's data for visualization
    except Exception as e:
        logger.error(f"Error creating animation data: {e}")
        return None

def compare_response_times(centroids, grid, response_times=[5, 10, 15, 20], days=60):
    """Compare different response times"""
    # DIAGNOSTIC: Verify different response times
    logger.info(f"Running comparison with response times: {response_times}")
    
    # Use our run_comparison function from lepto_model
    results = {}
    
    for response_day in response_times:
        logger.info(f"Simulating response at day {response_day}")
        daily_totals, household_results = run_simulation(centroids, response_day=response_day, days=days)
        results[response_day] = (daily_totals, household_results)
        
        # Plot results for this response time
        plot_outbreak_curve(daily_totals, response_day, output_dir="outputs")
        
        # Create animation data for this scenario
        animation_dir = f"outputs/animations/response_{response_day}"
        ensure_directory(animation_dir)
        grid_with_data = create_animation_data(centroids, household_results, grid, days=days, output_dir=animation_dir)
        
        # Plot spatial result for last day if grid data is available
        if grid_with_data is not None:
            plot_spatial_result(grid_with_data, days, response_day, output_dir="outputs")
        
        # NEW: Create leptospirosis distribution maps at regular intervals
        distributions_dir = f"outputs/distributions/response_{response_day}"
        ensure_directory(distributions_dir)
        
        # Pass the response_day parameter to the distribution function
        create_lepto_distribution_maps(
            centroids, 
            household_results, 
            grid, 
            interval=10,  # Create a map every 10 days
            days=days, 
            output_dir=distributions_dir,
            response_day=response_day  # Add this parameter
        )
    
    # Plot comparison of different response times
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot cumulative cases with different colors
    colors = {5: 'green', 10: 'blue', 15: 'orange', 20: 'red', 30: 'purple', 40: 'brown'}
    
    for response_day, (daily_totals, _) in results.items():
        color = colors.get(response_day, 'black')
        ax1.plot(daily_totals["day"], daily_totals["cumulative_cases"], 
                label=f"Response Day {response_day}", color=color)
        # Add vertical line at intervention point
        ax1.axvline(x=response_day, linestyle="--", color=color, alpha=0.5)
    
    ax1.set_xlabel("Days")
    ax1.set_ylabel("Cumulative Cases")
    ax1.set_title("Impact of Response Timing on Cumulative Cases")
    ax1.legend()
    ax1.grid(True)
    
    # Plot deaths
    for response_day, (daily_totals, _) in results.items():
        color = colors.get(response_day, 'black')
        ax2.plot(daily_totals["day"], daily_totals["deaths"], 
                label=f"Response Day {response_day}", color=color)
        # Add vertical line at intervention point
        ax2.axvline(x=response_day, linestyle="--", color=color, alpha=0.5)
    
    ax2.set_xlabel("Days")
    ax2.set_ylabel("Deaths")
    ax2.set_title("Impact of Response Timing on Deaths")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("outputs/response_time_comparison.png", dpi=300)
    plt.close()
    
    # DIAGNOSTIC: Print final values
    logger.info("Final outcome values:")
    for rd, (dt, _) in results.items():
        final_cases = dt["cumulative_cases"].iloc[-1]
        final_deaths = dt["deaths"].iloc[-1]
        logger.info(f"Response day {rd}: Cases={final_cases:.1f}, Deaths={final_deaths:.1f}")
    
    return results

def main():
    """Main function to run the leptospirosis outbreak case study."""
    logger.info("Starting Leptospirosis Case Study")
    
    # Set up directories
    input_dir = "input"
    output_dir = "outputs"
    ensure_directory(input_dir)
    ensure_directory(output_dir)
    
    # File paths
    raw_centroids_path = os.path.join(input_dir, "raw_centroids.geojson")
    grid_path = os.path.join(input_dir, "grid.geojson")
    prepared_centroids_path = os.path.join(output_dir, "centroids_prepared.geojson")
    
    # Step 1: Check if we have input files
    if not os.path.exists(raw_centroids_path):
        logger.error(f"Required input file not found: {raw_centroids_path}")
        print("Please export your centroid layer from QGIS with the density_no and FloodRisk fields.")
        sys.exit(1)
        
    if not os.path.exists(grid_path):
        logger.error(f"Required input file not found: {grid_path}")
        print("Please export your hexagonal grid layer from QGIS.")
        sys.exit(1)
    
    # Step 2: Prepare centroids data
    print("\n=== Preparing Statistical Data for Centroids ===")
    try:
        centroids = prepare_centroids(raw_centroids_path, prepared_centroids_path)
        logger.info(f"Successfully prepared data for {len(centroids)} centroids")
        
        # Create statistical visualizations
        stats_dir = os.path.join(output_dir, "statistics")
        ensure_directory(stats_dir)
        plot_statistics(centroids, output_dir=stats_dir)
        logger.info(f"Created statistical visualizations in {stats_dir}")
    except Exception as e:
        logger.error(f"Error preparing centroids data: {e}")
        sys.exit(1)
    
    # Step 3: Load grid
    try:
        grid = gpd.read_file(grid_path)
        logger.info(f"Successfully loaded grid with {len(grid)} cells")
    except Exception as e:
        logger.error(f"Error loading grid: {e}")
        sys.exit(1)
    
    # Step 4: Run simulations for different response times
    print("\n=== Running SEIR Model Simulations ===")
    # Using the requested response days
    response_times = [10, 20, 30, 40]
    simulation_output_dir = os.path.join(output_dir, "simulations")
    ensure_directory(simulation_output_dir)
    
    try:
        # Use our compare_response_times function
        results = compare_response_times(centroids, grid, response_times, days=60)
        
        logger.info("Completed all simulations.")
    except Exception as e:
        logger.error(f"Error during simulation: {e}", exc_info=True)
        sys.exit(1)
    
    # Step 5: Create visualizations and reports
    print("\n=== Creating Visualizations and Reports ===")
    try:
        # Create response comparison charts
        logger.info("Creating response comparison charts...")
        stats_df = create_response_comparison_charts(results, output_dir=output_dir)
        logger.info("Response comparison completed.")
        
        # Create risk factor visualizations
        logger.info("Creating risk factor visualizations...")
        create_risk_factor_visualization(centroids, output_dir=output_dir)
        
        # Generate report data
        logger.info("Generating report data...")
        summary_stats, outbreak_df = generate_report_data(results, centroids, output_dir=output_dir)
        
        # Print summary of findings
        print("\n=== Summary of Findings ===")
        print(f"Total population analyzed: {int(summary_stats.loc[0, 'Value'])}")
        print(f"Properties in flood zone: {summary_stats.loc[2, 'Value']:.1f}%")
        
        print("\nOutbreak statistics by response time:")
        for _, row in outbreak_df.iterrows():
            print(f"  Response at Day {int(row['Response Day'])}: {int(row['Total Cases'])} cases, "
                 f"{int(row['Total Deaths'])} deaths, "
                 f"{row['Attack Rate (%)']:.1f}% attack rate")
        
        earliest_response = outbreak_df.loc[outbreak_df['Response Day'] == min(response_times)]
        latest_response = outbreak_df.loc[outbreak_df['Response Day'] == max(response_times)]
        
        deaths_prevented = latest_response['Total Deaths'].values[0] - earliest_response['Total Deaths'].values[0]
        cases_prevented = latest_response['Total Cases'].values[0] - earliest_response['Total Cases'].values[0]
        
        print(f"\nEarly response (Day {min(response_times)}) compared to late response (Day {max(response_times)}):")
        print(f"  Prevented {int(cases_prevented)} cases")
        print(f"  Saved {int(deaths_prevented)} lives")
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}", exc_info=True)
        sys.exit(1)
    
    print("\n=== Case Study Complete ===")
    print(f"All outputs have been saved to the {output_dir} directory.")
    print("You can find:")
    print(f"  - Prepared data in {output_dir}")
    print(f"  - Statistical visualizations in {os.path.join(output_dir, 'statistics')}")
    print(f"  - Simulation results in {os.path.join(output_dir, 'simulations')}")
    print(f"  - Comparison charts in {os.path.join(output_dir, 'comparisons')}")
    print(f"  - Risk factor analysis in {os.path.join(output_dir, 'risk_factors')}")
    print(f"  - Report data in {os.path.join(output_dir, 'report')}")
    print(f"  - Leptospirosis distributions in {os.path.join(output_dir, 'distributions')}")

if __name__ == "__main__":
    main()