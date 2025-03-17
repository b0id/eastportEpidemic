import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# Setup basic logging
logger = logging.getLogger(__name__)

def plot_spatial_result(grid_with_data, day, response_day, output_dir="outputs"):
    """Plot spatial distribution of infection with improved visualization"""
    if grid_with_data is None:
        logger.error("Cannot create spatial plot: grid data is None")
        return
        
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Create custom colormap (green to yellow to red)
    cmap = plt.cm.get_cmap('YlOrRd')
    
    # Determine appropriate color scale based on data
    max_infection = grid_with_data["percent_infected"].max()
    if max_infection < 1:  # Very early in outbreak, use a smaller scale
        vmax = 2  # Use a fixed scale of 0-2% for early days to show detail
    elif max_infection < 5:
        vmax = 5  # Use a 0-5% scale
    else:
        vmax = max(10, max_infection)  # Use at least 0-10% or higher if needed
        
    # Plot grid with infection rate - with a fixed scale for consistency
    grid_with_data.plot(column="percent_infected", 
                       ax=ax, 
                       cmap=cmap,
                       vmin=0, vmax=vmax,
                       legend=True,
                       legend_kwds={'label': "Percent of Population Infected",
                                   'orientation': "horizontal"})
    
    # Create SEIR category markers for legend
    susceptible_patch = plt.matplotlib.patches.Patch(color='blue', label='Susceptible')
    exposed_patch = plt.matplotlib.patches.Patch(color='orange', label='Exposed')
    infectious_patch = plt.matplotlib.patches.Patch(color='red', label='Infectious')
    recovered_patch = plt.matplotlib.patches.Patch(color='green', label='Recovered')
    
    # Add legend manually
    plt.legend(handles=[susceptible_patch, exposed_patch, infectious_patch, recovered_patch],
              loc='lower right')
    
    # Add title
    plt.title(f"Leptospirosis Spatial Distribution - Day {day}\nResponse started at Day {response_day}")
    
    # Add statistics text box
    try:
        stats_text = (
            f"Day {day} Statistics:\n"
            f"Susceptible: {grid_with_data['current_susceptible'].sum():.0f}\n"
            f"Exposed: {grid_with_data['current_exposed'].sum():.0f}\n"
            f"Infectious: {grid_with_data['current_infectious'].sum():.0f}\n"
            f"Recovered: {grid_with_data['current_recovered'].sum():.0f}\n"
            f"Max Infection %: {max_infection:.2f}%"
        )
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=12, 
                  bbox=dict(facecolor='white', alpha=0.7))
    except:
        pass
    
    # Save figure
    plt.tight_layout()
    plt.savefig(f"{output_dir}/spatial_day_{day}_response_{response_day}.png", dpi=300)
    plt.close()
    
    return True

def create_lepto_distribution_maps(centroids, household_results, grid, interval=10, days=60, output_dir="distributions", response_day=None):
    """
    Create distribution maps of leptospirosis infection states at specified intervals.
    Enhanced version to show SEIR dynamics more clearly.
    
    Args:
        centroids: GeoDataFrame with centroid data
        household_results: Results from the simulation for a specific response day
        grid: GeoDataFrame with the hexagonal grid
        interval: Time interval (in days) between maps
        days: Total simulation days
        output_dir: Directory to save output files
        response_day: The response day for this simulation (for title)
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logger.info(f"Creating enhanced SEIR distribution maps at {interval}-day intervals")
    
    # Join household centroids with grid for spatial mapping
    centroids_with_grid = gpd.sjoin(centroids, grid, how="left", predicate="within")
    
    # Create a mapping from centroids to grid cells
    centroids_grid_map = {}
    for idx, row in centroids_with_grid.iterrows():
        if "index_right" in row and not pd.isna(row["index_right"]):
            centroids_grid_map[idx] = int(row["index_right"])  # Convert to int to avoid NaN issues
    
    # Process each time interval
    for day in range(0, days+1, interval):
        logger.info(f"Processing day {day}")
        
        # Initialize grid data for this day
        grid_data = grid.copy()
        
        # Initialize SEIR columns with float type to avoid dtype warnings
        for column in ["current_susceptible", "current_exposed", "current_infectious", 
                      "current_recovered", "cumulative_cases", "deaths", "house_size"]:
            grid_data[column] = 0.0  # Using float instead of int
        
        # Process each household result
        for result in household_results:
            if result is None:
                continue
            
            # Get data for current day
            day_data = result[result["day"] == day]
            if day_data.empty:
                continue
            
            # Get household index - try different approaches
            try:
                if "household_id" in day_data.columns:
                    household_idx = int(day_data["household_id"].iloc[0])
                else:
                    # Otherwise use the first index value
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
        
        # Calculate total population and infected percentage
        grid_data["total_population"] = grid_data["house_size"].replace(0, 1)  # Prevent division by zero
        grid_data["percent_infected"] = (
            (grid_data["current_exposed"] + grid_data["current_infectious"] + grid_data["current_recovered"]) / 
            grid_data["total_population"] * 100
        ).fillna(0)
        
        # Cap percent_infected at 100%
        grid_data["percent_infected"] = grid_data["percent_infected"].clip(upper=100)
        
        # Calculate dominant state and state percentages
        grid_data["dominant_state"] = grid_data.apply(
            lambda row: "None" if row["total_population"] == 0 else
            max([
                ("S", row["current_susceptible"]), 
                ("E", row["current_exposed"]), 
                ("I", row["current_infectious"]), 
                ("R", row["current_recovered"])
            ], key=lambda x: x[1])[0],
            axis=1
        )
        
        grid_data["s_percent"] = (grid_data["current_susceptible"] / grid_data["total_population"] * 100).fillna(0)
        grid_data["e_percent"] = (grid_data["current_exposed"] / grid_data["total_population"] * 100).fillna(0)
        grid_data["i_percent"] = (grid_data["current_infectious"] / grid_data["total_population"] * 100).fillna(0)
        grid_data["r_percent"] = (grid_data["current_recovered"] / grid_data["total_population"] * 100).fillna(0)
        
        # Create the visualization with SEIR dynamics
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Define colors for SEIR states
        state_colors = {
            "S": "blue",
            "E": "orange",
            "I": "red",
            "R": "green",
            "None": "lightgrey"
        }
        
        # Plot each cell with color based on dominant state
        for state, color in state_colors.items():
            subset = grid_data[grid_data["dominant_state"] == state]
            if not subset.empty:
                subset.plot(ax=ax, color=color, edgecolor='black', linewidth=0.5, alpha=0.8)
        
        # Create legend for SEIR categories
        susceptible_patch = mpatches.Patch(color='blue', label='Susceptible')
        exposed_patch = mpatches.Patch(color='orange', label='Exposed')
        infectious_patch = mpatches.Patch(color='red', label='Infectious')
        recovered_patch = mpatches.Patch(color='green', label='Recovered')
        
        # Add legend manually
        plt.legend(handles=[susceptible_patch, exposed_patch, infectious_patch, recovered_patch],
                  loc='lower right')
        
        # Add title
        if response_day is not None:
            plt.title(f"Leptospirosis Spatial Distribution - Day {day}\nResponse started at Day {response_day}")
        else:
            plt.title(f"Leptospirosis Spatial Distribution - Day {day}")
            
        # Add statistics text box
        stats_text = (
            f"Day {day} Statistics:\n"
            f"Susceptible: {grid_data['current_susceptible'].sum():.0f}\n"
            f"Exposed: {grid_data['current_exposed'].sum():.0f}\n"
            f"Infectious: {grid_data['current_infectious'].sum():.0f}\n"
            f"Recovered: {grid_data['current_recovered'].sum():.0f}\n"
            f"Max Infection %: {grid_data['percent_infected'].max():.2f}%"
        )
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=12, 
                  bbox=dict(facecolor='white', alpha=0.7))
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{output_dir}/lepto_distribution_day_{day}.png", dpi=300)
        plt.close()
        
        # Create pie charts visualization to show full SEIR dynamics
        fig2, ax2 = plt.subplots(figsize=(15, 10))
        
        # Plot the base map
        grid.plot(ax=ax2, color='lightgrey', edgecolor='black', linewidth=0.5)
        
        # Draw pie charts on active cells
        active_cells = grid_data[(grid_data["current_susceptible"] > 0) | 
                                 (grid_data["current_exposed"] > 0) | 
                                 (grid_data["current_infectious"] > 0) | 
                                 (grid_data["current_recovered"] > 0)]
        
        for idx, row in active_cells.iterrows():
            if row["total_population"] > 0:
                # Get the center of the cell
                center_x, center_y = row.geometry.centroid.x, row.geometry.centroid.y
                
                # Calculate SEIR proportions
                s_prop = row["current_susceptible"] / row["total_population"]
                e_prop = row["current_exposed"] / row["total_population"]
                i_prop = row["current_infectious"] / row["total_population"]
                r_prop = row["current_recovered"] / row["total_population"]
                
                # Size of pie chart (proportional to population)
                size = min(30, max(5, row["total_population"] / 5))
                
                # Draw pie chart if there's any non-zero value
                if s_prop + e_prop + i_prop + r_prop > 0:
                    pie_data = [s_prop, e_prop, i_prop, r_prop]
                    if sum(pie_data) > 0:
                        ax2.pie([p if p > 0 else 0 for p in pie_data], 
                               colors=['blue', 'orange', 'red', 'green'],
                               center=(center_x, center_y), 
                               radius=size,
                               wedgeprops=dict(width=size*0.6, edgecolor='w'))
        
        # Add title
        if response_day is not None:
            plt.title(f"Leptospirosis SEIR Dynamics - Day {day}\nResponse started at Day {response_day}")
        else:
            plt.title(f"Leptospirosis SEIR Dynamics - Day {day}")
        
        # Add legend
        plt.legend(handles=[susceptible_patch, exposed_patch, infectious_patch, recovered_patch],
                  loc='lower right')
        
        # Add the same statistics text box
        plt.figtext(0.02, 0.02, stats_text, fontsize=12, 
                  bbox=dict(facecolor='white', alpha=0.7))
        
        # Save figure
        plt.tight_layout()
        plt.savefig(f"{output_dir}/lepto_dynamics_day_{day}.png", dpi=300)
        plt.close()
        
        logger.info(f"Created enhanced distribution maps for day {day} with max infection rate of {grid_data['percent_infected'].max():.2f}%")
    
    logger.info(f"Completed all enhanced leptospirosis distribution maps in {output_dir}")
    return True