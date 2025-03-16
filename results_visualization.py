import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

def create_hexgrid_animation(output_dir="outputs", animation_dir="animations"):
    """
    Create an animation from daily grid files.
    """
    # Create animation directory if it doesn't exist
    if not os.path.exists(animation_dir):
        os.makedirs(animation_dir)
    
    # Get list of all daily grid files
    grid_files = sorted([f for f in os.listdir(output_dir) if f.startswith("grid_day_") and f.endswith(".geojson")])
    
    if not grid_files:
        print("No grid files found to create animation.")
        return
    
    # Get max day
    max_day = len(grid_files) - 1
    
    # Load the first grid file to get geometry
    first_grid = gpd.read_file(os.path.join(output_dir, grid_files[0]))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create custom colormap (green to yellow to red)
    cmap = LinearSegmentedColormap.from_list("infection_cmap", 
                                           [(0, 'lightgreen'), 
                                            (0.5, 'yellow'), 
                                            (1, 'darkred')])
    
    # Function to update the plot for each frame
    def update(frame):
        ax.clear()
        
        # Load grid data for this day
        grid_file = os.path.join(output_dir, f"grid_day_{frame}.geojson")
        grid_data = gpd.read_file(grid_file)
        
        # Plot grid with infection rate
        grid_data.plot(column="percent_infected", 
                     ax=ax, 
                     cmap=cmap,
                     legend=True,
                     legend_kwds={'label': "Percent of Population Infected",
                                  'orientation': "horizontal"})
        
        # Add title
        ax.set_title(f"Leptospirosis Outbreak - Day {frame}")
        
        return ax,
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=range(0, max_day+1, 1), 
                       interval=200, blit=False)
    
    # Save animation
    ani.save(os.path.join(animation_dir, "outbreak_animation.mp4"), 
            writer='ffmpeg', fps=5, dpi=300)
    
    plt.close()
    
    print(f"Animation saved to {os.path.join(animation_dir, 'outbreak_animation.mp4')}")

def create_response_comparison_charts(results_dict, output_dir="outputs"):
    """
    Create charts comparing different response scenarios.
    
    Args:
        results_dict: Dictionary of results keyed by response day
        output_dir: Directory to save output files
    """
    # Create comparison charts directory
    comparison_dir = os.path.join(output_dir, "comparisons")
    if not os.path.exists(comparison_dir):
        os.makedirs(comparison_dir)
    
    # Create line colors for different response times
    colors = {
        5: 'green',
        10: 'blue',
        15: 'orange',
        20: 'red'
    }
    
    # Plot comparison of total cases
    plt.figure(figsize=(12, 8))
    
    for response_day, (daily_totals, _) in results_dict.items():
        plt.plot(daily_totals["day"], daily_totals["cumulative_cases"], 
                label=f"Response at Day {response_day}", 
                color=colors.get(response_day, 'black'),
                linewidth=2)
    
    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Cumulative Cases", fontsize=12)
    plt.title("Impact of Response Timing on Total Leptospirosis Cases", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(comparison_dir, "cases_comparison.png"), dpi=300)
    plt.close()
    
    # Plot comparison of active cases (infectious)
    plt.figure(figsize=(12, 8))
    
    for response_day, (daily_totals, _) in results_dict.items():
        plt.plot(daily_totals["day"], daily_totals["infectious"], 
                label=f"Response at Day {response_day}", 
                color=colors.get(response_day, 'black'),
                linewidth=2)
    
    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Active Cases", fontsize=12)
    plt.title("Impact of Response Timing on Active Leptospirosis Cases", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(comparison_dir, "active_cases_comparison.png"), dpi=300)
    plt.close()
    
    # Plot comparison of deaths
    plt.figure(figsize=(12, 8))
    
    for response_day, (daily_totals, _) in results_dict.items():
        plt.plot(daily_totals["day"], daily_totals["deaths"], 
                label=f"Response at Day {response_day}", 
                color=colors.get(response_day, 'black'),
                linewidth=2)
    
    plt.xlabel("Days", fontsize=12)
    plt.ylabel("Deaths", fontsize=12)
    plt.title("Impact of Response Timing on Leptospirosis Mortality", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(comparison_dir, "deaths_comparison.png"), dpi=300)
    plt.close()
    
    # Create final statistics table
    final_stats = []
    
    for response_day, (daily_totals, _) in results_dict.items():
        final_day = daily_totals.iloc[-1]
        total_cases = final_day["cumulative_cases"]
        total_deaths = final_day["deaths"]
        cfr = (total_deaths / total_cases) * 100 if total_cases > 0 else 0
        
        final_stats.append({
            "Response Day": response_day,
            "Total Cases": total_cases,
            "Total Deaths": total_deaths,
            "Case Fatality Rate (%)": cfr,
            "Deaths Prevented": 0  # To be calculated
        })
    
    # Calculate deaths prevented (compared to latest response)
    latest_response = max(results_dict.keys())
    latest_deaths = next(stats["Total Deaths"] for stats in final_stats 
                       if stats["Response Day"] == latest_response)
    
    for stats in final_stats:
        stats["Deaths Prevented"] = latest_deaths - stats["Total Deaths"]
    
    # Create DataFrame and save to CSV
    stats_df = pd.DataFrame(final_stats)
    stats_df.to_csv(os.path.join(comparison_dir, "response_comparison_stats.csv"), index=False)
    
    # Create bar chart for deaths prevented
    plt.figure(figsize=(10, 6))
    
    response_days = [stats["Response Day"] for stats in final_stats]
    deaths_prevented = [stats["Deaths Prevented"] for stats in final_stats]
    
    bars = plt.bar(response_days, deaths_prevented, color=[colors.get(day, 'gray') for day in response_days])
    
    plt.xlabel("Response Day", fontsize=12)
    plt.ylabel("Deaths Prevented", fontsize=12)
    plt.title("Lives Saved by Earlier Response", fontsize=16)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f"{height:.1f}",
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "deaths_prevented.png"), dpi=300)
    plt.close()
    
    return stats_df

def create_risk_factor_visualization(centroids, output_dir="outputs"):
    """
    Create visualizations showing the relationship between risk factors and outcomes.
    
    Args:
        centroids: GeoDataFrame with centroid data
        output_dir: Directory to save output files
    """
    # Create risk factors directory
    risk_dir = os.path.join(output_dir, "risk_factors")
    if not os.path.exists(risk_dir):
        os.makedirs(risk_dir)
    
    # Plot relationship between flood risk and leptospirosis risk
    plt.figure(figsize=(10, 6))
    
    # Group by flood risk and calculate mean leptospirosis risk
    FloodRisk_groups = centroids.groupby("FloodRisk")["lepto_risk"].mean().reset_index()
    
    plt.bar(FloodRisk_groups["FloodRisk"], FloodRisk_groups["lepto_risk"], 
           color='steelblue')
    
    plt.xlabel("Flood Risk Level", fontsize=12)
    plt.ylabel("Average Leptospirosis Risk", fontsize=12)
    plt.title("Relationship Between Flood Risk and Leptospirosis Risk", fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(os.path.join(risk_dir, "FloodRisk_impact.png"), dpi=300)
    plt.close()
    
    # Plot relationship between age category and leptospirosis risk
    plt.figure(figsize=(10, 6))
    
    # Group by age category and calculate mean leptospirosis risk
    age_risk_groups = centroids.groupby("age_cat")["lepto_risk"].mean().reset_index()
    
    # Define order of age categories from young to elderly
    age_order = ["young", "adult", "senior", "elderly"]
    
    # Filter to include only the categories in age_order
    age_risk_groups = age_risk_groups[age_risk_groups["age_cat"].isin(age_order)]
    
    # Sort by the defined order
    age_risk_groups["age_cat"] = pd.Categorical(
        age_risk_groups["age_cat"], 
        categories=age_order, 
        ordered=True
    )
    age_risk_groups = age_risk_groups.sort_values("age_cat")
    
    plt.bar(age_risk_groups["age_cat"], age_risk_groups["lepto_risk"], 
           color='darkorange')
    
    plt.xlabel("Age Category", fontsize=12)
    plt.ylabel("Average Leptospirosis Risk", fontsize=12)
    plt.title("Relationship Between Age and Leptospirosis Risk", fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(os.path.join(risk_dir, "age_risk_impact.png"), dpi=300)
    plt.close()
    
    # Plot relationship between disability status and leptospirosis risk
    plt.figure(figsize=(10, 6))
    
    # Group by disability status and calculate mean leptospirosis risk
    disability_risk_groups = centroids.groupby("disability")["lepto_risk"].mean().reset_index()
    
    # Create labels for disability status
    disability_labels = {0: "No Disability", 1: "With Disability"}
    disability_risk_groups["disability_label"] = disability_risk_groups["disability"].map(disability_labels)
    
    plt.bar(disability_risk_groups["disability_label"], disability_risk_groups["lepto_risk"], 
           color='crimson')
    
    plt.xlabel("Disability Status", fontsize=12)
    plt.ylabel("Average Leptospirosis Risk", fontsize=12)
    plt.title("Impact of Disability on Leptospirosis Risk", fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    plt.savefig(os.path.join(risk_dir, "disability_risk_impact.png"), dpi=300)
    plt.close()
    
    # Create a combined visualization of all risk factors
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Flood risk
    axes[0].bar(FloodRisk_groups["FloodRisk"], FloodRisk_groups["lepto_risk"], 
              color='steelblue')
    axes[0].set_xlabel("Flood Risk Level", fontsize=12)
    axes[0].set_ylabel("Average Leptospirosis Risk", fontsize=12)
    axes[0].set_title("Flood Risk Impact", fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Age category
    axes[1].bar(age_risk_groups["age_cat"], age_risk_groups["lepto_risk"], 
              color='darkorange')
    axes[1].set_xlabel("Age Category", fontsize=12)
    axes[1].set_ylabel("Average Leptospirosis Risk", fontsize=12)
    axes[1].set_title("Age Impact", fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Disability status
    axes[2].bar(disability_risk_groups["disability_label"], disability_risk_groups["lepto_risk"], 
              color='crimson')
    axes[2].set_xlabel("Disability Status", fontsize=12)
    axes[2].set_ylabel("Average Leptospirosis Risk", fontsize=12)
    axes[2].set_title("Disability Impact", fontsize=14)
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(risk_dir, "combined_risk_factors.png"), dpi=300)
    plt.close()

def generate_report_data(results_dict, centroids, output_dir="outputs"):
    """
    Generate data for a comprehensive report.
    
    Args:
        results_dict: Dictionary of results keyed by response day
        centroids: GeoDataFrame with centroid data
        output_dir: Directory to save output files
    """
    # Create report directory
    report_dir = os.path.join(output_dir, "report")
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    # Get total population
    total_population = centroids[centroids["vacancy"] == 0]["house_size"].sum()
    
    # Get number of occupied properties
    occupied_properties = (centroids["vacancy"] == 0).sum()
    
    # Get distribution of age categories
    age_distribution = centroids[centroids["vacancy"] == 0]["age_cat"].value_counts().to_dict()
    
    # Calculate percentage of properties in flood zone
    in_flood_zone = (centroids["FloodRisk"] > 0).sum()
    flood_zone_percentage = (in_flood_zone / len(centroids)) * 100
    
    # Get highest risk areas (top 10% by lepto_risk)
    high_risk_threshold = centroids["lepto_risk"].quantile(0.9)
    high_risk_properties = centroids[centroids["lepto_risk"] >= high_risk_threshold]
    
    # Create summary statistics DataFrame
    summary_stats = pd.DataFrame({
        "Statistic": [
            "Total Population",
            "Occupied Properties",
            "Properties in Flood Zone (%)",
            "Young Households (%)",
            "Adult Households (%)",
            "Senior Households (%)",
            "Elderly Households (%)",
            "Households with Disability (%)",
            "Average Household Size",
            "High Risk Properties"
        ],
        "Value": [
            total_population,
            occupied_properties,
            flood_zone_percentage,
            (age_distribution.get("young", 0) / occupied_properties) * 100,
            (age_distribution.get("adult", 0) / occupied_properties) * 100,
            (age_distribution.get("senior", 0) / occupied_properties) * 100,
            (age_distribution.get("elderly", 0) / occupied_properties) * 100,
            (centroids[centroids["vacancy"] == 0]["disability"].sum() / occupied_properties) * 100,
            centroids[centroids["vacancy"] == 0]["house_size"].mean(),
            len(high_risk_properties)
        ]
    })
    
    # Save summary statistics
    summary_stats.to_csv(os.path.join(report_dir, "summary_statistics.csv"), index=False)
    
    # Get outbreak statistics for each response scenario
    outbreak_stats = []
    
    for response_day, (daily_totals, _) in results_dict.items():
        # Get final statistics
        final_day = daily_totals.iloc[-1]
        
        # Attack rate (% of population infected)
        attack_rate = (final_day["cumulative_cases"] / total_population) * 100
        
        # Peak day and peak active cases
        peak_day = daily_totals["infectious"].idxmax()
        peak_active_cases = daily_totals.loc[peak_day, "infectious"]
        
        outbreak_stats.append({
            "Response Day": response_day,
            "Total Cases": final_day["cumulative_cases"],
            "Total Deaths": final_day["deaths"],
            "Attack Rate (%)": attack_rate,
            "Case Fatality Rate (%)": (final_day["deaths"] / final_day["cumulative_cases"]) * 100 if final_day["cumulative_cases"] > 0 else 0,
            "Peak Active Cases": peak_active_cases,
            "Peak Day": daily_totals.loc[peak_day, "day"]
        })
    
    # Create DataFrame and save to CSV
    outbreak_df = pd.DataFrame(outbreak_stats)
    outbreak_df.to_csv(os.path.join(report_dir, "outbreak_statistics.csv"), index=False)
    
    return summary_stats, outbreak_df