import geopandas as gpd
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import logging

# Setup basic logging
logger = logging.getLogger(__name__)

def assign_household_size(density_norm):
    """
    Assign household size based on normalized density.
    Higher density areas tend to have smaller household sizes.
    """
    # Higher density areas have smaller household sizes in Eastport
    # Adjust weights based on density (higher density = more small households)
    
    if density_norm > 0.7:  # High density areas
        weights = [0.40, 0.35, 0.15, 0.10]  # More 1-2 person households
    elif density_norm > 0.3:  # Medium density
        weights = [0.30, 0.40, 0.20, 0.10]
    else:  # Low density areas
        weights = [0.20, 0.35, 0.25, 0.20]  # More larger households
    
    r = random.random()
    
    # Cumulative probabilities
    if r < weights[0]:
        return 1
    elif r < weights[0] + weights[1]:
        return 2
    elif r < weights[0] + weights[1] + weights[2]:
        return 3
    else:
        return random.randint(4, 6)

def assign_age_category():
    """
    Assign age category based on Eastport demographics.
    Eastport has an older than average population.
    """
    # Categories: "young", "adult", "senior", "elderly"
    categories = ["young", "adult", "senior", "elderly"]
    weights = [0.15, 0.30, 0.35, 0.20]  # Adjusted for older population
    
    return random.choices(categories, weights=weights)[0]

def determine_vacancy(density_norm):
    """
    Determine if a property is vacant based on density.
    Lower density areas have higher vacancy rates.
    """
    # Base vacancy rate for Eastport (approximately 25%)
    base_rate = 0.25
    
    # Adjust based on density (higher density = lower vacancy)
    # At max density, vacancy drops to about 5%
    # At min density, vacancy can be up to 40%
    adjusted_rate = base_rate - (density_norm * 0.20)
    
    # Ensure rate stays within reasonable bounds
    adjusted_rate = max(0.05, min(adjusted_rate, 0.40))
    
    # Return 1 if vacant, 0 if occupied
    return 1 if random.random() < adjusted_rate else 0

def assign_disability(age_cat):
    """
    Assign disability status based on age category.
    Older age categories have higher disability rates.
    """
    # Disability rates by age category
    rates = {
        "young": 0.08,
        "adult": 0.15,
        "senior": 0.25,
        "elderly": 0.35
    }
    
    # Get rate for this age category
    rate = rates.get(age_cat, 0.15)
    
    # Return 1 if has disability, 0 if not
    return 1 if random.random() < rate else 0

def calculate_lepto_risk(FloodRisk, age_cat, density_norm, disability, vacancy):
    """
    Calculate leptospirosis risk score based on multiple factors.
    """
    # Skip vacant properties
    if vacancy == 1:
        return 0
    
    # Base risk from flooding (0-5 scale, adjust if your FloodRisk is different)
    base_risk = FloodRisk * 1.0
    
    # Age vulnerability factor
    age_factor = {
        "young": 0.7,
        "adult": 1.0,
        "senior": 1.3,
        "elderly": 1.5
    }.get(age_cat, 1.0)
    
    # Density increases transmission
    density_factor = 0.8 + (density_norm * 0.4)  # 0.8-1.2 range
    
    # Disability increases vulnerability
    disability_factor = 1.25 if disability == 1 else 1.0
    
    # Calculate composite risk (normalized to 0-10 scale)
    risk = base_risk * age_factor * density_factor * disability_factor
    
    # Cap at 10
    return min(risk, 10)

def prepare_centroids(centroids_path, output_path):
    """
    Read centroid data, add statistical attributes, and save to output file.
    """
    # Read centroid data
    centroids = gpd.read_file(centroids_path)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Add additional fields
    centroids["house_size"] = centroids["density_no"].apply(assign_household_size)
    centroids["age_cat"] = centroids.apply(lambda x: assign_age_category(), axis=1)
    centroids["vacancy"] = centroids["density_no"].apply(determine_vacancy)
    centroids["disability"] = centroids["age_cat"].apply(assign_disability)
    
    # Calculate leptospirosis risk
    centroids["lepto_risk"] = centroids.apply(
        lambda x: calculate_lepto_risk(
            x["FloodRisk"], 
            x["age_cat"], 
            x["density_no"], 
            x["disability"], 
            x["vacancy"]
        ), 
        axis=1
    )
    
    # Ensure explicit coordinates
    centroids["x_coord"] = centroids.geometry.x
    centroids["y_coord"] = centroids.geometry.y
    
    # Log preparation results
    logger.info(f"Prepared {len(centroids)} centroids with the following statistics:")
    logger.info(f"  Total properties: {len(centroids)}")
    logger.info(f"  Occupied properties: {(centroids['vacancy'] == 0).sum()}")
    logger.info(f"  Average household size: {centroids[centroids['vacancy'] == 0]['house_size'].mean():.2f}")
    logger.info(f"  Average leptospirosis risk: {centroids[centroids['vacancy'] == 0]['lepto_risk'].mean():.2f}")
    
    # Save to output file
    centroids.to_file(output_path, driver="GeoJSON")
    
    return centroids

def plot_statistics(centroids, output_dir="stats"):
    """
    Create visualizations of the statistical data.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Plot household size distribution
    plt.figure(figsize=(10, 6))
    centroids["house_size"].value_counts().sort_index().plot(kind="bar", color="steelblue")
    plt.title("Household Size Distribution", fontsize=14)
    plt.xlabel("Household Size", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/household_size_distribution.png", dpi=300)
    plt.close()
    
    # Plot age category distribution
    plt.figure(figsize=(10, 6))
    centroids["age_cat"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=["skyblue", "lightgreen", "salmon", "khaki"])
    plt.title("Age Category Distribution", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/age_category_distribution.png", dpi=300)
    plt.close()
    
    # Plot vacancy rate by density
    plt.figure(figsize=(10, 6))
    # Create density bins
    centroids["density_bin"] = pd.cut(centroids["density_no"], bins=5)
    vacancy_by_density = centroids.groupby("density_bin")["vacancy"].mean() * 100  # Convert to percentage
    vacancy_by_density.plot(kind="bar", color="darkorange")
    plt.title("Vacancy Rate by Density", fontsize=14)
    plt.xlabel("Density (Normalized)", fontsize=12)
    plt.ylabel("Vacancy Rate (%)", fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/vacancy_by_density.png", dpi=300)
    plt.close()
    
    # Plot leptospirosis risk distribution
    plt.figure(figsize=(10, 6))
    plt.hist(centroids["lepto_risk"], bins=20, color="crimson", alpha=0.7)
    plt.title("Leptospirosis Risk Distribution", fontsize=14)
    plt.xlabel("Risk Score", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lepto_risk_distribution.png", dpi=300)
    plt.close()
    
    # Create a spatial plot of leptospirosis risk
    plt.figure(figsize=(12, 10))
    centroids.plot(column="lepto_risk", cmap="YlOrRd", legend=True, 
                  legend_kwds={'label': "Leptospirosis Risk Score"})
    plt.title("Spatial Distribution of Leptospirosis Risk", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lepto_risk_spatial.png", dpi=300)
    plt.close()
    
    # Plot relationship between flood risk and leptospirosis risk
    plt.figure(figsize=(10, 6))
    flood_risk_groups = centroids.groupby("FloodRisk")["lepto_risk"].mean().reset_index()
    plt.bar(flood_risk_groups["FloodRisk"], flood_risk_groups["lepto_risk"], color="teal")
    plt.title("Relationship Between Flood Risk and Leptospirosis Risk", fontsize=14)
    plt.xlabel("Flood Risk Level", fontsize=12)
    plt.ylabel("Average Leptospirosis Risk", fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/flood_risk_impact.png", dpi=300)
    plt.close()
    
    logger.info(f"Created statistical visualizations in {output_dir}")
    
    return True

def main():
    """
    Main function to prepare centroid data.
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Update these paths to match your file locations
    input_path = "raw_centroids.geojson"
    output_path = "centroids.geojson"
    
    # Prepare centroids
    centroids = prepare_centroids(input_path, output_path)
    
    # Plot statistics
    plot_statistics(centroids)
    
    print(f"Processed {len(centroids)} centroids and saved to {output_path}")
    print(f"Created statistical visualizations in 'stats' directory")

if __name__ == "__main__":
    main()