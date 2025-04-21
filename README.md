# Leptospirosis Outbreak Case Study

## Overview
This project simulates a leptospirosis outbreak in Eastport, Maine following a hurricane event. It implements a spatially-explicit SEIR (Susceptible-Exposed-Infectious-Recovered) model to simulate disease spread through a population and evaluates the impact of different intervention timelines.
<p float="left">

  <img src="Screenshot 2025-03-16 061750.png" width="200"/>
  <img src="Screenshot 2025-03-16 021630.png" width="200"/>
   <img src="Screenshot 2025-03-16 013641.png" width="200"/>
  <img src="Screenshot 2025-03-16 011436.png" width="200"/>
  <img src="Screenshot 2025-03-14 235034.png" width="200"/>
  
## Features
- Geospatial modeling of leptospirosis spread based on property locations
- Integration with flood risk data to model environmental exposure
- Demographic factors affecting disease vulnerability:
  - Age (young, adult, senior, elderly)
  - Disability status
  - Household size
  - Population density
- Housing factors:
  - Vacancy rates
  - Property locations
- Comparison of intervention scenarios at different time points (Day 10, 20, 30, 40)
- Spatial and temporal visualization of outbreak progression
- Statistical analysis of outbreak patterns and intervention effectiveness
- Geographic transmission patterns influenced by proximity and flood conditions

## Requirements
- Python 3.8+
- GeoPandas
- Pandas
- NumPy
- Matplotlib
- SciPy
- Joblib

## Installation
1. Clone this repository
   ```
   git clone https://github.com/b0id/leptospirosis-model.git
   cd leptospirosis-model
   ```
2. Create a virtual environment:
   ```
   conda create -n epi_model python=3.9
   conda activate epi_model
   ```
3. Install the required packages:
   ```
   pip install geopandas pandas numpy matplotlib scipy joblib
   ```

## Input Data
The model requires the following input files in the `input` directory:
- `raw_centroids.geojson`: Property centroids with the following attributes:
  - `density_no`: Normalized population density
  - `FloodRisk`: Flood risk score (0-5)
- `grid.geojson`: Hexagonal census grid for aggregating results

## Usage
Run the main simulation:
```
python main.py
```

This will:
1. Prepare the statistical data for centroids
2. Run the SEIR model simulations for different response times
3. Create visualizations and statistical reports
4. Generate distribution maps showing the spatial patterns of infection

## Output
The simulation generates the following outputs in the `outputs` directory:
- Statistical visualizations in `statistics/`
  - Household size distribution
  - Age category distribution
  - Vacancy rates by density
  - Leptospirosis risk distribution
  - Spatial risk distribution
  - Risk by flood zone
- Simulation results in `simulations/`
  - SEIR curves for each response scenario
  - Cumulative impact charts
- Comparison charts in `comparisons/`
  - Cases comparison across scenarios
  - Deaths comparison across scenarios
  - Deaths prevented analysis
- Risk factor analysis in `risk_factors/`
  - Flood risk impact
  - Age impact
  - Disability impact
  - Combined risk factor analysis
- Report data in `report/`
  - Summary statistics
  - Outbreak statistics by response day
- Leptospirosis distribution maps in `distributions/response_X/`
  - Spatial distribution maps for days 0, 10, 20, 30, 40, 50, 60
- Animation data in `animations/response_X/`
  - GeoJSON files for animation frames

## Model Description
The model implements a spatially-explicit SEIR framework with the following components:

### Disease Parameters
- Incubation period: 10 days (σ = 1/10)
- Infectious period: 14 days (γ = 1/14)
- Base transmission rate: 0.4
- Flood-enhanced transmission rate: 1.0
- Case fatality rates varying by demographic:
  - Young: 2%
  - Adult: 4%
  - Senior: 7%
  - Elderly: 12%
  - Increased by 25% for individuals with disabilities

### Environmental Factors
- Flood risk amplifies transmission probability
- Geographic location affects transmission patterns
- Neighbor proximity within 150 meters affects transmission
- Population density modifies transmission rate

### Intervention Effects
- Day 10 response: 85% reduction in transmission
- Day 20 response: 65% reduction in transmission
- Day 30 response: 45% reduction in transmission
- Day 40 response: 25% reduction in transmission
- Gradual implementation of interventions over 7 days
- Location-based variations in intervention effectiveness
- Early responses also reduce mortality through better treatment

## Files
- `main.py`: Main execution script
- `data_preparation.py`: Prepares statistical data for centroids
- `lepto_model.py`: Implements the SEIR model
- `distribution_visualization.py`: Creates visualization maps
- `results_visualization.py`: Generates comparative visualizations and reports

## Extending the Model
The model can be extended by:
1. Modifying disease parameters in `lepto_model.py`
2. Adding new risk factors in `data_preparation.py`
3. Creating new visualization types in `distribution_visualization.py`
4. Implementing additional intervention scenarios
5. Adding socioeconomic factors to influence intervention access
6. Incorporating more detailed environmental data

## Future Work
- Interactive web visualization using Flask
- Integration with real-time weather data
- More sophisticated intervention modeling including:
  - Resource constraints
  - Behavioral factors affecting compliance
  - Vaccination campaigns
- Enhanced visualization showing SEIR transitions with death markers
- Socioeconomic factors affecting intervention access
- Multi-pathogen modeling for post-disaster scenarios

## License
MIT License

Copyright (c) 2025 b0id

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Acknowledgements
This project was developed as a case study for demonstrating epidemiological modeling techniques in a spatial context following hurricane events. The model structure and parameters are based on established leptospirosis literature and post-disaster outbreak patterns.
