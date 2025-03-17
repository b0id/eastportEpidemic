import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import glob


class DiseaseSpreadAnimation:
    """
    A class to create animations of disease spread from GeoJSON files.
    
    This class loads a series of GeoJSON files representing disease states at different time points,
    and creates an animation showing the spread of the disease over time.
    """
    
    def __init__(self, data_dir='.', file_pattern='grid_day_*.geojson', dpi=100):
        """
        Initialize the animation class.
        
        Args:
            data_dir: Directory containing the GeoJSON files
            file_pattern: Pattern to match the GeoJSON files
            dpi: DPI for the output animation
        """
        self.data_dir = data_dir
        self.file_pattern = file_pattern
        self.dpi = dpi
        self.files = []
        self.data_by_day = {}
        self.days = []
        self.current_view = 'current_infectious'  # Default view
        
        # Define colormaps for different disease states
        self.colormaps = {
            'current_susceptible': plt.cm.Blues,
            'current_exposed': plt.cm.Oranges,
            'current_infectious': plt.cm.Reds,
            'current_recovered': plt.cm.Greens,
            'deaths': plt.cm.Purples,
            'percent_infected': plt.cm.YlOrRd
        }
        
    def load_data(self):
        """Load all the GeoJSON files matching the pattern."""
        # Get a list of all files matching the pattern
        file_path = os.path.join(self.data_dir, self.file_pattern)
        self.files = sorted(glob.glob(file_path))
        
        if not self.files:
            raise FileNotFoundError(f"No files found matching {file_path}")
        
        print(f"Found {len(self.files)} files to animate")
        
        # Extract day numbers and sort
        for file in self.files:
            try:
                # Extract the day number from the filename
                day = int(os.path.basename(file).split('_')[2].split('.')[0])
                
                # Load the data
                with open(file, 'r') as f:
                    data = json.load(f)
                
                self.data_by_day[day] = data
                self.days.append(day)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        # Sort days
        self.days.sort()
        
        if not self.days:
            raise ValueError("No valid data files could be loaded")
        
        print(f"Loaded data for days: {self.days}")
        return self
    
    def set_view(self, view_type):
        """
        Set the view type for the animation.
        
        Args:
            view_type: One of 'current_susceptible', 'current_exposed', 
                       'current_infectious', 'current_recovered', 'deaths', 'percent_infected'
        """
        if view_type not in self.colormaps:
            raise ValueError(f"Invalid view type. Choose from: {list(self.colormaps.keys())}")
        
        self.current_view = view_type
        return self
    
    def _get_max_value(self):
        """Get the maximum value for the current view for proper scaling."""
        max_val = 0
        for day in self.days:
            data = self.data_by_day[day]
            for feature in data['features']:
                val = feature['properties'].get(self.current_view, 0)
                max_val = max(max_val, val)
        return max_val
    
    def create_animation(self, output_file='disease_animation.gif', fps=2, figsize=(10, 8), title=None):
        """
        Create and save the animation.
        
        Args:
            output_file: Name of the output file (GIF or MP4)
            fps: Frames per second in the animation
            figsize: Size of the figure (width, height) in inches
            title: Title for the animation. If None, will use the current view type.
            
        Returns:
            Path to the saved animation file
        """
        if not self.data_by_day:
            self.load_data()
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set title
        if title is None:
            title = f"Disease Spread: {self.current_view.replace('_', ' ').title()}"
        
        # Find the geographic bounds by examining the first day's data
        first_day_data = self.data_by_day[self.days[0]]
        min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
        
        for feature in first_day_data['features']:
            if feature['geometry']['type'] == 'MultiPolygon':
                coords = feature['geometry']['coordinates'][0][0]
                for x, y in coords:
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
        
        # Add some padding to the bounds
        padding = 0.01
        x_range = max_x - min_x
        y_range = max_y - min_y
        min_x -= x_range * padding
        max_x += x_range * padding
        min_y -= y_range * padding
        max_y += y_range * padding
        
        # Set the axis limits
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        
        # Get the maximum value for color scaling
        max_val = self._get_max_value()
        norm = mcolors.Normalize(vmin=0, vmax=max_val)
        
        # Create a colormap
        cmap = self.colormaps[self.current_view]
        
        # Create a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label(self.current_view.replace('_', ' ').title())
        
        # Function to draw a single frame
        def update(day_idx):
            ax.clear()
            day = self.days[day_idx]
            data = self.data_by_day[day]
            
            # Get properties for visualization
            polygons = []
            values = []
            
            for feature in data['features']:
                if feature['geometry']['type'] == 'MultiPolygon':
                    # Get the value for this feature
                    val = feature['properties'].get(self.current_view, 0)
                    
                    # Create a polygon for this feature
                    for polygon_coords in feature['geometry']['coordinates']:
                        poly = Polygon(polygon_coords[0], closed=True)
                        polygons.append(poly)
                        values.append(val)
            
            # Create a patch collection
            collection = PatchCollection(
                polygons, 
                cmap=cmap,
                edgecolor='black',
                linewidth=0.5,
                alpha=0.8
            )
            
            # Set the colors of the polygons based on their values
            collection.set_array(np.array(values))
            collection.set_clim(0, max_val)
            
            # Add the collection to the plot
            ax.add_collection(collection)
            
            # Set axis limits to ensure consistent display
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_aspect('equal')
            
            # Set title with current day
            ax.set_title(f"{title} - Day {day}")
            
            # Remove axes for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Return the added collection
            return collection,
        
        # Create the animation
        anim = animation.FuncAnimation(
            fig, 
            update, 
            frames=len(self.days),
            interval=1000/fps,
            blit=True
        )
        
        # Save the animation
        if output_file.endswith('.gif'):
            # Save as GIF
            anim.save(output_file, writer='pillow', fps=fps, dpi=self.dpi)
        elif output_file.endswith('.mp4'):
            # Save as MP4
            anim.save(output_file, writer='ffmpeg', fps=fps, dpi=self.dpi)
        else:
            # Default to GIF
            if '.' not in output_file:
                output_file += '.gif'
            anim.save(output_file, writer='pillow', fps=fps, dpi=self.dpi)
        
        plt.close(fig)
        print(f"Animation saved to {output_file}")
        return output_file

    def create_comparison_animation(self, properties=['current_infectious', 'current_exposed', 'current_recovered'], 
                                   output_file='disease_comparison.gif', fps=2, figsize=(16, 6)):
        """
        Create a side-by-side comparison animation of multiple properties.
        
        Args:
            properties: List of properties to compare
            output_file: Name of the output file
            fps: Frames per second
            figsize: Size of the figure
            
        Returns:
            Path to the saved animation file
        """
        if not self.data_by_day:
            self.load_data()
        
        # Make sure all properties are valid
        for prop in properties:
            if prop not in self.colormaps:
                raise ValueError(f"Invalid property: {prop}. Choose from: {list(self.colormaps.keys())}")
        
        # Create figure with subplots
        n_props = len(properties)
        fig, axes = plt.subplots(1, n_props, figsize=figsize)
        
        # Find the geographic bounds
        first_day_data = self.data_by_day[self.days[0]]
        min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
        
        for feature in first_day_data['features']:
            if feature['geometry']['type'] == 'MultiPolygon':
                coords = feature['geometry']['coordinates'][0][0]
                for x, y in coords:
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
        
        # Add padding
        padding = 0.01
        x_range = max_x - min_x
        y_range = max_y - min_y
        min_x -= x_range * padding
        max_x += x_range * padding
        min_y -= y_range * padding
        max_y += y_range * padding
        
        # Get the maximum values for each property
        max_vals = {}
        for prop in properties:
            max_vals[prop] = 0
            for day in self.days:
                data = self.data_by_day[day]
                for feature in data['features']:
                    val = feature['properties'].get(prop, 0)
                    max_vals[prop] = max(max_vals[prop], val)
        
        # Create colorbars for each property
        scalar_mappables = []
        for i, prop in enumerate(properties):
            norm = mcolors.Normalize(vmin=0, vmax=max_vals[prop])
            cmap = self.colormaps[prop]
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=axes[i])
            cbar.set_label(prop.replace('_', ' ').title())
            scalar_mappables.append((norm, cmap))
        
        # Function to draw a single frame
        def update(day_idx):
            day = self.days[day_idx]
            data = self.data_by_day[day]
            collections = []
            
            for i, prop in enumerate(properties):
                ax = axes[i]
                ax.clear()
                
                # Get polygons and values for this property
                polygons = []
                values = []
                
                for feature in data['features']:
                    if feature['geometry']['type'] == 'MultiPolygon':
                        val = feature['properties'].get(prop, 0)
                        for polygon_coords in feature['geometry']['coordinates']:
                            poly = Polygon(polygon_coords[0], closed=True)
                            polygons.append(poly)
                            values.append(val)
                
                # Create patch collection
                norm, cmap = scalar_mappables[i]
                collection = PatchCollection(
                    polygons,
                    cmap=cmap,
                    edgecolor='black',
                    linewidth=0.5,
                    alpha=0.8
                )
                
                # Set colors
                collection.set_array(np.array(values))
                collection.set_clim(0, max_vals[prop])
                
                # Add collection to the plot
                ax.add_collection(collection)
                collections.append(collection)
                
                # Set axis properties
                ax.set_xlim(min_x, max_x)
                ax.set_ylim(min_y, max_y)
                ax.set_aspect('equal')
                ax.set_title(f"{prop.replace('_', ' ').title()} - Day {day}")
                ax.set_xticks([])
                ax.set_yticks([])
            
            fig.suptitle(f"Disease Spread - Day {day}", fontsize=16)
            plt.tight_layout()
            
            return collections
        
        # Create the animation
        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(self.days),
            interval=1000/fps,
            blit=True
        )
        
        # Save the animation
        if output_file.endswith('.gif'):
            anim.save(output_file, writer='pillow', fps=fps, dpi=self.dpi)
        elif output_file.endswith('.mp4'):
            anim.save(output_file, writer='ffmpeg', fps=fps, dpi=self.dpi)
        else:
            if '.' not in output_file:
                output_file += '.gif'
            anim.save(output_file, writer='pillow', fps=fps, dpi=self.dpi)
        
        plt.close(fig)
        print(f"Comparison animation saved to {output_file}")
        return output_file


# Example usage
if __name__ == "__main__":
    # Simple single property animation
    animator = DiseaseSpreadAnimation()
    animator.load_data()
    animator.set_view('current_infectious')
    animator.create_animation(output_file='infectious_spread.gif', fps=1)
    
    # Multi-property comparison animation
    animator.create_comparison_animation(
        properties=['current_infectious', 'current_exposed', 'current_recovered'],
        output_file='disease_comparison.gif',
        fps=1
    )