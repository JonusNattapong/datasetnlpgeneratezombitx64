from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from data_sources.base_source import DataType
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class BaseAnalyzer(ABC):
    """Abstract base class for data analyzers."""
    
    def __init__(self, data_type: DataType):
        """Initialize the analyzer.
        
        Args:
            data_type (DataType): The type of data this analyzer handles.
        """
        self.data_type = data_type
        self._metrics = {}
        self._visualizations = {}
    
    @abstractmethod
    def calculate_metrics(self, data: Any) -> Dict[str, Any]:
        """Calculate metrics for the provided data.
        
        Args:
            data (Any): The data to analyze.
            
        Returns:
            Dict[str, Any]: Dictionary of calculated metrics.
        """
        pass
    
    @abstractmethod
    def generate_visualizations(self, data: Any) -> Dict[str, plt.Figure]:
        """Generate visualizations for the provided data.
        
        Args:
            data (Any): The data to visualize.
            
        Returns:
            Dict[str, plt.Figure]: Dictionary of generated visualizations.
        """
        pass
    
    @abstractmethod
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions of the metrics this analyzer calculates.
        
        Returns:
            Dict[str, str]: Dictionary mapping metric names to descriptions.
        """
        pass
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the currently calculated metrics.
        
        Returns:
            Dict[str, Any]: Dictionary of metrics.
        """
        return self._metrics.copy()
    
    def get_visualizations(self) -> Dict[str, plt.Figure]:
        """Get the currently generated visualizations.
        
        Returns:
            Dict[str, plt.Figure]: Dictionary of visualizations.
        """
        return self._visualizations.copy()
    
    def get_data_type(self) -> DataType:
        """Get the type of data this analyzer handles.
        
        Returns:
            DataType: The type of data this analyzer handles.
        """
        return self.data_type
    
    def clear(self) -> None:
        """Clear all calculated metrics and generated visualizations."""
        self._metrics.clear()
        for fig in self._visualizations.values():
            plt.close(fig)
        self._visualizations.clear()
    
    def to_report(self, format: str = "markdown") -> str:
        """Generate a formatted report of the analysis.
        
        Args:
            format (str): Output format ("markdown" or "html").
            
        Returns:
            str: Formatted report.
        """
        if format not in ["markdown", "html"]:
            raise ValueError("Unsupported format. Use 'markdown' or 'html'.")
        
        if format == "markdown":
            report = "# Data Analysis Report\n\n"
            
            # Add metrics section
            report += "## Metrics\n\n"
            descriptions = self.get_metric_descriptions()
            for name, value in self._metrics.items():
                desc = descriptions.get(name, "No description available.")
                report += f"### {name}\n"
                report += f"{desc}\n\n"
                report += f"Value: {value}\n\n"
            
            # Add visualizations section
            report += "## Visualizations\n\n"
            report += "(Visualizations are available through the web interface)\n\n"
            
        else:  # HTML format
            report = "<h1>Data Analysis Report</h1>"
            
            # Add metrics section
            report += "<h2>Metrics</h2>"
            descriptions = self.get_metric_descriptions()
            for name, value in self._metrics.items():
                desc = descriptions.get(name, "No description available.")
                report += f"<h3>{name}</h3>"
                report += f"<p>{desc}</p>"
                report += f"<p>Value: {value}</p>"
            
            # Add visualizations section
            report += "<h2>Visualizations</h2>"
            report += "<p>(Visualizations are available through the web interface)</p>"
        
        return report
    
    def save_visualizations(self, output_dir: str) -> List[str]:
        """Save all visualizations to files.
        
        Args:
            output_dir (str): Directory to save the visualizations in.
            
        Returns:
            List[str]: List of saved file paths.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        for name, fig in self._visualizations.items():
            filename = os.path.join(output_dir, f"{name}.png")
            fig.savefig(filename, bbox_inches='tight', dpi=300)
            saved_files.append(filename)
        
        return saved_files