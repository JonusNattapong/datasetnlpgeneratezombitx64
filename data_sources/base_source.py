from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from enum import Enum

class DataType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    TIME_SERIES = "time_series"

class BaseDataSource(ABC):
    """Abstract base class for all data sources."""
    
    def __init__(self, data_type: DataType):
        """Initialize the data source.
        
        Args:
            data_type (DataType): The type of data this source handles.
        """
        self.data_type = data_type
    
    @abstractmethod
    def load_data(self, source_path: str, **kwargs) -> Any:
        """Load data from the specified source.
        
        Args:
            source_path (str): Path to the data source.
            **kwargs: Additional arguments specific to the data type.
            
        Returns:
            Any: The loaded data in a format appropriate for the data type.
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: Any) -> bool:
        """Validate the loaded data.
        
        Args:
            data (Any): The data to validate.
            
        Returns:
            bool: True if the data is valid, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the data source.
        
        Returns:
            Dict[str, Any]: Dictionary containing metadata about the data source.
        """
        pass
    
    @abstractmethod
    def get_sample(self, data: Any, n_samples: int = 1) -> List[Any]:
        """Get sample(s) from the data.
        
        Args:
            data (Any): The data to sample from.
            n_samples (int): Number of samples to return.
            
        Returns:
            List[Any]: List of samples.
        """
        pass
    
    def get_data_type(self) -> DataType:
        """Get the type of data this source handles.
        
        Returns:
            DataType: The type of data this source handles.
        """
        return self.data_type