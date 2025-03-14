from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from data_sources.base_source import DataType

class BaseAugmenter(ABC):
    """Abstract base class for all data augmentation techniques."""
    
    def __init__(self, data_type: DataType):
        """Initialize the augmenter.
        
        Args:
            data_type (DataType): The type of data this augmenter handles.
        """
        self.data_type = data_type
        self._params = {}
    
    @abstractmethod
    def augment(self, data: Any, **kwargs) -> List[Any]:
        """Augment the provided data.
        
        Args:
            data (Any): The data to augment.
            **kwargs: Additional parameters for the augmentation technique.
            
        Returns:
            List[Any]: List of augmented data samples.
        """
        pass
    
    @abstractmethod
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate the augmentation parameters.
        
        Args:
            params (Dict[str, Any]): Parameters to validate.
            
        Returns:
            bool: True if parameters are valid, False otherwise.
        """
        pass
    
    @abstractmethod
    def get_param_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about the augmentation parameters.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary containing parameter information.
                Each parameter should have:
                - description: str
                - type: str
                - range: Optional[Tuple] for numerical params
                - options: Optional[List] for categorical params
        """
        pass
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set the augmentation parameters.
        
        Args:
            params (Dict[str, Any]): Parameters to set.
            
        Raises:
            ValueError: If parameters are invalid.
        """
        if not self.validate_params(params):
            raise ValueError("Invalid parameters")
        self._params = params.copy()
    
    def get_params(self) -> Dict[str, Any]:
        """Get the current augmentation parameters.
        
        Returns:
            Dict[str, Any]: Current parameters.
        """
        return self._params.copy()
    
    def get_data_type(self) -> DataType:
        """Get the type of data this augmenter handles.
        
        Returns:
            DataType: The type of data this augmenter handles.
        """
        return self.data_type
    
    def preview(self, data: Any, n_samples: int = 1, **kwargs) -> List[Any]:
        """Generate preview of augmented samples.
        
        Args:
            data (Any): The data to augment.
            n_samples (int): Number of preview samples to generate.
            **kwargs: Additional parameters for the augmentation.
            
        Returns:
            List[Any]: List of augmented samples for preview.
        """
        augmented = self.augment(data, **kwargs)
        return augmented[:n_samples] if len(augmented) > n_samples else augmented
    
    @abstractmethod
    def get_description(self) -> str:
        """Get a description of the augmentation technique.
        
        Returns:
            str: Description of what the augmentation does and how it works.
        """
        pass