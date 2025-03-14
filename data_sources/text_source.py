import os
from typing import Any, Dict, List, Optional
import json
from .base_source import BaseDataSource, DataType

class TextFormat:
    PLAIN = "plain"
    CONLL = "conll"
    JSON = "json"

class TextDataSource(BaseDataSource):
    """Data source implementation for text data."""
    
    def __init__(self, text_format: str = TextFormat.PLAIN):
        """Initialize the text data source.
        
        Args:
            text_format (str): Format of the text data. One of TextFormat values.
        """
        super().__init__(DataType.TEXT)
        self.text_format = text_format
        self._metadata = {
            "format": text_format,
            "n_samples": 0,
            "vocab_size": 0,
            "avg_length": 0
        }
    
    def load_data(self, source_path: str, **kwargs) -> List[str]:
        """Load text data from the specified source.
        
        Args:
            source_path (str): Path to the text file.
            **kwargs: Additional arguments (e.g., encoding).
            
        Returns:
            List[str]: List of text samples.
        
        Raises:
            FileNotFoundError: If the source file doesn't exist.
            ValueError: If the file format is invalid.
        """
        if not os.path.exists(source_path):
            raise FileNotFoundError(f"File not found: {source_path}")
            
        encoding = kwargs.get("encoding", "utf-8")
        
        if self.text_format == TextFormat.PLAIN:
            with open(source_path, 'r', encoding=encoding) as f:
                data = [line.strip() for line in f if line.strip()]
                
        elif self.text_format == TextFormat.CONLL:
            data = []
            current_sentence = []
            with open(source_path, 'r', encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        current_sentence.append(line)
                    elif current_sentence:
                        data.append('\n'.join(current_sentence))
                        current_sentence = []
                if current_sentence:
                    data.append('\n'.join(current_sentence))
                    
        elif self.text_format == TextFormat.JSON:
            with open(source_path, 'r', encoding=encoding) as f:
                json_data = json.load(f)
                if isinstance(json_data, list):
                    data = [str(item) for item in json_data]
                elif isinstance(json_data, dict):
                    data = [str(value) for value in json_data.values()]
                else:
                    data = [str(json_data)]
        else:
            raise ValueError(f"Unsupported text format: {self.text_format}")
            
        self._update_metadata(data)
        return data
    
    def validate_data(self, data: List[str]) -> bool:
        """Validate the loaded text data.
        
        Args:
            data (List[str]): The data to validate.
            
        Returns:
            bool: True if the data is valid, False otherwise.
        """
        if not isinstance(data, list):
            return False
            
        if not all(isinstance(item, str) for item in data):
            return False
            
        if self.text_format == TextFormat.CONLL:
            # Check if all samples follow CoNLL format (word<space>tag)
            for sample in data:
                lines = sample.split('\n')
                for line in lines:
                    if line and len(line.split()) < 2:
                        return False
                        
        return True
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the text data.
        
        Returns:
            Dict[str, Any]: Dictionary containing metadata.
        """
        return self._metadata
    
    def get_sample(self, data: List[str], n_samples: int = 1) -> List[str]:
        """Get random samples from the text data.
        
        Args:
            data (List[str]): The data to sample from.
            n_samples (int): Number of samples to return.
            
        Returns:
            List[str]: List of sampled texts.
        """
        import random
        n_samples = min(n_samples, len(data))
        return random.sample(data, n_samples)
    
    def _update_metadata(self, data: List[str]) -> None:
        """Update metadata based on the loaded data.
        
        Args:
            data (List[str]): The loaded data.
        """
        # Count unique words across all samples
        words = set()
        total_length = 0
        
        for sample in data:
            if self.text_format == TextFormat.CONLL:
                # For CoNLL format, only count the words (first column)
                for line in sample.split('\n'):
                    if line:
                        word = line.split()[0]
                        words.add(word)
                        total_length += 1
            else:
                # For other formats, split on whitespace
                sample_words = sample.split()
                words.update(sample_words)
                total_length += len(sample_words)
        
        self._metadata.update({
            "n_samples": len(data),
            "vocab_size": len(words),
            "avg_length": total_length / len(data) if data else 0
        })