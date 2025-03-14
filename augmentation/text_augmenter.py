import random
from typing import Any, Dict, List, Optional, Tuple
import nltk
from nltk.corpus import wordnet
from .base_augmenter import BaseAugmenter
from data_sources.base_source import DataType

class TextAugmenter(BaseAugmenter):
    """Implementation of text augmentation techniques."""
    
    def __init__(self):
        """Initialize the text augmenter."""
        super().__init__(DataType.TEXT)
        
        # Download required NLTK data
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            nltk.download('punkt')
            nltk.download('averaged_perceptron_tagger')
        
        # Default parameters
        self._params = {
            "synonym_replace_prob": 0.3,
            "random_insert_prob": 0.1,
            "random_delete_prob": 0.1,
            "random_swap_prob": 0.1,
            "max_augmentations": 5
        }
    
    def augment(self, data: str, **kwargs) -> List[str]:
        """Augment the provided text data.
        
        Args:
            data (str): The text to augment.
            **kwargs: Additional parameters to override defaults.
            
        Returns:
            List[str]: List of augmented text samples.
        """
        # Update parameters with any provided kwargs
        params = self._params.copy()
        params.update(kwargs)
        
        augmented_texts = []
        max_aug = params["max_augmentations"]
        
        # Apply each augmentation technique
        texts = []
        
        # Synonym replacement
        if params["synonym_replace_prob"] > 0:
            texts.extend(self._synonym_replacement(
                data,
                params["synonym_replace_prob"],
                max_aug // 4 or 1
            ))
        
        # Random insertion
        if params["random_insert_prob"] > 0:
            texts.extend(self._random_insertion(
                data,
                params["random_insert_prob"],
                max_aug // 4 or 1
            ))
        
        # Random deletion
        if params["random_delete_prob"] > 0:
            texts.extend(self._random_deletion(
                data,
                params["random_delete_prob"],
                max_aug // 4 or 1
            ))
        
        # Random swap
        if params["random_swap_prob"] > 0:
            texts.extend(self._random_swap(
                data,
                params["random_swap_prob"],
                max_aug // 4 or 1
            ))
        
        # Ensure we don't return duplicates
        return list(set(texts))[:max_aug]
    
    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate the augmentation parameters.
        
        Args:
            params (Dict[str, Any]): Parameters to validate.
            
        Returns:
            bool: True if parameters are valid, False otherwise.
        """
        required_params = {
            "synonym_replace_prob",
            "random_insert_prob",
            "random_delete_prob",
            "random_swap_prob",
            "max_augmentations"
        }
        
        # Check if all required parameters are present
        if not all(param in params for param in required_params):
            return False
        
        # Validate probability values
        probs = [
            params["synonym_replace_prob"],
            params["random_insert_prob"],
            params["random_delete_prob"],
            params["random_swap_prob"]
        ]
        
        if not all(isinstance(p, (int, float)) and 0 <= p <= 1 for p in probs):
            return False
        
        # Validate max_augmentations
        if not isinstance(params["max_augmentations"], int) or params["max_augmentations"] <= 0:
            return False
        
        return True
    
    def get_param_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about the augmentation parameters.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary containing parameter information.
        """
        return {
            "synonym_replace_prob": {
                "description": "Probability of replacing a word with its synonym",
                "type": "float",
                "range": (0.0, 1.0)
            },
            "random_insert_prob": {
                "description": "Probability of inserting a random word",
                "type": "float",
                "range": (0.0, 1.0)
            },
            "random_delete_prob": {
                "description": "Probability of deleting a word",
                "type": "float",
                "range": (0.0, 1.0)
            },
            "random_swap_prob": {
                "description": "Probability of swapping two words",
                "type": "float",
                "range": (0.0, 1.0)
            },
            "max_augmentations": {
                "description": "Maximum number of augmented samples to generate",
                "type": "int",
                "range": (1, 100)
            }
        }
    
    def get_description(self) -> str:
        """Get a description of the text augmentation techniques.
        
        Returns:
            str: Description of the augmentation techniques.
        """
        return (
            "Applies various text augmentation techniques including:\n"
            "1. Synonym Replacement: Replaces words with their synonyms\n"
            "2. Random Insertion: Inserts random words into the text\n"
            "3. Random Deletion: Randomly deletes words from the text\n"
            "4. Random Swap: Randomly swaps the position of two words"
        )
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word using WordNet.
        
        Args:
            word (str): Word to find synonyms for.
            
        Returns:
            List[str]: List of synonyms.
        """
        synonyms = set()
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms.add(lemma.name())
        
        return list(synonyms)
    
    def _synonym_replacement(self, text: str, prob: float, n_samples: int) -> List[str]:
        """Replace words with their synonyms.
        
        Args:
            text (str): Input text.
            prob (float): Probability of replacing each word.
            n_samples (int): Number of samples to generate.
            
        Returns:
            List[str]: Augmented texts.
        """
        words = text.split()
        results = []
        
        for _ in range(n_samples):
            new_words = words.copy()
            
            for i, word in enumerate(words):
                if random.random() < prob:
                    synonyms = self._get_synonyms(word)
                    if synonyms:
                        new_words[i] = random.choice(synonyms)
            
            results.append(" ".join(new_words))
        
        return results
    
    def _random_insertion(self, text: str, prob: float, n_samples: int) -> List[str]:
        """Randomly insert words into the text.
        
        Args:
            text (str): Input text.
            prob (float): Probability of insertion at each position.
            n_samples (int): Number of samples to generate.
            
        Returns:
            List[str]: Augmented texts.
        """
        words = text.split()
        results = []
        
        for _ in range(n_samples):
            new_words = words.copy()
            
            for i in range(len(words)):
                if random.random() < prob:
                    random_word = random.choice(words)
                    synonyms = self._get_synonyms(random_word)
                    if synonyms:
                        new_words.insert(i, random.choice(synonyms))
            
            results.append(" ".join(new_words))
        
        return results
    
    def _random_deletion(self, text: str, prob: float, n_samples: int) -> List[str]:
        """Randomly delete words from the text.
        
        Args:
            text (str): Input text.
            prob (float): Probability of deleting each word.
            n_samples (int): Number of samples to generate.
            
        Returns:
            List[str]: Augmented texts.
        """
        words = text.split()
        results = []
        
        for _ in range(n_samples):
            new_words = []
            
            for word in words:
                if random.random() >= prob:  # Keep the word
                    new_words.append(word)
            
            if not new_words:  # Ensure we don't return empty text
                new_words = [random.choice(words)]
            
            results.append(" ".join(new_words))
        
        return results
    
    def _random_swap(self, text: str, prob: float, n_samples: int) -> List[str]:
        """Randomly swap pairs of words in the text.
        
        Args:
            text (str): Input text.
            prob (float): Probability of swapping at each position.
            n_samples (int): Number of samples to generate.
            
        Returns:
            List[str]: Augmented texts.
        """
        words = text.split()
        results = []
        
        for _ in range(n_samples):
            new_words = words.copy()
            
            for i in range(len(words) - 1):
                if random.random() < prob:
                    j = random.randint(0, len(words) - 1)
                    new_words[i], new_words[j] = new_words[j], new_words[i]
            
            results.append(" ".join(new_words))
        
        return results