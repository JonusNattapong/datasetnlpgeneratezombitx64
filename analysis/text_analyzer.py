from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .base_analyzer import BaseAnalyzer
from data_sources.base_source import DataType

class TextAnalyzer(BaseAnalyzer):
    """Implementation of text data analysis."""
    
    def __init__(self):
        """Initialize the text analyzer."""
        super().__init__(DataType.TEXT)
        self._metric_descriptions = {
            "vocabulary_size": "Number of unique words in the dataset",
            "avg_sentence_length": "Average number of words per sentence",
            "type_token_ratio": "Ratio of unique words to total words",
            "hapax_legomena": "Number of words that appear exactly once",
            "redundancy_rate": "Percentage of near-duplicate content",
            "class_balance": "Distribution of classes (if applicable)",
        }
    
    def calculate_metrics(self, data: List[str]) -> Dict[str, Any]:
        """Calculate metrics for the text data.
        
        Args:
            data (List[str]): List of text samples to analyze.
            
        Returns:
            Dict[str, Any]: Dictionary of calculated metrics.
        """
        # Clear previous metrics
        self._metrics.clear()
        
        # Calculate basic statistics
        words = self._get_all_words(data)
        word_counts = Counter(words)
        total_words = len(words)
        unique_words = len(word_counts)
        
        # Calculate metrics
        self._metrics["vocabulary_size"] = unique_words
        self._metrics["total_words"] = total_words
        self._metrics["type_token_ratio"] = unique_words / total_words if total_words > 0 else 0
        self._metrics["hapax_legomena"] = sum(1 for count in word_counts.values() if count == 1)
        
        # Calculate average sentence length
        sentence_lengths = [len(text.split()) for text in data]
        self._metrics["avg_sentence_length"] = np.mean(sentence_lengths)
        self._metrics["std_sentence_length"] = np.std(sentence_lengths)
        
        # Calculate redundancy metrics
        redundancy_info = self._calculate_redundancy(data)
        self._metrics["redundancy_rate"] = redundancy_info["redundancy_rate"]
        self._metrics["near_duplicate_pairs"] = redundancy_info["duplicate_pairs"]
        
        return self._metrics
    
    def generate_visualizations(self, data: List[str]) -> Dict[str, plt.Figure]:
        """Generate visualizations for the text data.
        
        Args:
            data (List[str]): List of text samples to visualize.
            
        Returns:
            Dict[str, plt.Figure]: Dictionary of generated visualizations.
        """
        # Clear previous visualizations
        self.clear()
        
        # Word frequency distribution
        fig_freq = plt.figure(figsize=(10, 6))
        words = self._get_all_words(data)
        word_counts = Counter(words).most_common(20)  # Top 20 words
        words, counts = zip(*word_counts)
        plt.bar(words, counts)
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 20 Word Frequencies')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        self._visualizations["word_frequencies"] = fig_freq
        
        # Sentence length distribution
        fig_lengths = plt.figure(figsize=(10, 6))
        sentence_lengths = [len(text.split()) for text in data]
        sns.histplot(sentence_lengths, bins=30)
        plt.title('Sentence Length Distribution')
        plt.xlabel('Number of Words')
        plt.ylabel('Count')
        self._visualizations["sentence_lengths"] = fig_lengths
        
        # Word cloud (if wordcloud package is available)
        try:
            from wordcloud import WordCloud
            text = " ".join(data)
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            
            fig_cloud = plt.figure(figsize=(10, 6))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title('Word Cloud')
            self._visualizations["word_cloud"] = fig_cloud
        except ImportError:
            pass  # Skip word cloud if package not available
        
        return self._visualizations
    
    def get_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions of the metrics this analyzer calculates.
        
        Returns:
            Dict[str, str]: Dictionary mapping metric names to descriptions.
        """
        return self._metric_descriptions
    
    def _get_all_words(self, texts: List[str]) -> List[str]:
        """Get all words from the texts.
        
        Args:
            texts (List[str]): List of text samples.
            
        Returns:
            List[str]: List of all words.
        """
        words = []
        for text in texts:
            # Simple word tokenization (split on whitespace)
            # Could be improved with proper tokenization (e.g., NLTK)
            words.extend(text.lower().split())
        return words
    
    def _calculate_redundancy(self, texts: List[str]) -> Dict[str, Any]:
        """Calculate redundancy metrics using TF-IDF and cosine similarity.
        
        Args:
            texts (List[str]): List of text samples.
            
        Returns:
            Dict[str, Any]: Dictionary containing redundancy metrics.
        """
        if len(texts) < 2:
            return {
                "redundancy_rate": 0.0,
                "duplicate_pairs": []
            }
        
        # Convert texts to TF-IDF vectors
        vectorizer = TfidfVectorizer(strip_accents='unicode', stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Calculate pairwise similarities
        similarities = cosine_similarity(tfidf_matrix)
        
        # Find near-duplicate pairs (similarity > 0.9)
        duplicate_pairs = []
        n_duplicates = 0
        n = len(texts)
        
        for i in range(n):
            for j in range(i + 1, n):
                if similarities[i, j] > 0.9:  # Threshold for near-duplicates
                    n_duplicates += 1
                    duplicate_pairs.append((i, j, similarities[i, j]))
        
        # Calculate redundancy rate
        total_pairs = (n * (n - 1)) // 2  # Number of possible pairs
        redundancy_rate = n_duplicates / total_pairs if total_pairs > 0 else 0.0
        
        return {
            "redundancy_rate": redundancy_rate,
            "duplicate_pairs": duplicate_pairs
        }
    
    def analyze_class_balance(self, texts: List[str], labels: List[Any]) -> Dict[str, Any]:
        """Analyze class balance in labeled text data.
        
        Args:
            texts (List[str]): List of text samples.
            labels (List[Any]): List of corresponding labels.
            
        Returns:
            Dict[str, Any]: Class balance metrics and visualization.
        """
        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")
        
        # Calculate class distribution
        label_counts = Counter(labels)
        total_samples = len(labels)
        class_distribution = {
            label: count / total_samples
            for label, count in label_counts.items()
        }
        
        # Create class distribution plot
        fig = plt.figure(figsize=(10, 6))
        plt.bar(class_distribution.keys(), class_distribution.values())
        plt.title('Class Distribution')
        plt.xlabel('Classes')
        plt.ylabel('Proportion')
        plt.xticks(rotation=45, ha='right')
        
        # Add to visualizations
        self._visualizations["class_distribution"] = fig
        
        # Calculate class balance metrics
        n_classes = len(label_counts)
        expected_prob = 1 / n_classes
        max_deviation = max(abs(p - expected_prob) for p in class_distribution.values())
        
        balance_metrics = {
            "class_distribution": class_distribution,
            "number_of_classes": n_classes,
            "max_class_deviation": max_deviation,
            "is_balanced": max_deviation < 0.1  # Consider balanced if max deviation < 10%
        }
        
        # Update metrics
        self._metrics.update(balance_metrics)
        
        return balance_metrics