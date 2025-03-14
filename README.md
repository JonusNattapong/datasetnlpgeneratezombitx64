# datasetnlpgeneratezombitx64
generate dataset for nlp
# NLP Dataset Generator

This repository contains tools for generating synthetic datasets for various NLP tasks. The generator creates datasets in the Hugging Face Datasets format, making them ready for use with the 🤗 Transformers library.

## Features

Generate synthetic datasets for the following NLP tasks:

- Text Classification
- Token Classification (NER)
- Question Answering
- Summarization
- Translation
- Sentence Similarity
- Fill-Mask
- Zero-Shot Classification
- Text Generation
- Text2Text Generation
- Table Question Answering
- Feature Extraction

## Installation

1. Clone this repository:
```bash
git clone https://github.com/JonusNattapong/datasetnlpgeneratezombitx64.git
cd datasetnlpgeneratezombitx64
```

2. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

To generate datasets, run the `main.py` script with the desired task and options.

```bash
# Generate all datasets (default)
python main.py

# Generate a specific dataset type
python main.py --task text_classification

# Specify the number of samples
python main.py --task question_answering --samples 1000

# Specify output directory
python main.py --output my_datasets

```

## Using as a Library
You can also use the dataset generator in your own code:
```python
from dataset_generator import NLPDatasetGenerator

# Initialize the generator
generator = NLPDatasetGenerator(output_dir="my_datasets")

# Generate specific datasets
classification_dataset = generator.generate_text_classification(num_samples=500)
qa_dataset = generator.generate_question_answering(num_samples=300)

# Generate all datasets
generator.generate_all_datasets(samples_per_task=500)
```

## Example Datasets

Text Classification
```python
{
  "text": "This is a positive review about the movie.",
  "label": 0
}
```
Question Answering
```python
{
  "context": "The capital of France is Paris. It is known for landmarks such as the Eiffel Tower.",
  "question": "What is the capital of France?",
  "answers": {"text": "Paris", "answer_start": 23}
}
```

## Customization
You can customize the dataset generation by modifying the templates in dataset_generator.py. The code is designed to be easily extendable for additional examples or task types.