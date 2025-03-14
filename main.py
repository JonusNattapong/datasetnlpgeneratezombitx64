#!/usr/bin/env python3
"""
NLP Dataset Generator CLI

This script provides a command-line interface to generate synthetic datasets 
for various NLP tasks using the NLPDatasetGenerator class.
"""

import argparse
import os
from dataset_generator import NLPDatasetGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic datasets for NLP tasks")
    parser.add_argument(
        "--task", 
        type=str, 
        choices=[
            "text_classification", 
            "token_classification", 
            "question_answering", 
            "summarization", 
            "translation", 
            "sentence_similarity",
            "fill_mask", 
            "zero_shot_classification", 
            "text_generation", 
            "text2text_generation", 
            "table_qa", 
            "feature_extraction", 
            "all"
        ],
        default="all",
        help="NLP task to generate a dataset for (default: all)"
    )
    parser.add_argument(
        "--samples", 
        type=int, 
        default=500,
        help="Number of samples to generate per task (default: 500)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="generated_datasets",
        help="Output directory for generated datasets (default: generated_datasets)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Create the dataset generator
    generator = NLPDatasetGenerator(output_dir=args.output, verbose=args.verbose)
    
    # Generate the specified dataset(s)
    if args.task == "all":
        generator.generate_all_datasets(samples_per_task=args.samples)
    elif args.task == "text_classification":
        generator.generate_text_classification(num_samples=args.samples)
    elif args.task == "token_classification":
        generator.generate_token_classification(num_samples=args.samples)
    elif args.task == "question_answering":
        generator.generate_question_answering(num_samples=args.samples)
    elif args.task == "summarization":
        generator.generate_summarization(num_samples=args.samples)
    elif args.task == "translation":
        generator.generate_translation(num_samples=args.samples)
    elif args.task == "sentence_similarity":
        generator.generate_sentence_similarity(num_samples=args.samples)
    elif args.task == "fill_mask":
        generator.generate_fill_mask(num_samples=args.samples)
    elif args.task == "zero_shot_classification":
        generator.generate_zero_shot_classification(num_samples=args.samples)
    elif args.task == "text_generation":
        generator.generate_text_generation(num_samples=args.samples)
    elif args.task == "text2text_generation":
        generator.generate_text2text_generation(num_samples=args.samples)
    elif args.task == "table_qa":
        generator.generate_table_qa(num_samples=args.samples)
    elif args.task == "feature_extraction":
        generator.generate_feature_extraction(num_samples=args.samples)
    
    print(f"Dataset generation for '{args.task}' completed successfully!")
    print(f"Output saved to: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()