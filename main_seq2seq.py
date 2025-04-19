#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Entry Point for Python Interview System with BERT-based Feedback Generation

This script orchestrates the entire Python interview system workflow with enhanced feedback:
1. Generates training data (if needed)
2. Trains the BERT classification model
3. Trains the BERT sequence-to-sequence model for detailed feedback generation
4. Runs the terminal interview system with enhanced feedback

Ensures that all models are properly trained before conducting interviews.
"""

import os
import sys
import json
import argparse
from colorama import init, Fore, Style

# Import system components
import data_generator
import model_trainer
import model_trainer_seq2seq
from interview_system import InterviewSystem
from utils import ensure_directories_exist

# Initialize colorama for colored terminal output
init(autoreset=True)

def check_data_exists():
    """Check if training data exists."""
    return os.path.exists('data/training_data.json') and os.path.getsize('data/training_data.json') > 0

def check_feedback_data_exists():
    """Check if feedback dataset exists."""
    return os.path.exists('data/feedback_dataset.json') and os.path.getsize('data/feedback_dataset.json') > 0

def check_model_exists():
    """Check if trained model exists."""
    return os.path.exists('models/bert_interview_final.pt')

def check_feedback_model_exists():
    """Check if trained feedback model exists."""
    return os.path.exists('models/bert_feedback_final.pt')

def generate_data():
    """Generate training data if it doesn't exist."""
    print(f"{Fore.YELLOW}Checking training data...")
    
    if check_data_exists():
        print(f"{Fore.GREEN}Training data already exists.")
        return True
    
    print(f"{Fore.YELLOW}Training data not found. Generating training data...")
    try:
        # Call the data generation function from data_generator.py
        if hasattr(data_generator, 'main') and callable(data_generator.main):
            data_generator.main()
        else:
            print(f"{Fore.RED}Error: data_generator.py does not have a main() function.")
            return False
        
        if check_data_exists():
            print(f"{Fore.GREEN}Training data generated successfully.")
            return True
        else:
            print(f"{Fore.RED}Error: Failed to generate training data.")
            return False
    except Exception as e:
        print(f"{Fore.RED}Error generating training data: {str(e)}")
        return False

def train_model():
    """Train the BERT model if it doesn't exist."""
    print(f"{Fore.YELLOW}Checking for trained model...")
    
    if check_model_exists():
        print(f"{Fore.GREEN}Trained model already exists.")
        return True
    
    print(f"{Fore.YELLOW}Trained model not found. Starting model training...")
    print(f"{Fore.YELLOW}This may take some time depending on your hardware.")
    
    try:
        # Call the model training function from model_trainer.py
        model_trainer.main()
        
        if check_model_exists():
            print(f"{Fore.GREEN}Model trained successfully.")
            return True
        else:
            print(f"{Fore.RED}Error: Failed to train model.")
            return False
    except Exception as e:
        print(f"{Fore.RED}Error training model: {str(e)}")
        return False

def train_feedback_model():
    """Train the BERT sequence-to-sequence model for feedback generation."""
    print(f"{Fore.YELLOW}Checking for trained feedback model...")
    
    if check_feedback_model_exists():
        print(f"{Fore.GREEN}Trained feedback model already exists.")
        return True
    
    print(f"{Fore.YELLOW}Checking feedback dataset...")
    if not check_feedback_data_exists():
        print(f"{Fore.RED}Feedback dataset not found. Cannot train feedback model.")
        return False
    
    print(f"{Fore.YELLOW}Starting feedback model training...")
    print(f"{Fore.YELLOW}This may take some time depending on your hardware.")
    
    try:
        # Call the model training function from model_trainer_seq2seq.py
        model_trainer_seq2seq.main()
        
        if check_feedback_model_exists():
            print(f"{Fore.GREEN}Feedback model trained successfully.")
            return True
        else:
            print(f"{Fore.RED}Error: Failed to train feedback model.")
            return False
    except Exception as e:
        print(f"{Fore.RED}Error training feedback model: {str(e)}")
        return False

def run_interview_system(use_seq2seq=False):
    """Run the terminal interview system."""
    print(f"{Fore.GREEN}Starting the Python Interview System...")
    
    try:
        # Initialize and run the interview system
        interview = InterviewSystem(use_seq2seq=use_seq2seq)
        interview.run()
        return True
    except Exception as e:
        print(f"{Fore.RED}Error running interview system: {str(e)}")
        return False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Python Interview System with BERT-based Feedback')
    parser.add_argument('--train-feedback', action='store_true', help='Train the feedback generation model')
    parser.add_argument('--use-seq2seq', action='store_true', help='Use the sequence-to-sequence model for feedback')
    return parser.parse_args()

def main():
    """Main function to orchestrate the entire workflow."""
    print(f"{Fore.CYAN}{Style.BRIGHT}" + "=" * 80)
    print(f"{Fore.CYAN}{Style.BRIGHT}" + " " * 15 + "PYTHON INTERVIEW SYSTEM INITIALIZATION" + " " * 15)
    print(f"{Fore.CYAN}{Style.BRIGHT}" + "=" * 80 + "\n")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Ensure all required directories exist
    ensure_directories_exist()
    
    # Step 1: Generate training data if needed
    if not generate_data():
        print(f"{Fore.RED}Failed to prepare training data. Exiting.")
        return False
    
    # Step 2: Train the classification model if needed
    if not train_model():
        print(f"{Fore.RED}Failed to train the model. Exiting.")
        return False
    
    # Step 3: Train the feedback model if requested
    if args.train_feedback:
        if not train_feedback_model():
            print(f"{Fore.YELLOW}Failed to train the feedback model. Will use rule-based feedback generation.")
    
    # Step 4: Run the interview system
    if not run_interview_system(use_seq2seq=args.use_seq2seq):
        print(f"{Fore.RED}Failed to run the interview system. Exiting.")
        return False
    
    return True

if __name__ == "__main__":
    main()