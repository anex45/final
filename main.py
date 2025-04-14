#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main Entry Point for Python Interview System

This script orchestrates the entire Python interview system workflow:
1. Generates training data (if needed)
2. Trains the BERT model
3. Runs the terminal interview system

Ensures that the model is properly trained before conducting interviews.
"""

import os
import sys
import json
from colorama import init, Fore, Style

# Import system components
import data_generator
import model_trainer
from interview_system import InterviewSystem
from utils import ensure_directories_exist

# Initialize colorama for colored terminal output
init(autoreset=True)

def check_data_exists():
    """Check if training data exists."""
    return os.path.exists('data/training_data.json') and os.path.getsize('data/training_data.json') > 0

def check_model_exists():
    """Check if trained model exists."""
    return os.path.exists('models/bert_interview_final.pt')

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

def run_interview_system():
    """Run the terminal interview system."""
    print(f"{Fore.GREEN}Starting the Python Interview System...")
    
    try:
        # Initialize and run the interview system
        interview = InterviewSystem()
        interview.run()
        return True
    except Exception as e:
        print(f"{Fore.RED}Error running interview system: {str(e)}")
        return False

def main():
    """Main function to orchestrate the entire workflow."""
    print(f"{Fore.CYAN}{Style.BRIGHT}" + "=" * 80)
    print(f"{Fore.CYAN}{Style.BRIGHT}" + " " * 15 + "PYTHON INTERVIEW SYSTEM INITIALIZATION" + " " * 15)
    print(f"{Fore.CYAN}{Style.BRIGHT}" + "=" * 80 + "\n")
    
    # Ensure all required directories exist
    ensure_directories_exist()
    
    # Step 1: Generate training data if needed
    if not generate_data():
        print(f"{Fore.RED}Failed to prepare training data. Exiting.")
        return False
    
    # Step 2: Train the model if needed
    if not train_model():
        print(f"{Fore.RED}Failed to train the model. Exiting.")
        return False
    
    # Step 3: Run the interview system
    run_interview_system()
    
    return True

if __name__ == "__main__":
    main()