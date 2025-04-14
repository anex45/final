# Terminal Python Interview System with BERT

This project implements a terminal-based Python interview system that uses a fine-tuned BERT model to evaluate candidate responses and provide dynamic feedback. The system includes components for generating training data, fine-tuning the BERT model, conducting interviews, and providing personalized feedback.

## Features

- Training dataset generator for interview Q&A pairs
- BERT model fine-tuning for response evaluation
- Terminal-based interview interface
- Dynamic feedback mechanism based on semantic analysis
- No predefined feedback phrases - responses are generated contextually

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Generate training data:
```bash
python data_generator.py
```

2. Fine-tune the BERT model:
```bash
python model_trainer.py
```

3. Start the interview system:
```bash
python interview_system.py
```

## Project Structure

- `data_generator.py`: Generates training dataset of interview Q&A pairs
- `model_trainer.py`: Fine-tunes BERT model on the training dataset
- `interview_system.py`: Main interface for conducting interviews
- `feedback_generator.py`: Generates dynamic feedback based on candidate responses
- `utils.py`: Utility functions for the system
- `data/`: Directory containing training and evaluation datasets
- `models/`: Directory for storing fine-tuned models