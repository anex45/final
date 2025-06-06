# Terminal Python Interview System with Advanced ML Models

This project implements a terminal-based Python interview system that uses fine-tuned machine learning models to evaluate candidate responses and provide dynamic feedback. The system includes components for generating training data, fine-tuning models (BERT, Seq2Seq, and T5), conducting interviews, and providing personalized feedback.

## Features

- Training dataset generator for interview Q&A pairs
- Multiple model options for feedback generation:
  - BERT model for basic evaluation
  - Sequence-to-sequence model for improved feedback
  - T5 model for advanced, contextual feedback generation
- Terminal-based interview interface
- Dynamic feedback mechanism based on semantic analysis
- No predefined feedback phrases - responses are generated contextually

## Installation

```bash
pip install -r requirements.txt
```

## Usage

You can run the entire system with a single command:
```bash
python main_unified.py
```

This will:
1. Generate training data (if needed)
2. Generate feedback dataset (if needed)
3. Fine-tune the BERT model (if needed)
4. Fine-tune the feedback model (if needed)
5. Start the interview system

You can also use different feedback generation models:
```bash
# Use the sequence-to-sequence model for feedback generation
python main_unified.py --use-seq2seq

# Use the T5 model for advanced feedback generation
python main_unified.py --use-t5
```

Alternatively, you can run each component separately:

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
- `model_trainer_seq2seq.py`: Trains sequence-to-sequence model for feedback
- `interview_system.py`: Main interface for conducting interviews
- `feedback_generator.py`: Generates dynamic feedback based on candidate responses
- `feedback_generator_seq2seq.py`: Generates feedback using sequence-to-sequence model
- `feedback_generator_t5.py`: Generates advanced feedback using T5 model
- `utils.py`: Utility functions for the system
- `data/`: Directory containing training and evaluation datasets
- `models/`: Directory for storing fine-tuned models#   f i n a l 
 
 