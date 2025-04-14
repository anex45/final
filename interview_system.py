#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Terminal Python Interview System

This module implements a terminal-based Python interview system that uses a
fine-tuned BERT model to evaluate candidate responses and provide dynamic feedback.
"""

import os
import json
import random
import time
from colorama import init, Fore, Style
from prettytable import PrettyTable
from feedback_generator import FeedbackGenerator

# Initialize colorama for colored terminal output
init(autoreset=True)

class InterviewSystem:
    """Terminal-based Python interview system."""
    
    def __init__(self):
        # Load questions from training data
        self.questions = self.load_questions()
        self.categories = list(set([q['category'] for q in self.questions]))
        
        # Initialize feedback generator
        self.feedback_gen = FeedbackGenerator()
        
        # Interview session state
        self.current_session = {
            'candidate_name': '',
            'questions_asked': [],
            'responses': [],
            'scores': [],
            'feedback': []
        }
    
    def load_questions(self, filepath='data/training_data.json'):
        """Load interview questions from training data."""
        if not os.path.exists(filepath):
            print(f"Warning: Training data file {filepath} not found.")
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract unique questions with their categories
        questions = []
        seen_questions = set()
        
        for item in data:
            question = item['question']
            if question not in seen_questions:
                questions.append({
                    'question': question,
                    'category': item['category']
                })
                seen_questions.add(question)
        
        return questions
    
    def display_welcome(self):
        """Display welcome message and instructions."""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}" + "=" * 80)
        print(f"{Fore.CYAN}{Style.BRIGHT}" + " " * 20 + "PYTHON INTERVIEW SYSTEM" + " " * 20)
        print(f"{Fore.CYAN}{Style.BRIGHT}" + "=" * 80 + "\n")
        
        print(f"{Fore.WHITE}Welcome to the Python Interview System!")
        print("This system will ask you Python interview questions and provide")
        print("dynamic feedback on your responses based on a fine-tuned BERT model.\n")
        
        print(f"{Fore.YELLOW}Instructions:")
        print("1. You will be asked a series of Python interview questions.")
        print("2. Type your answer and press Enter to submit.")
        print("3. The system will evaluate your response and provide feedback.")
        print("4. Type 'exit' at any time to end the interview.\n")
    
    def get_candidate_info(self):
        """Get candidate information."""
        print(f"{Fore.GREEN}Please enter your name:")
        name = input("> ")
        self.current_session['candidate_name'] = name
        print(f"\nHello, {name}! Let's begin the interview.\n")
    
    def select_category(self):
        """Let the candidate select a question category."""
        print(f"{Fore.GREEN}Please select a question category:")
        
        for i, category in enumerate(self.categories, 1):
            # Format category name for display (replace underscores with spaces and capitalize)
            display_name = category.replace('_', ' ').title()
            print(f"{i}. {display_name}")
        
        while True:
            try:
                choice = input("> ")
                if choice.lower() == 'exit':
                    return None
                
                choice = int(choice)
                if 1 <= choice <= len(self.categories):
                    return self.categories[choice - 1]
                else:
                    print(f"{Fore.RED}Invalid choice. Please enter a number between 1 and {len(self.categories)}.")
            except ValueError:
                print(f"{Fore.RED}Invalid input. Please enter a number.")
    
    def select_question(self, category):
        """Select a random question from the chosen category."""
        # Filter questions by category
        category_questions = [q for q in self.questions if q['category'] == category]
        
        # Filter out questions that have already been asked in this session
        available_questions = [q for q in category_questions 
                              if q['question'] not in self.current_session['questions_asked']]
        
        if not available_questions:
            print(f"{Fore.YELLOW}No more questions available in this category.")
            return None
        
        # Select a random question
        question = random.choice(available_questions)['question']
        self.current_session['questions_asked'].append(question)
        
        return question
    
    def ask_question(self, question):
        """Ask a question and get the candidate's response."""
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Question:{Style.RESET_ALL} {question}")
        print(f"{Fore.GREEN}Your answer (type 'exit' to end the interview):")
        
        # Get multiline input
        print("> ", end="")
        lines = []
        while True:
            line = input()
            if line.lower() == 'exit':
                return 'exit'
            if line.strip() == '' and lines and lines[-1].strip() == '':
                break
            lines.append(line)
        
        response = '\n'.join(lines)
        self.current_session['responses'].append(response)
        
        return response
    
    def evaluate_response(self, question, response):
        """Evaluate the candidate's response and provide feedback."""
        print(f"\n{Fore.YELLOW}Evaluating your response...")
        
        # Simulate processing time for a more realistic experience
        time.sleep(1.5)
        
        # Get score and feedback
        score = self.feedback_gen.get_score(question, response)
        feedback = self.feedback_gen.generate_feedback(question, response)
        
        self.current_session['scores'].append(score)
        self.current_session['feedback'].append(feedback)
        
        # Display results
        print(f"\n{Fore.CYAN}{Style.BRIGHT}Evaluation Results:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Score: {score}/100")
        print(f"\n{Fore.MAGENTA}Feedback:{Style.RESET_ALL} {feedback}\n")
        
        return score, feedback
    
    def display_summary(self):
        """Display a summary of the interview session."""
        if not self.current_session['questions_asked']:
            return
        
        print(f"\n{Fore.CYAN}{Style.BRIGHT}" + "=" * 80)
        print(f"{Fore.CYAN}{Style.BRIGHT}" + " " * 20 + "INTERVIEW SUMMARY" + " " * 20)
        print(f"{Fore.CYAN}{Style.BRIGHT}" + "=" * 80 + "\n")
        
        print(f"{Fore.WHITE}Candidate: {self.current_session['candidate_name']}")
        print(f"Questions Answered: {len(self.current_session['questions_asked'])}")
        
        if self.current_session['scores']:
            avg_score = sum(self.current_session['scores']) / len(self.current_session['scores'])
            print(f"Average Score: {avg_score:.1f}/100\n")
        
        # Create a table for detailed results
        table = PrettyTable()
        table.field_names = ["Question", "Score"]
        table.align["Question"] = "l"
        table.align["Score"] = "r"
        
        for i, question in enumerate(self.current_session['questions_asked']):
            if i < len(self.current_session['scores']):
                table.add_row([question[:50] + "..." if len(question) > 50 else question, 
                              f"{self.current_session['scores'][i]}/100"])
        
        print(table)
        print("\nThank you for completing the interview!\n")
    
    def save_session(self):
        """Save the interview session to a file."""
        if not self.current_session['questions_asked']:
            return
        
        # Create directory if it doesn't exist
        if not os.path.exists('sessions'):
            os.makedirs('sessions')
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"sessions/interview_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.current_session, f, indent=2)
        
        print(f"Interview session saved to {filename}")
    
    def run(self):
        """Run the interview system."""
        self.display_welcome()
        self.get_candidate_info()
        
        while True:
            # Select category
            category = self.select_category()
            if category is None:
                break
            
            # Select and ask question
            question = self.select_question(category)
            if question is None:
                continue
            
            # Get response
            response = self.ask_question(question)
            if response == 'exit':
                break
            
            # Evaluate response
            self.evaluate_response(question, response)
            
            # Ask if the candidate wants to continue
            print(f"{Fore.GREEN}Do you want to answer another question? (y/n)")
            choice = input("> ").lower()
            if choice != 'y':
                break
        
        # Display summary and save session
        self.display_summary()
        self.save_session()

def main():
    """Main function to run the interview system."""
    # Create necessary directories
    for directory in ['data', 'models', 'sessions']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Run the interview system
    interview_system = InterviewSystem()
    interview_system.run()

if __name__ == "__main__":
    main()