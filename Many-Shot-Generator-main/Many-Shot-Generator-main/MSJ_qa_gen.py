import json
import logging
import os
from tqdm import tqdm
import requests
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

config = load_json('config.json')
prompts = load_json('prompts.json')

model_name = config['MODEL_NAME']
output_dir = config['OUTPUT_DIR']
api_url = config['REQUEST_POST']

os.makedirs(output_dir, exist_ok=True)

def generate_questions(num_questions, question_prompts):
    questions = []
    unique_questions = set()
    
    with tqdm(total=num_questions, desc="Generating Questions", unit="question") as pbar:
        while len(questions) < num_questions:
            question_type = random.choice(list(question_prompts.keys()))
            question_prompt = question_prompts[question_type]['question']
            messages = [{"role": "user", "content": question_prompt}]
            response = send_api_request(messages)
            question = response['content'].strip()
            
            if question and question not in unique_questions:
                questions.append({
                    "type": question_type,
                    "prompt": question_prompt,
                    "question": question
                })
                unique_questions.add(question)
                pbar.update(1)
        
    temp_file = os.path.join(output_dir, "qa_temp.json")
    with open(temp_file, "w", encoding="utf-8") as file:
        json.dump(questions, file, indent=2, ensure_ascii=False)
    logging.info(f"{len(questions)} questions saved to {temp_file}")
                
    return questions

def generate_answers(questions, answer_prompts):
    answers = []
    for question_data in tqdm(questions, desc="Generating Answers", unit="answer"):
        question_type = question_data["type"]
        question = question_data["question"]
        answer_prompt = answer_prompts[question_type]['answer']
        input_for_model = f"{answer_prompt} Q: {question}"
        messages = [{"role": "user", "content": input_for_model}]
        response = send_api_request(messages)
        answers.append(response['content'])
    return answers

def send_api_request(messages):
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": True
    }
    response = requests.post(api_url, json=payload)
    response.raise_for_status()
    output = ""
    for line in response.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            message = body.get("message", "")
            content = message.get("content", "")
            output += content
            print(content, end="", flush=True)
        if body.get("done", False):
            message["content"] = output
            return message

def main():
    while True:
        print("\nWelcome to the Many-Shots Jailbreak Q&A Pair Generator!")
        print("Please select an option:")
        print("1. Generate Q&A pairs with random prompts.")
        print("2. Generate a Q&A pair answering deceptively to a normal question.")
        print("3. Generate a Q&A pair directly answering a harmful question.")
        print("4. Generate a Q&A pair directly answering a misleading question.")
        print("Type 'quit' or 'exit' to end the program.")

        choice = input("\nYour choice: ").strip().lower()
        if choice in ["quit", "exit"]:
            print("Thank you for using the Q&A Pair Generator. Goodbye!")
            break

        try:
            choice_number = int(choice)  # Convert the choice to an integer
            if choice_number < 1 or choice_number > 4:
                raise ValueError("Invalid choice. Please enter a number between 1 and 4.")

            num_pairs = int(input("Enter the number of Q&A pairs to generate: "))
            if num_pairs <= 0:
                raise ValueError("Number of Q&A pairs must be a positive integer.")

            if choice_number == 1:
                question_prompts = {k: v for k, v in prompts.items() if k in ["2", "3", "4"]}
            else:
                question_prompts = {str(choice_number): prompts[str(choice_number)]}
            
            questions = generate_questions(num_pairs, question_prompts)
            answers = generate_answers(questions, prompts)

            # Save answers to a file (or other desired operations)
            answer_file = os.path.join(output_dir, "answers.json")
            with open(answer_file, "w", encoding="utf-8") as file:
                json.dump(answers, file, indent=2, ensure_ascii=False)
            logging.info(f"{len(answers)} answers saved to {answer_file}")

        except ValueError as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()