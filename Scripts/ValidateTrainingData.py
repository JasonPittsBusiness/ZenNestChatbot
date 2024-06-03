#Learn More
#https://cookbook.openai.com/examples/chat_finetuning_data_prep
#This script checks the validity of your training data and estimates costs
#The results of your data and costs are output to a specified .json
#Please review the output data before uploading to the OpenAI API

import json
import tiktoken  # for token counting
import numpy as np
from collections import defaultdict

# Define file paths
training_data_path = r"\YourTrainingData.json"
validation_data_path = r"\YourValidationData.json"
output_file_path = r"\YourDataResults.json"

#Load the dataset
def load_dataset(file_path):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

training_dataset = load_dataset(training_data_path)
validation_dataset = load_dataset(validation_data_path)

#Function to check dataset format
def check_format(dataset):
    format_errors = defaultdict(int)
    for ex in dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue
        
        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue
        
        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1
            
            if any(k not in ("role", "content", "name", "function_call", "weight") for k in message):
                format_errors["message_unrecognized_key"] += 1
            
            if message.get("role", None) not in ("system", "user", "assistant", "function"):
                format_errors["unrecognized_role"] += 1
            
            content = message.get("content", None)
            function_call = message.get("function_call", None)
            
            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1
        
        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    return format_errors

#Check format of datasets
training_format_errors = check_format(training_dataset)
validation_format_errors = check_format(validation_dataset)

#Collect format errors into a dictionary
def format_errors_to_dict(format_errors, dataset_name):
    errors_dict = {"dataset_name": dataset_name, "errors": {}}
    if format_errors:
        for k, v in format_errors.items():
            errors_dict["errors"][k] = v
    return errors_dict

training_errors_dict = format_errors_to_dict(training_format_errors, "Training Data")
validation_errors_dict = format_errors_to_dict(validation_format_errors, "Validation Data")

#Token counting functions
encoding = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

def print_distribution(values, name):
    distribution = {
        "min": min(values),
        "max": max(values),
        "mean": np.mean(values),
        "median": np.median(values),
        "p10": np.quantile(values, 0.1),
        "p90": np.quantile(values, 0.9)
    }
    return distribution

#Analyze datasets
def analyze_dataset(dataset, dataset_name):
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))
    
    dataset_analysis = {
        "n_missing_system": n_missing_system,
        "n_missing_user": n_missing_user,
        "n_messages_distribution": print_distribution(n_messages, f"num_messages_per_example in {dataset_name}"),
        "convo_lens_distribution": print_distribution(convo_lens, f"num_total_tokens_per_example in {dataset_name}"),
        "assistant_message_lens_distribution": print_distribution(assistant_message_lens, f"num_assistant_tokens_per_example in {dataset_name}"),
        "n_too_long": sum(l > 4096 for l in convo_lens)
    }
    return dataset_analysis, convo_lens

training_data_analysis, training_convo_lens = analyze_dataset(training_dataset, "Training Data")
validation_data_analysis, validation_convo_lens = analyze_dataset(validation_dataset, "Validation Data")

#Initial dataset stats
def dataset_stats(dataset, name):
    stats = {
        "num_examples": len(dataset),
        "first_example": dataset[0] if dataset else "No data"
    }
    return stats

training_stats = dataset_stats(training_dataset, "Training Data")
validation_stats = dataset_stats(validation_dataset, "Validation Data")

# Pricing and default n_epochs estimate
MAX_TOKENS_PER_EXAMPLE = 4096
TARGET_EPOCHS = 3
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25

n_epochs = TARGET_EPOCHS
n_train_examples = len(training_dataset)
if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
    n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
    n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

n_billing_tokens_in_dataset = sum(min(MAX_TOKENS_PER_EXAMPLE, length) for length in training_convo_lens)
pricing_info = {
    "billing_tokens": n_billing_tokens_in_dataset,
    "n_epochs": n_epochs,
    "total_tokens_charged": n_epochs * n_billing_tokens_in_dataset
}

#Collect results
results = {
    "training_data_errors": training_errors_dict,
    "validation_data_errors": validation_errors_dict,
    "training_data_analysis": training_data_analysis,
    "validation_data_analysis": validation_data_analysis,
    "training_stats": training_stats,
    "validation_stats": validation_stats,
    "pricing_info": pricing_info
}

#Save results to output json
with open(output_file_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)

print(f"Results saved to {output_file_path}")
