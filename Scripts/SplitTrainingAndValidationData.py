import json
import random

#Define the file paths for your training data
#The input file will be split into a validation and a training .json
#Input file is your target input data
#Validation and training files are output data
#The validation file will consist of 20% of the split data and the 
#training will be the emaining 80%
input_file_path = "/YourInput.json"
validation_file_path = "/YourOutput.json"
training_file_path = "/YourOutput.json"

with open(input_file_path, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

split_ratio = 0.2  # 20% for validation
num_validation = int(len(dataset) * split_ratio)

#Set seed for comparisons among training but 
#You could randomize the seed each time to get
#A larger spread comparison of the data
random.seed(42)
validation_data = random.sample(dataset, num_validation)

#Move the remaining data to the training set
training_data = [data for data in dataset if data not in validation_data]

#Write and save validation data
with open(validation_file_path, 'w', encoding='utf-8') as f:
    for item in validation_data:
        f.write(json.dumps(item) + "\n")

#Write and save training data
with open(training_file_path, 'w', encoding='utf-8') as f:
    for item in training_data:
        f.write(json.dumps(item) + "\n")
