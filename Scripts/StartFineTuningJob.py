#Sets up the fine-tuning "payload" to be sent to OpenAI
#Starts the fine-tuning job built with the training
#and validation files that we have created and uploaded
#Uses the Weights&Biases integration to track data visualizations
#Learn more: https://platform.openai.com/docs/guides/fine-tuning/fine-tuning-integrations

import os
import requests
from dotenv import load_dotenv

#Load environment variables, your OpenAI API Key from .env
load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')

#Define the file IDs for training and validation
#If you need the fileIDs you can also find them in the OpenAI playground
training_file_id = "Returned file-ID from UploadTrainingData.py" #Example file-IDzxcvzxcvzcv
validation_file_id = "Returned file-ID from UploadTrainingData.py" #Example file-IDzxcvzxcvzcv

#Set up the fine-tuning job payload
payload = {
    "model": "gpt-3.5-turbo",
    "training_file": training_file_id,
    "validation_file": validation_file_id,
    "integrations": [
        {
            "type": "wandb", #Using the Weights&Biases integration
            "wandb": {
                "project": "ZenNest Chatbot Assistant",  # Replace with your Weights&Biases project name
                "entity": ""  # Replace with your Weights&Biases entity
            }
        }
    ]
}

#Create the fine-tuning job
response = requests.post(
    "https://api.openai.com/v1/fine_tuning/jobs",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json=payload
)

#Check the response
if response.status_code == 200:
    fine_tuning_response = response.json()
    print(f"Fine-tuning job created: {fine_tuning_response['id']}")
else:
    print(f"Error: {response.status_code}")
    print(response.json())
