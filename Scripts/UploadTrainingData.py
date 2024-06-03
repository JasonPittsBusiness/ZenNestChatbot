#Uploads your training and validation files to OpenAI
#Your API key must be initilized in a .env
#Learn more env: https://www.geeksforgeeks.org/how-to-create-and-use-env-files-in-python/
#Do not commit your API key
#Learn more fine tuning: https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset
#FileIDs will be returned to the terminal
import os
from openai import OpenAI
from dotenv import load_dotenv

#Load environment variables, your OpenAI API Key from .env
load_dotenv()

API_KEY = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=API_KEY)

#Upload the training file
training_file_response = client.files.create(
  file=open(r"\YourTrainingData.json", "rb"),
  purpose="fine-tune"
)

#Returns the training file ID from the OpenAI API
print(f"Training file uploaded: {training_file_response}")

#Upload the validation file
validation_file_response = client.files.create(
  file=open(r"\YourValidationData.json", "rb"),
  purpose="fine-tune"
)

#Returns the validation file ID from the OpenAI API
print(f"Validation file uploaded: {validation_file_response}")
