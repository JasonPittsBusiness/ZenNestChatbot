<h1 style="text-align: center;">ZenNest Chatbot Assistant</h1>
<h4 style="text-align: center;">By Jason Pitts</h4>

## Overview

Welcome to the ZenNest Chatbot assistant! This project leverages OpenAI's API to create a customer service chatbot for the ZenNest home system. The ZenNest home system is a futuristic home automation system that integrates with a chatbot assistant that aims to solve user issues quickly and efficiently. This chatbot was trained using Python, JSON training data, and integrated with Weights and Biases to record training results. Following along with this project, you will gain a basic understanding of working with LLMs, specifically OpenAI's API, to achieve desired results.

### Project Requirements

**Note**: Training GPT models is not free, though it is relatively [inexpensive](https://openai.com/api/pricing/) (gpt-3.5-turbo for this project).

1. A rudimentary understanding of [Python](https://www.python.org/)
2. A rudimentary understanding of [GitHub](https://docs.github.com/en/get-started/start-your-journey/hello-world)
3. An [OpenAI](https://platform.openai.com/docs/overview) account for access to the API
4. 1-2 USD to pay for the model training
5. A [Weights & Biases](https://wandb.ai/site) account to visualize your training results
6. An [IDE](https://www.codecademy.com/article/what-is-an-ide) (e.g., [VSCode](https://code.visualstudio.com/), [PyCharm](https://www.jetbrains.com/pycharm/)) and a [Python environment](https://www.python.org/downloads/)
7. [Training data](https://www.cloudfactory.com/training-data-guide) (Included for this project)

### Included Project Files

1. [Two sets of training data for A/B/C comparison](https://github.com/JasonPittsBusiness/ZenNestChatbot/tree/main/Training%20Data)
2. [Python scripts for splitting, validating, uploading, and interfacing the fine-tuning mechanism](https://github.com/JasonPittsBusiness/ZenNestChatbot/tree/main/Scripts)

## Introduction to the ZenNest Chatbot

To train a large language model, you first need a clear purpose. Enter the ZenNest Chatbot assistant. The assistant is named Sarah and Sarah can assist customers with any adjustments to their ZenNest home system as well as billing and other typical issues that customers may experience with a product or service. Now we have a clear purpose in modifying the base GPT responses.

## Data Preparation

When training the GPT model there are certain roles that we need to keep in mind.

- **System**: This is the way that we want the chatbot to act.
- **User**: This is the input from the user.
- **Assistant**: The response from the assistant.

With these roles in mind we can begin to craft our chatbot. As the ZenNest assistant is named Sarah and they support the ZenNest product we'll want to begin catering Sarah's responses to customers seeking assistance. Here is an example from the training data with the format required from OpenAI.

```json
{
    "messages": [
        {
            "role": "system",
            "content": "Sarah is a chatbot that assists customers having trouble with their ZenNest home system."
        },
        {
            "role": "user",
            "content": "What is a ZenNest home system?"
        },
        {
            "role": "assistant",
            "content": "ZenNest is a smart home system blending technology with Zen principles for serene living. It offers security, energy management, and personalized settings for a harmonious home environment."
        }
    ]
}
```

## Data Preparation

When training the GPT model there are certain roles that we need to keep in mind.

- **System**: This is the way that we want the chatbot to act.
- **User**: This is the input from the user.
- **Assistant**: The response from the assistant.

With these roles in mind, we can begin to craft our chatbot. As the ZenNest assistant is named Sarah and they support the ZenNest product, we'll want to begin catering Sarah's responses to customers seeking assistance. Here is an example from the training data with the format required from OpenAI.

```json
{
    "messages": [
        {
            "role": "system",
            "content": "Sarah is a chatbot that assists customers having trouble with their ZenNest home system."
        },
        {
            "role": "user",
            "content": "What is a ZenNest home system?"
        },
        {
            "role": "assistant",
            "content": "ZenNest is a smart home system blending technology with Zen principles for serene living. It offers security, energy management, and personalized settings for a harmonious home environment."
        }
    ]
}
```

Now the objective has been defined for Sarah but this won't be enough to create a functional customer service assistant of the future. We will need to tackle several key ideas in order to make Sarah function. The following is a non-exhaustive list for starting to create a functional custom chatbot with a LLM.

- **Main Objective** - The purpose and identity statements of the bot
- **Knowledge Base Integration** - Real world knowledge of the product
- **Contextual Understanding** - The base model is strong but our product is specific with related jargon
- **User Interaction Design** - How users are expected to interact with the bot
- **Multi-turn Dialogue Handling** - How the bot will respond in extended conversations
- **Error Handling** - Unexpected user inputs can cause unexpected responses that we must account for

With these principal objectives in mind we can begin to create our training data. OpenAI typically sees [clear improvements from fine-tuning on 50 to 100 training examples with gpt-3.5-turbo.](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset)

The first set of training data, [ZenHomeTrainingData.json](https://github.com/JasonPittsBusiness/ZenNestChatbot/blob/main/Training%20Data/ZenHomeTrainingData.json), consists of 100 basic conversations between the user and assistant covering our basic objectives.

Now take the [ZenHomeTrainingData.json](https://github.com/JasonPittsBusiness/ZenNestChatbot/blob/main/Training%20Data/ZenHomeTrainingData.json) and use the [SplitTrainingAndValidation.py](https://github.com/JasonPittsBusiness/ZenNestChatbot/blob/main/Scripts/SplitTrainingAndValidationData.py) to create two files that consist of:
- **ValidationData.json**: 20% of the data randomly selected from the total
- **TrainingData.json**: 80% the remaining data 

Once the data has been split into training and validation data we can check the validity of the data format. Using [ValidateTrainingData.py](https://github.com/JasonPittsBusiness/ZenNestChatbot/blob/main/Scripts/ValidateTrainingData.py) set the appropriate data paths for the input and outputs.

The output file will give an estimation of the number of epochs, batches, and tokens which will in turn estimate your total cost. OpenAI does not charge tokens for [validation data](https://platform.openai.com/docs/guides/fine-tuning/preparing-your-dataset) so we do not calculate those.

The following are examples of errors and token counts to estimate costs. Note: No errors exist in this dataset so the dictionary is empty.

```json
{
    "training_data_errors": {
        "dataset_name": "Training Data",
        "errors": {}
    }
    "pricing_info": {
        "billing_tokens": 14490,
        "n_epochs": 3,
        "total_tokens_charged": 43470
    }
}
```

## Uploading the data

Once the data has returned with no errors and we understand the estimated pricing (around $0.26 at the time of this project) we can begin to upload the training data. Open the [UploadTrainingData.py](https://github.com/JasonPittsBusiness/ZenNestChatbot/blob/main/Scripts/UploadTrainingData.py) and set your file paths. This is the first time that you will need to ensure that your OpenAI API key [environment variable](https://github.com/JasonPittsBusiness/ZenNestChatbot/blob/main/ExampleEnv.txt) has been set.

Assuming the script processes with no errors you will be provided two fileIDs returned from the API. Keep these fileIDs as they will be used in the next step to initiate the training of the model.

## Training the model

Using the fileIDs that were returned in Uploading the data, begin filling in your relevant data to [StartFineTuningJob.py](https://github.com/JasonPittsBusiness/ZenNestChatbot/blob/main/Scripts/StartFineTuningJob.py). Note: You will need to provide your Weights & Biases API key to [integrate with the OpenAI API](https://platform.openai.com/docs/guides/fine-tuning/fine-tuning-integrations).

You can review the job as it completes in your Weights & Biases project as well as at the [fine-tuning interface](https://platform.openai.com/finetune). Once the fine-tuning has completed you can begin to work with your newly trained chatbot assistant at the [OpenAI Playground](https://platform.openai.com/playground).

## Further training of the model



Learn more at https://jasonpittsbusiness.github.io/ZenNest%20Chatbot/.
