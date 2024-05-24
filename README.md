# Text-Summerizer


This project aims to develop a robust text summarization tool using advanced natural language processing (NLP) techniques. The primary goal is to create a pipeline that can ingest large text datasets, validate and transform the data, and then train a model capable of generating concise and coherent summaries. The project utilizes the Hugging Face 'transformers' library, specifically leveraging models such as Pegasus for summarization tasks. Key stages of the project include data ingestion, data validation, data transformation, and model training. The implementation is built with 'Python' and employs 'FastAPI' for the web framework, making it efficient for real-time summarization tasks.

1. Project Setup
Install Anaconda: If you haven't installed Anaconda, download and install it from Anaconda's official website.

Create a New Conda Environment:

bash
Copy code
conda create --name summary python=3.11
conda activate summary
Install Required Packages:
Create a requirements.txt file with the necessary dependencies:

Copy code
fastapi
uvicorn
transformers
torch
pydantic
yaml
Then, install the packages:

bash
Copy code
pip install -r requirements.txt
2. Project Structure
Organize your project directory as follows:

markdown
Copy code
Text-Summarization/
│
├── main.py
├── requirements.txt
├── config/
│   ├── config.yaml
│   ├── params.yaml
├── src/
│   ├── __init__.py
│   ├── textSummarizer/
│       ├── __init__.py
│       ├── components/
│       │   ├── __init__.py
│       │   ├── data_ingestion.py
│       │   ├── data_validation.py
│       │   ├── data_transformation.py
│       │   ├── model_trainer.py
│       ├── pipeline/
│           ├── __init__.py
│           ├── stage_01_data_ingestion.py
│           ├── stage_02_data_validation.py
│           ├── stage_03_data_transformation.py
│           ├── stage_04_model_trainer.py
└── artifacts/
3. Configuration Files
config/config.yaml:

yaml
Copy code
data_ingestion:
  dataset_url: "https://example.com/dataset.csv"
  raw_data_dir: "artifacts/data_ingestion/"

data_validation:
  schema_file: "schema.json"

data_transformation:
  transformed_data_dir: "artifacts/data_transformation/"

model_trainer:
  model_name: "google/pegasus-cnn_dailymail"
  output_dir: "artifacts/model_trainer/"
config/params.yaml:

yaml
Copy code
batch_size: 16
epochs: 5
learning_rate: 3e-5
4. Implementation
Data Ingestion (src/textSummarizer/components/data_ingestion.py):

python
Copy code
import os
import requests

class DataIngestion:
    def __init__(self, config):
        self.config = config

    def download_data(self):
        response = requests.get(self.config['data_ingestion']['dataset_url'])
        os.makedirs(self.config['data_ingestion']['raw_data_dir'], exist_ok=True)
        with open(os.path.join(self.config['data_ingestion']['raw_data_dir'], 'data.csv'), 'wb') as f:
            f.write(response.content)
Data Validation (src/textSummarizer/components/data_validation.py):

python
Copy code
import json
import os

class DataValidation:
    def __init__(self, config):
        self.config = config

    def validate_data(self):
        schema_path = self.config['data_validation']['schema_file']
        with open(schema_path) as schema_file:
            schema = json.load(schema_file)
        # Implement validation logic
Data Transformation (src/textSummarizer/components/data_transformation.py):

python
Copy code
import os
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.tokenizer = PegasusTokenizer.from_pretrained(config['model_trainer']['model_name'])

    def transform_data(self, data):
        # Implement transformation logic
        pass
Model Trainer (src/textSummarizer/components/model_trainer.py):

python
Copy code
from transformers import PegasusForConditionalGeneration, Trainer, TrainingArguments

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = PegasusForConditionalGeneration.from_pretrained(config['model_trainer']['model_name'])

    def train(self, train_dataset, eval_dataset):
        training_args = TrainingArguments(
            output_dir=self.config['model_trainer']['output_dir'],
            num_train_epochs=self.config['params']['epochs'],
            per_device_train_batch_size=self.config['params']['batch_size'],
            evaluation_strategy="epoch"
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        trainer.train()
Pipeline Stages (e.g., src/textSummarizer/pipeline/stage_01_data_ingestion.py):

python
Copy code
from src.textSummarizer.components.data_ingestion import DataIngestion

class DataIngestionTrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.data_ingestion = DataIngestion(config)

    def main(self):
        self.data_ingestion.download_data()
Main Script (main.py):

python
Copy code
from src.textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.textSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.textSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.textSummarizer.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline

if __name__ == "__main__":
    config = load_config("config/config.yaml")
    
    data_ingestion_pipeline = DataIngestionTrainingPipeline(config)
    data_ingestion_pipeline.main()
    
    data_validation_pipeline = DataValidationTrainingPipeline(config)
    data_validation_pipeline.main()
    
    data_transformation_pipeline = DataTransformationTrainingPipeline(config)
    data_transformation_pipeline.main()
    
    model_trainer_pipeline = ModelTrainerTrainingPipeline(config)
    model_trainer_pipeline.main()
5. Running the Project
Activate the conda environment and run the main script:

bash
Copy code
conda activate summary
python main.py
