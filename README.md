# NER (Named Entity Recognition) - Mountains

This project aims to perform Named Entity Recognition (NER) to identify mountain names in text using a fine-tuned RoBERTa model. The model is trained to classify each token as either part of a mountain name or not, using a dataset from Hugging Face.

## Project Structure

Here's an overview of the files in the repository:

- `Additional_solutions.ipynb`: Jupyter notebook with alternative solutions and experiments related to the NER task.
- `Dockerfile`: The configuration file used to build the Docker image for the project.
- `fine_tuning_RoBERTa(colab).ipynb`: Jupyter notebook for fine-tuning the RoBERTa model on Google Colab.
- `fine_tuning_roberta(colab).py`: Python script equivalent of the fine-tuning process for use outside of Jupyter.
- `inference.py`: Script for running inference using the fine-tuned model to identify mountain names in input text.
- `process_dataset.ipynb`: Jupyter notebook for preprocessing the dataset before training.
- `README.md`: This file, providing an overview and instructions for the project.
- `requirements.txt`: List of dependencies required to run the project.
- `training_new_model.ipynb`: Jupyter notebook for training a new model from scratch.
- `Report.pdf`: This file provides report with used and future strategies and results.
- `demo.ipynb`: Demo

### Directories

- `data/`: Contains dataset files used during model training and inference. This directory also includes a `models/` subdirectory, which contains the trained model files.

## Dataset

The dataset used for this project is from Hugging Face and is designed for NER tasks, specifically focusing on mountain names. It consists of labeled tokens where each token is classified as either a part of a mountain name or not. The dataset can be found here: [NER-Mountains Dataset on Hugging Face](https://huggingface.co/datasets/telord/ner-mountains-first-dataset).

## Model
The main resulting model is fine-tuned with LORA [distilroberta-base](https://huggingface.co/day88ild/ner_mountain_roberta_fine_tuned/tree/main)

## Getting Started

Follow these instructions to build the Docker image and run the inference script.

### Prerequisites

Make sure you have Docker installed on your machine. If not, install it from the [official Docker website](https://www.docker.com/get-started).

### Building the Docker Image

To build the Docker image directly from the GitHub repository, use the following command (it might take a while due to installation of large python libraries):

```bash
docker build -t ner-mountain https://github.com/day88ild/NER_mountain.git
```

To run a container with inference.py run the following command after creating an image:

```bash
docker run -it --rm ner-mountain
```
