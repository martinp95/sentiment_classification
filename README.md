# Sentiment Classification Project

## Overview
This project aims to classify the sentiment of text data using machine learning techniques. The goal is to determine whether a given piece of text expresses a positive or negative sentiment.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [License](#license)

## Installation
To get started, clone the repository and set up the conda environment:
```bash
git clone https://github.com/martinp95/sentiment_classification.git
cd sentiment_classification
conda env create -f environment.yaml
conda activate sentiment_classification
```

## Usage
To run the sentiment classification model, use the following command:
```bash
python src/main.py
```

## Model
The model used for sentiment classification is a fine-tuned DistilBERT model. It is trained to achieve high accuracy in sentiment prediction.

## Results
The model achieves high accuracy in classifying text as positive or negative.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.