# GPT - GC
GPT - GC (Genre Classifier) is a GPT-2 based model fine-tuned for text classification task of predicting movie genre based on its description. The goal of this model is to create a machine learning model that can accurately classify movies into their respective genres based on their textual descriptions.

## Requirements
- Python 3.x
- PyTorch
- transformers
- ftfy
- tqdm
- sklearn

## Dataset
The dataset was sourced from kaggle and can be found at this link: https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb.

## Preprocessing
Before fine-tuning the GPT-GC model, the dataset was preprocessed by tokenizing the text using the `GPT2Tokenizer` and encoding the genre labels as integers. The text was also truncated to a maximum length of 256 tokens to ensure that training is faster and smoother.

## Fine-tuning
During the fine-tuning process, the GPT-2 model was fine-tuned on the preprocessed dataset using the Hugging Face Transformers library. The training procedure utilized a batch size of 16, a maximum sequence length of 256, and involved 26 output labels. The model was trained for approximately four epochs to optimize its performance on the classification task.

## Evaluation
The performance of the model was evaluated at every epoch during training and validation steps. Following observations were made:

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
| --- | --- | --- | --- | --- |
| 1 | 1.73948 | 1.44000 | 0.48922 | 0.57324 |
| 2 | 1.38842 | 1.35102 | 0.57627 | 0.59354 |
| 3 | 1.30279 | 1.32914 | 0.59807 | 0.60073 |
| 4 | 1.26404 | 1.30743 | 0.60914 | 0.60341 |

## Credits
This project was developed by Malik Talha. The Movie Genre Classification dataset was obtained from Kaggle. The GPT-2 model was developed by OpenAI. The Transformers library was developed by Hugging Face.
