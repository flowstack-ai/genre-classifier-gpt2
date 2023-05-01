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

## Inference
The GPT-GC was given sample movie descriptions for inference purposes and it showed impressive results for a model that was fine-tuned on a decoder only tranformer not  suitable for text classification tasks. Below are some samples with their predictions and actual labels:

| # | Movie Description | Predicted Genre | Actual Genre |
| --- | --- | --- | --- |
| 1 | Disenchanted pest sprayer and single dad Freddie Manning is always trying to hustle a tiny bit more from his disappointing life when he suffers a series of setbacks that puts him in the unlikely position of his biggest hustle yet. It all starts when he falls upon a large sum of mob money, which Freddie and his cynical, naysaying best friend, Junior, lose in a crazy chase through the city. Freddie and Junior are left owing Asian crime boss Toshiba Ohara and his spicy daughter, Han, $10,000 ~ in seven days! The pair don't know what to do until they're mistaken for the Replacement Ministers expected at a struggling, run-down Baptist church during a pest call. They trade their Bug World outfits for minister robes and soon are inspiring the tiny but enthusiastic church into a fevered fundraising campaign. Freddie knows nothing about preaching, but his hilarious, straight-from-the-heart sermons seem to touch people and no one is more surprised than Freddie when his fake healings become real ones. Along the way, the congregation and community are forever changed, but not as much as Freddie and Junior themselves. With fresh, exhilarating performances by a top-notch cast, including Al Shearer, Charlie Murphy, Tamala Jones, Brigitte Nielsen, David Alan Grier and others, The Hustle will warm your soul and tickle your funny bone. | comedy | comedy |
| 2 | Seventeen years after slaughtering all but one member of a family, a vicious serial killer known only as \"The Sandman\" awaits execution. But first, his jailers allow a minister to visit the killer to give him last rites, unaware that the minister is a voodoo priest and an ally of the condemned prisoner. The priest places a hex on the Sandman so that when he is executed, his soul migrates into a new body made of sand. To sever his ties with his former life and achieve absolute power, the sandman must find and kill a man named Griffin, the sole survivor of the last family murdered by the killer. | horror | horror |
| 3 | The Marginal Way, once an old Native American trail, is now a famous south coastal Maine attraction. Like the trail, the film roams through the Ogunquit Village, following the artists, fishermen, hippies, hermits, and the pensioners who call this place home. A portrait of Ogunquit, Maine, The Marginal Way is told through the lives of its residents and tourists in the summer of 1973. This hour-long film was shown nationally on PBS through WNET New York. | documentary | documentary |
| 4 | Sarah and Aubrey believe that a wedding can be planned with lots of fun, adventure and imagination they just have opposing opinions of what that means. The Belles aim to inspire their viewers to think outside the matrimonial box with unique interview guests and out of studio segments. | reality-tv | talk-show |
| 5 | Vince and Zach are two seasoned cops from New York City. They are both asked to resign because of several cases involving underground crime scenarios they could not close. Consequently, their Police Captain wants them out of the City; he's surreptitiously involved with the crime family and wants these guys out of the force. Both Vince and Zach come from a long line of Police officers and they are forced to temporarily resign; As the story unfolds and the action ensues they are constantly harassing each other about their dysfunctional marriages, life, and now job security. Vince and Zach decide to start a new Career in Miami as Private Detectives. In Miami They are sought out and hired by a rich and beautiful woman named Monic Lopez who doesn't trust her husband or the Legal system. Monic demands the best and hires the two to locate and keep tabs on her husband Tommy who is shifty and always unavailable. He is secretly leading a flamboyant lifestyle in Miami. Description of main conflict: Zach and Vince are exhausting their resources trying to locate Tommy. The two detectives encounter characters who have them running around the streets of Miami on a wild goose chase. One of the first encounters is at an underground cage fighting event where Tommy is a suspected Gambler. The streets of Miami are much different than what they were used to in New York City. Every lead seems to bring them to another dead end! One lead brings them to an alligator wrestler farm. At one point they end up in a female impersonator club called: \"Twist n Twirl\" Tommy is leading a secret life as a female impersonator, and is also the owner of the club. He has been doing this for years and his wife had no idea. | crime | action |

In above table we can see that out of 5 samples our model correctly predicted 3 and the fourth prediction is also really close. Pretty impressive :).

## Credits
This project was developed by Malik Talha and Hunaid Sohail. The Movie Genre Classification dataset was obtained from Kaggle. The GPT-2 model was developed by OpenAI. The Transformers library was developed by Hugging Face. Special thanks to [George Mihaila](https://github.com/gmihaila) for his helpful content regarding fine-tuning GPT-2 and other language models.
