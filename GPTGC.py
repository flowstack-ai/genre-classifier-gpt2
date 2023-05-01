"""
    GPT-GC (Genre Classifier)
    GPT-GC is a GPT-2 based model fine-tuned for classifying movie genres.
"""
import torch, logging
from torch.utils.data import DataLoader, random_split
from Trainer import Trainer
from DataLoader import MovieGenresDataset, GPT2ClassificationCollate
from Common import *
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score
from transformers import (GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification, get_linear_schedule_with_warmup, set_seed)
import matplotlib.pyplot as plt

# Model Configuration
params = {
    "NAME": "GPTGC",
    "LOG_DIR": "logs/",
    "MODEL_DIR": "models/",
    "BATCH_SIZE": 16,
    "EPOCHS": 4,
    "LEARNING_RATE": 2e-5,
    "WARMUP_STEPS": 0,
    "MAX_SEQ_LEN": 256,
    "GRADIENT_ACCUMULATION_STEPS": 1,
    "WEIGHT_DECAY": 0.01,
    "EPS": 1e-8,
    "MAX_GRAD_NORM": 1.0,
    "SEED": 42
}

class GPTGC():
    def __init__(self, device, fine_tune=False,  resume=False):
        set_seed(params["SEED"])
        self.device = device
        self.base_model = "gpt2"
        self.model = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.tokenizer = None
        self.load_tokenizer()
        if fine_tune:
            print("-> Preparing to fine-tune GPT-2 on movie genre classification dataset ...")
            self.load_dataset()
            self.dataset_info()
            self.init_model()
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=params["LEARNING_RATE"], eps=params["EPS"])
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                             num_warmup_steps=params["WARMUP_STEPS"], 
                                                             num_training_steps=len(self.train_dataloader) * params["EPOCHS"])
            self.trainer = Trainer(self.model, self.optimizer, self.scheduler, self.device)
        else:
            print("-> Initializing GPT-GC ...")
            params["OUTPUT_LABELS"] = OUTPUT_LABELS
            params["N_OUTPUT_LABELS"] = N_OUTPUT_LABELS
            self.init_model()
            self.load_model()
    
    def load_tokenizer(self):
        print('-> Loading tokenizer ...')
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_dataset(self):
        print("-> Loading Train/Val datasets ...")
        self.train_dataset = MovieGenresDataset("dataset/train_data.txt")
        self.test_dataset = MovieGenresDataset("dataset/test_data_solution.txt")
        # Keep only 20% of test dataset for validation
        test_size = int(0.2 * len(self.test_dataset))
        self.test_dataset, discarded_dataset = random_split(self.test_dataset, [test_size, len(self.test_dataset) - test_size])
        params["OUTPUT_LABELS"] = self.train_dataset.output_labels
        params["N_OUTPUT_LABELS"] = self.train_dataset.n_output_labels
        print("-> Loading data collator ...")
        collator = GPT2ClassificationCollate(self.tokenizer, params["OUTPUT_LABELS"], params["MAX_SEQ_LEN"])
        print("-> Initializing Train/Val dataloaders ...")
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=params["BATCH_SIZE"], shuffle=True, collate_fn=collator)
        self.val_dataloader = DataLoader(self.test_dataset, batch_size=params["BATCH_SIZE"], shuffle=False, collate_fn=collator)
    
    def dataset_info(self):
        print("-> Dataset information:")
        print("\t- Train Dataset Count: ", len(self.train_dataset))
        print("\t- Test Dataset Count: ", len(self.test_dataset))
        # print("\t- Output Labels: ", OUTPUT_LABELS)
        print("\t- Number of Output Labels: ", params["N_OUTPUT_LABELS"])
        # print("\t- Sample Text: ", train_dataset[0]['text'])
        # print("\t- Sample Label: ", train_dataset[0]['label'])
        print("\t- Batch Size: ", params["BATCH_SIZE"])
        print("\t- Number of Train Batches: ", len(self.train_dataloader))
        print("\t- Number of Test Batches: ", len(self.val_dataloader))
    
    def init_model(self):
        print("-> Configuring GPT-2 model ...")
        model_config = GPT2Config.from_pretrained(self.base_model, num_labels=params["N_OUTPUT_LABELS"])
        print("-> Loading GPT-2 model ...")
        self.model = GPT2ForSequenceClassification.from_pretrained(self.base_model, config=model_config)
        # resize model embedding to match new tokenizer and fix model padding token id
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.to(self.device)
    
    def fit(self):
        # Init a log file
        logging.basicConfig(filename=f"{params['LOG_DIR']}/{params['NAME']}.csv", level=logging.INFO)
        print("-> Fine-tuning starts ...")
        losses = {
            'train_loss': [], 
            'val_loss': []
        }
        accuracies = {
            'train_accuracy': [], 
            'val_accuracy': []
        }
        for epoch in range(params["EPOCHS"]):
            print("x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x")
            print(f"Epoch: [{epoch+1}]\[{params['EPOCHS']}]")
            # Perform one full pass over the training set.
            print('Training ...')
            train_labels, train_predict, train_loss = self.trainer.train(self.train_dataloader)
            train_accuracy = accuracy_score(train_labels, train_predict)
            # Get prediction form model on validation data. 
            print('Validating ...')
            valid_labels, valid_predict, val_loss = self.validation(self.val_dataloader)
            val_accuracy = accuracy_score(valid_labels, valid_predict)
            # Print loss and accuracy values to see how training evolves.
            print("Train_Loss: %.5f - Val_Loss: %.5f - Train_Accuracy: %.5f - Val_Accuracy: %.5f"%(train_loss, val_loss, train_accuracy, val_accuracy))
            if epoch == 0:
                logging.info("Epoch, Training Loss, Validation Loss, Training Accuracy, Validation Accuracy")
            logging.info(f"{epoch}, {train_loss:.6f}, {val_loss:.6f}, {train_accuracy:.6f}, {val_accuracy:.6f}")
            # Store the loss value for plotting the learning curve.
            losses['train_loss'].append(train_loss)
            losses['val_loss'].append(val_loss)
            accuracies['train_accuracy'].append(train_accuracy)
            accuracies['val_accuracy'].append(val_accuracy)
            # Save the model
            self.save_model()
        return losses, accuracies
    
    def validation(self, dataloader):
        predictions_labels = []
        true_labels = []
        total_loss = 0
        self.model.eval()
        for batch in tqdm(dataloader, total=len(dataloader)):
            true_labels += batch['labels'].numpy().flatten().tolist()
            batch = { k:v.type(torch.long).to(self.device) for k,v in batch.items() }
            with torch.no_grad():        
                outputs = self.model(**batch)
                loss, logits = outputs[:2]
                logits = logits.detach().cpu().numpy()
                total_loss += loss.item()
                predict_content = logits.argmax(axis=-1).flatten().tolist()
                predictions_labels += predict_content
        avg_epoch_loss = total_loss / len(dataloader)
        return true_labels, predictions_labels, avg_epoch_loss
    
    def save_model(self):
        print("-> Saving model ...")
        torch.save(self.model.state_dict(), f"{params['MODEL_DIR']}/{params['NAME']}.pt")
    
    def load_model(self):
        print("-> Loading the fine-tuned GPTGC model from disk ...")
        self.model.load_state_dict(torch.load(f"{params['MODEL_DIR']}/{params['NAME']}.pt"))
        print("-> GPT-GC Model loaded successfully!")
    
    def predict(self, text):
        print("-> Predicting ...")
        predictions_labels = []
        dataset = MovieGenresDataset("", infer=True, data_instance=text)
        collator = GPT2ClassificationCollate(self.tokenizer, params["OUTPUT_LABELS"], params["MAX_SEQ_LEN"], infer=True)
        dataloader = DataLoader(dataset, batch_size=params["BATCH_SIZE"], shuffle=False, collate_fn=collator)
        self.model.eval()
        for batch in dataloader:
            batch = { k:v.type(torch.long).to(self.device) for k,v in batch.items() }
            with torch.no_grad():        
                outputs = self.model(**batch)
                loss, logits = outputs[:2]
                logits = logits.detach().cpu().numpy()
                predict_content = logits.argmax(axis=-1).flatten().tolist()
                predictions_labels += predict_content
        result = [key for key, value in params["OUTPUT_LABELS"].items() if value == predictions_labels[0]][0]
        print("-> Done. Prediction: ", result)
        return result

    def plot_accuracy(self, accuracies):
        plt.plot(accuracies['train_accuracy'], label='Train Accuracy')
        plt.plot(accuracies['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.show()
    
    def plot_loss(self, losses):
        plt.plot(losses['train_loss'], label='Train Loss')
        plt.plot(losses['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()