import torch
from tqdm import tqdm

class Trainer():
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, dataloader):
        predictions_labels = []
        true_labels = []
        total_loss = 0
        self.model.train()
        for batch in tqdm(dataloader, total=len(dataloader)):
            true_labels += batch['labels'].numpy().flatten().tolist()
            batch = {k:v.type(torch.long).to(self.device) for k,v in batch.items()}
            self.model.zero_grad()
            outputs = self.model(**batch)
            loss, logits = outputs[:2]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            logits = logits.detach().cpu().numpy()
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()
        avg_epoch_loss = total_loss / len(dataloader)
        return true_labels, predictions_labels, avg_epoch_loss