import torch
import torch.nn as nn
import pandas as pd
from typing import Tuple
from tqdm import tqdm
from distillbert import DistilBERTClassifier
from finetuning import load_model
from torch.utils.data import Dataset, DataLoader, random_split
from transformers.models.bert import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

def get_data():
    #Prepare the dataset

    df = pd.read_csv("df_file.csv")
    df = df.rename(columns={"Label":"labels", "Text": "text"}) #for huggingface's trainer

    class TokenizedData(Dataset):
        def __init__(self, df, tokenizer):
            self.text = df["text"]
            self.labels = df["labels"]
            self.tokenizer = tokenizer

        def __len__(self):
            return len(df["text"])

        def __getitem__(self, idx):
            return (
                self.tokenizer(self.text[idx][:512], padding="max_length", return_tensors="pt", truncation=True),#truncate the text to 512 tokens to fit BERT's input size
                self.labels[idx]
                ) 

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    data = TokenizedData(df, tokenizer)
    dummy_data = TokenizedData(df.iloc[0:5], tokenizer)

    train_data, test_data = random_split(data, lengths=[0.7, 0.3], generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_data, batch_size=16, shuffle=True)
    test_dl = DataLoader(test_data, batch_size=8, shuffle=True)

    return train_dl, test_dl, dummy_data
# print(data[12])

def load_student_teacher_models(teacher_path: str ="milapp857/bert-finetuned-txt-classification", 
                                VOCAB_SIZE: int = 30000, 
                                N_SEGMENTS: int = 2, 
                                MAX_LEN: int = 512, 
                                EMBED_DIM: int = 768, 
                                ATTN_HEADS: int = 12,
                                N_LAYERS: int = 6, 
                                DROPOUT: float = 0.1,
                                NUM_CLASSES: int = 5) -> Tuple[DistilBERTClassifier, BertForSequenceClassification]:

 
    #Prepare the student and teacher models
    teacher_model_path = teacher_path
    student_model = DistilBERTClassifier(
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
        n_segs=N_SEGMENTS,
        embed_dim=EMBED_DIM,
        n_layers=N_LAYERS,
        attn_heads=ATTN_HEADS,
        dropout=DROPOUT,
        num_classes=NUM_CLASSES
    ).to(device)

    teacher_model = load_model(teacher_model_path).to(device)

    teacher_model.eval()
    return (student_model, teacher_model)
global device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def distill() -> None:
    #Load the data
    train_dl, test_dl, dummy_data = get_data()

    #Load the models
    student_model, teacher_model = load_student_teacher_models()

    #Move the models to the device
    student_model.to(device)
    teacher_model.to(device)

    
    #Training loop
    n_epochs = 8
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=1e-5)
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    temperature = 2.0
    alpha = 0.55

    training_loss_list = []
    training_kd_loss_list = []
    training_accuracy_list = []
    test_loss_list = []
    test_accuracy_list = []

    for epoch in range(n_epochs):
        student_model.train()
        train_loss =0.0
        train_kd_loss = 0.0
        train_accuracy = 0.0
        test_loss = 0.0
        test_accuracy = 0.0

        for batch in tqdm(train_dl, total=len(train_dl), desc=f"Training Epoch {epoch+1}"):
            input_ids = batch[0]["input_ids"].to(device)
            input_ids = input_ids.squeeze(1).to(device)
            # print("input_ids shape: ", input_ids.shape)
            attn_mask = batch[0]["attention_mask"].to(device)
            attn_mask = attn_mask.squeeze(1).to(device)
            # print("mask shape: ", attn_mask.shape)
            labels = batch[1].to(device)

            logits_student = student_model(input_ids)
            logits_teacher = teacher_model(input_ids, attn_mask).logits

            # print("logits_student shape: ", logits_student.shape)
            # print("logits_teacher shape: ", logits_teacher.shape)
            loss_kd = kl_loss_fn(
                F.log_softmax(logits_student / temperature, dim=-1),
                F.softmax(logits_teacher / temperature, dim=-1)
                ) * (temperature ** 2)
            loss_ce = F.cross_entropy(logits_student, labels)
            loss = alpha * loss_ce + (1 - alpha) * loss_kd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Update training metrics
            train_loss += loss.item()
            train_kd_loss += loss_kd.item()
            train_accuracy += (logits_student.argmax(dim=-1) == labels).float().mean().item()

        student_model.eval()
        for batch in tqdm(test_dl, total=len(test_dl), desc=f"Testing Epoch {epoch+1}"):
            input_ids = batch[0]["input_ids"].to(device)
            attn_mask = batch[0]["attention_mask"].to(device)
            labels = batch[1].to(device)

            with torch.no_grad():
                logits_student = student_model(input_ids)
                loss = F.cross_entropy(logits_student, labels)

            test_loss += loss.item()
            test_accuracy += (logits_student.argmax(dim=-1) == labels).float().mean().item()

        #Calculate average metrics
        train_loss /= len(train_dl)
        train_kd_loss /= len(train_dl)
        train_accuracy /= len(train_dl)
        test_loss /= len(test_dl)
        test_accuracy /= len(test_dl)

        training_loss_list.append(train_loss)
        training_kd_loss_list.append(train_kd_loss)
        training_accuracy_list.append(train_accuracy)
        test_loss_list.append(test_loss)
        test_accuracy_list.append(test_accuracy)

        print(f"Epoch {epoch+1}/{n_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Train KD Loss: {train_kd_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.4f}, "
              f"Test Loss: {test_loss:.4f}, "
              f"Test Accuracy: {test_accuracy:.4f}")

    #Save the student model
    student_model_path = "files/models/student_model.pt"
    torch.save(student_model.state_dict(), student_model_path)
    print(f"Student model saved to {student_model_path}")

if __name__ == "__main__":
    distill()
