import os
import numpy as np 
import pandas as pd 
import evaluate
import pyarrow
from typing import Dict, Tuple, List, Any
from transformers.models.bert import BertTokenizerFast, BertForSequenceClassification
from transformers.trainer import Trainer
from transformers.training_args import TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

os.environ["WANDB_DISABLED"] = "True"

def tokenize(sample):
    return tokenizer(sample["text"], padding='max_length',truncation=True, return_tensors='pt')

def compute_metric(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    return metric.compute(predictions=preds, references=labels)

def finetune() -> None:
    #Read the data and change column labels for huggingface's trainer
    df: pd.DataFrame = pd.read_csv("df_file.csv")
    df = df.rename(columns={"Label":"labels", "Text": "text"}) #for huggingface's trainer
    df = df.astype({"labels":int})

    train_data, test_data= train_test_split(df, test_size=0.2, shuffle=True, random_state=42)
    debug_data: pd.DataFrame = train_data.iloc[0:5]

    #Import base pre-trained BERT model from Huggingface
    model_id: str = "bert-base-uncased"

    # Convert your DataFrame to a Hugging Face Dataset
    train_data: Dataset = Dataset.from_pandas(train_data) 
    test_data: Dataset = Dataset.from_pandas(test_data)
    debug_data: Dataset = Dataset.from_pandas(debug_data)
    print("Data loaded")

    #Convert text data to tokens 
    global tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(model_id)
    tokenized_train_data: Dataset = train_data.map(tokenize, batched=True, remove_columns=["text"], batch_size=1000, num_proc=4) 
    tokenized_test_data: Dataset = test_data.map(tokenize, batched=True, remove_columns=["text"], batch_size=1000, num_proc=4)

    tokenized_debug_data =debug_data.map(tokenize, batched=True, remove_columns=["text"]) 
    print("Data tokenized")


    global metric
    metric = evaluate.load("accuracy")

    #Fine-tune the model on the dataset
    num_labels = 5
    label2id = {"Politics":0, "Sport":1, "Technology": 2, "Entertainment":3, "Business":4}
    id2label = {0:"Politics", 1:"Sport", 2:"Technology", 3:"Entertainment", 4:"Business"}

    #Initialize model instance
    teacher_model = BertForSequenceClassification.from_pretrained(
        model_id, num_labels=num_labels, label2id=label2id, id2label=id2label
    )
    print("Model loaded")

    #Training arguments
    training_args = TrainingArguments(
        output_dir = "outputs",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        learning_rate=1e-5,
    	num_train_epochs=5,
        logging_strategy="epoch",
        # bf16=True, # bfloat16 training 
    	torch_compile=True, # optimizations
        optim="adamw_torch_fused", # improved optimizer 
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        eval_strategy="epoch",
        report_to=None
    )

    #Trainer setup
    trainer = Trainer(
        model=teacher_model,
        args=training_args,
        # train_dataset=tokenized_debug_data,
        # eval_dataset=tokenized_debug_data,
        train_dataset=tokenized_train_data,
        eval_dataset=tokenized_test_data,
        compute_metrics=compute_metric
    )

    #start training
    print("Starting training")
    trainer.train(  )
    trainer.save_model("files/models/bert_finetuned")

def load_model(model_path: str) -> BertForSequenceClassification:
    """
    Load the fine-tuned model from the specified path.
    """
    return BertForSequenceClassification.from_pretrained(model_path)
if __name__ == "__main__":
    finetune()
