# Dynamic post training quantization
import os
import torch
import time
import evaluate
from transformers.models.bert import BertForSequenceClassification
from distillation import get_data

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def eval_with_time(model, eval_dl):
    accuracy = 0    
    st = time.time()
    model.eval()
    for batch in eval_dl:
        input_ids = batch[0]["input_ids"]
        input_ids = input_ids.squeeze(1)
        attn_mask = batch[0]["attention_mask"]
        labels = batch[1]
        with torch.inference_mode():
            preds = model(input_ids, attn_mask).logits
        accuracy +=(preds.argmax(dim=-1)==labels).float().mean().item()
        # break #for one batch only
    et = time.time()
    print(f"Time taken for one batch: , {et-st: .2f} seconds")
    print(f"Accuracy: {accuracy/len(eval_dl):.2f}")

model = BertForSequenceClassification.from_pretrained("milapp857/bert-finetuned-txt-classification")
model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

print("Size of model before quantization")
print_size_of_model(model)

print("Size of model after quantization")
print_size_of_model(quantized_model)

train_dl, eval_dl, dummy = get_data()

print("Evaluating model before quantization")
eval_with_time(model, eval_dl)

print("Evaluating model after quantization")
eval_with_time(quantized_model, eval_dl)