import torch
import os
from transformers import RobertaTokenizer
import peft
from colorama import Fore, Back, Style, init
import warnings
warnings.filterwarnings("ignore")


GREETING_TEXT = """
Hello!! Here you can interact with the NER model (Fine-Tuned RoBERTa)!!!!
To make a query just enter some sentence and the model will show you what words are Mountains and what are not.
(The mountains will be colored in orange)

Below are some EXAMPLES if you cannot come up with one:

    * "So how it was on Kilimanjaro?"

    * "White Glacier is a broad westward flowing tributary glacier which joins the Land Glacier on the north side of Mount McCoy in Marie Byrd Land ."

    * "Other notable sections of the cemetery are the cemetery of the Finnish Guard , the Artist 's Hill and the Statesmen 's Grove ."
    
    * "Why dont we hang out together? Lets go on a trip. What about Alpas?"
"""
MAX_LENGTH = 32

def inference_ner():
    print(GREETING_TEXT)
    
    path_to_model = input("Enter path to the model (recommended to use RoBERTa version):")
    sentence = input("Enter the sentence you want to pass into the model:")
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenized_sentence = tokenizer(sentence, 
                                   truncation=True,
                                   padding="max_length",
                                   max_length=MAX_LENGTH,
                                   return_tensors="pt",
                                   return_attention_mask=True)
    
    model = torch.load(path_to_model)
    model.eval()
    
    with torch.inference_mode():
        outputs = model(input_ids=tokenized_sentence["input_ids"],
                        attention_mask=tokenized_sentence["attention_mask"])

    outputs = torch.softmax(outputs.logits, dim=2).argmax(dim=2)

    
        
    outputs = list(outputs.squeeze().cpu().numpy())
    tokenized_sentence = list(tokenized_sentence["input_ids"].squeeze().cpu().numpy())
    
    print(Fore.GREEN + "\nHERE IS THE RESULT:\n")
    
    for i in range(len(outputs)):
        label = outputs[i]
        token = tokenized_sentence[i]
        if token in [0, 1, 2]:
            continue
            
        decoded_word = tokenizer.decode(token)
        if label:
            print(Fore.LIGHTYELLOW_EX + decoded_word, end=" ")
        else:
            print(Style.RESET_ALL + decoded_word, end=" ")    
    

if __name__ == "__main__":
    inference_ner()
