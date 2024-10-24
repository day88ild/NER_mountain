import torch
import os
from transformers import RobertaTokenizerFast
import peft
from colorama import Fore, Back, Style, init
import argparse
import warnings

# Ignore any warnings to reduce console clutter.
warnings.filterwarnings("ignore")

# The greeting text that will be displayed to the user when the program starts.
GREETING_TEXT = """
Hello!! Here you can interact with the NER model (Fine-Tuned RoBERTa)!!!!
To make a query just enter some sentence and the model will show you what words are Mountains and what are not.
(The mountains will be colored in orange)

Below are some EXAMPLES if you cannot come up with one:

    * "So how it was on Kilimanjaro?"

    * "White Glacier is a broad westward flowing tributary glacier which joins the Land Glacier on the north side of Mount McCoy in Marie Byrd Land."

    * "Other notable sections of the cemetery are the cemetery of the Finnish Guard, the Artist's Hill and the Statesmen's Grove."

    * "Why don't we hang out together? Let's go on a trip. What about Alpas?"
"""

# Define the maximum length for the tokenized sentence.
MAX_LENGTH = 32

def inference_ner(path_to_model, sentence):

    # Initialize the tokenizer for RoBERTa using the pre-trained 'roberta-base' version.
    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
    
    # Tokenize the user's input sentence.
    # - Truncate the sentence if it exceeds the maximum length.
    # - Pad the sentence to the maximum length.
    # - Return tensors for use with PyTorch ('pt').
    # - Include attention masks to differentiate padding from actual data.
    tokenized_sentence = tokenizer(sentence, 
                                   truncation=True,
                                   padding="max_length",
                                   max_length=MAX_LENGTH,
                                   return_tensors="pt",
                                   return_attention_mask=True)
    
    # Load the fine-tuned model from the specified path.
    model = torch.load(path_to_model)
    # Set the model to evaluation mode, disabling layers like dropout.
    model.eval()
    
    # Perform inference without calculating gradients (for efficiency).
    with torch.inference_mode():
        # Pass the tokenized input to the model, including attention masks.
        outputs = model(input_ids=tokenized_sentence["input_ids"],
                        attention_mask=tokenized_sentence["attention_mask"])

    # Apply a softmax to get probabilities and find the most likely class (argmax) for each token.
    outputs = torch.softmax(outputs.logits, dim=2).argmax(dim=2)

    # Convert outputs to a list of labels and tokenized sentence to numpy arrays for processing.
    outputs = list(outputs.squeeze().cpu().numpy())
    tokenized_sentence = list(tokenized_sentence["input_ids"].squeeze().cpu().numpy())
    
    # Print a message indicating that the results are about to be displayed.
    print(Fore.GREEN + "\nHERE IS THE RESULT:\n")
    
    # Iterate through each token and corresponding label.
    for i in range(len(outputs)):
        label = outputs[i]  # The predicted label for the current token.
        token = tokenized_sentence[i]  # The token ID from the tokenized input.
        
        # Skip special tokens like [CLS], [SEP], or padding.
        if token in [0, 1, 2]:
            continue
            
        # Decode the token ID back to a readable word.
        decoded_word = tokenizer.decode(token)
        
        # If the label indicates a mountain entity, print the word in a different color.
        if label:
            print(Fore.LIGHTYELLOW_EX + decoded_word, end=" ")
        else:
            # Otherwise, print the word in the default style.
            print(Style.RESET_ALL + decoded_word, end=" ")
    
    # Print a newline for better formatting after the output.
    print()

# If the script is being run directly, execute the inference function.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NER model inference.")
    parser.add_argument("--model_path", type=str, help="Path to the model file.")
    parser.add_argument("--sentence", type=str, help="The sentence to be processed by the model.")
    
    args = parser.parse_args()
    
    # Print the greeting message to introduce the program.
    print(GREETING_TEXT)

    # If arguments are not provided, prompt the user for input
    model_path = args.model_path or input("Enter the path to the model file: ")
    sentence = args.sentence or input("Enter the sentence you want to pass into the model: ")
    
    
    inference_ner(model_path, sentence)

