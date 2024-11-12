# inference.py
import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the tokenizer and model (paths may need adjusting based on your setup)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
checkpoint_path = 'models/final_checkpoint.pth'

# Load your trained model checkpoint
def load_checkpoint(model, file_path):
    checkpoint = torch.load(file_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Checkpoint loaded from {file_path}")

# Load checkpoint
load_checkpoint(model, checkpoint_path)
model.eval()

# Function to generate answers based on input questions
def ask_question(question, object_name=None, max_length=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Prepare the input
    combined_input = f"{object_name} {question}" if object_name else question

    inputs = tokenizer(
        combined_input,
        return_tensors="pt",
        max_length=128,
        truncation=True
    ).to(device)

    # Generate answer
    answer_ids = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        num_beams=4,
        early_stopping=True
    )

    # Decode the generated answer
    answer = tokenizer.decode(answer_ids[0], skip_special_tokens=True)
    return answer
