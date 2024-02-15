from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the fine-tuned model and tokenizer
model_path = "./gpt2_finetuned"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Ensure the model is in evaluation mode
model.eval()

# Prepare the input text
input_text = "answer: In computer programming, a module is a software component or part of a program that contains one or more routines."

# Tokenize the input text with added prompt for the model to generate a question
encoded_input = tokenizer.encode(input_text + " question:", return_tensors="pt")

# Generate predictions with adjusted parameters
output = model.generate(
    encoded_input,
    max_length=40,  # Adjusted for more focused output
    num_return_sequences=1,
    temperature=1,  # Increased for more diversity
    top_k=30,  # Adjusted for more focused sampling
    top_p=0.9,  # Slightly adjusted for balance
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
    no_repeat_ngram_size=1,  # Adjusted for diversity
    num_beams=10,  # Enabled beam search for quality improvement
    early_stopping=True  # Enabled early stopping
)

# Decode the generated tokens to get the question
generated_question = tokenizer.decode(output[0], skip_special_tokens=True)
generated_question = generated_question.split('<eos>')[0]
print("Generated Question:", generated_question)