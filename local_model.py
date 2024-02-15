from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

# Load the dataset from Hugging Face
dataset = load_dataset("domenicrosati/QA2D")

# Limit the dataset to the first 100 rows and remove unwanted columns
limited_dataset = {split: dataset[split].select(range(400)).remove_columns(['example_uid', 'rule-based', 'answer' ]) for split in dataset.keys()}

# Preprocess the dataset to use turker_answer as input and question as output
def preprocess_function(examples):
    return {
        "text": [
            f"{turker_answer} <eos> {question}"
            for turker_answer, question in zip(examples["turker_answer"], examples["question"])
        ]
    }

# Apply preprocessing
processed_dataset = {split: limited_dataset[split].map(preprocess_function, batched=True) for split in limited_dataset.keys()}

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Set PAD token to avoid warnings
model.config.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = {split: processed_dataset[split].map(tokenize_function, batched=True) for split in processed_dataset.keys()}
for split in tokenized_datasets.keys():
    tokenized_datasets[split].set_format(type='torch', columns=['input_ids', 'attention_mask'])

# Data collator to dynamically pad the batched samples
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./gpt2_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=4,  # Adjust epochs according to your dataset size and desired training duration
    per_device_train_batch_size=4,  # Adjust batch size according to your hardware capabilities
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    # eval_dataset=tokenized_datasets.get("validation"),  # Optional: if your dataset has a validation split
)

# Start training
trainer.train()

# Save the fine-tuned model and tokenizer
model.save_pretrained("./gpt2_finetuned")
tokenizer.save_pretrained("./gpt2_finetuned")
