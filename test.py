from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_dir = "./gemma-sql-finetuned"

tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)

model.to("cuda" if torch.cuda.is_available() else "cpu")

def ask_gemma(prompt, max_new_tokens=256, temperature=0.7):
    inputs = tokenizer(
        f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n",
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Trim everything before the model’s answer
    if "<start_of_turn>model" in response:
        response = response.split("<start_of_turn>model")[-1]
    return response.strip()


# Test the fine-tuned model with some example prompts

print(ask_gemma("Explain the difference between INNER JOIN and LEFT JOIN."))
print(ask_gemma("Create a SQL query to list departments with more than 5 employees."))
print(ask_gemma("What is normalization in SQL?"))