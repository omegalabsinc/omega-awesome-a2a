from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time

class GPT2MediumModel:
    def __init__(self):
        self.model_name = "gpt2-medium"
        print(f"Loading {self.model_name}...")
        start_time = time.time()
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    def generate(self, prompt, max_length=100, temperature=0.7):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            no_repeat_ngram_size=3,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
