from mistralai import MistralClient  

class MistralHandler:  
    def __init__(self, api_key):  
        self.client = MistralClient(api_key)  

    def generate_response(self, prompt, temperature=0.7, max_tokens=512, top_p=0.9):  
        try:  
            return self.client.generate(prompt=prompt, temperature=temperature, max_tokens=max_tokens, top_p=top_p)  
        except Exception as e:  
            raise RuntimeError(f"Error connecting to Mistral API: {e}")  
