AI Explorer with Mistral Integration
Overview
AI Explorer integrates cutting-edge AI models, allowing users to seamlessly switch between OpenAI and Mistral for text-based tasks. This project provides a customizable UI, robust backend integration, and performance comparisons to aid developers and researchers.

Features
Multi-Model Support: Use OpenAI or Mistral for prompt responses.
Dynamic Routing: Backend routes requests based on user selection.
Streamlined UI: Select models, submit prompts, and view results in an intuitive interface.
Quick Start
Prerequisites
Python 3.8 or higher.
Mistral API key from Mistral AI.
Setup
Clone the repository:

bash
Copy code
git clone https://github.com/your-repo/ai-explorer.git
cd ai-explorer
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Add the API key:

bash
Copy code
export MISTRAL_API_KEY="your_api_key"
Run the application:

bash
Copy code
streamlit run ui/ui.py
Usage
Select a Model:
Choose between OpenAI or Mistral in the dropdown menu.

Enter a Prompt:
Type your prompt in the text area.

Generate Response:
Click the "Generate Response" button to get the output.

Integration Details
Backend:

mistral_handler.py handles API requests to Mistral.
main.py dynamically routes requests based on model selection.
Frontend:

ui.py provides model selection and response visualization.
Performance Comparison
Model	Latency (ms)	Cost/Token (USD)	Response Quality
Mistral	~120	$0.002	High (context-focused)
OpenAI GPT	~140	$0.03	Very High
Demonstration
Watch the integration in action: YouTube Demo.

Contributions
We welcome your feedback! Submit issues or pull requests to help improve the project.

Unit Tests
Add unit tests to validate the functionality of MistralHandler and API routing:

python
Copy code
# tests/test_mistral_handler.py
import unittest
from explorer.mistral_handler import MistralHandler

class TestMistralHandler(unittest.TestCase):
    def setUp(self):
        self.handler = MistralHandler(api_key="dummy_key")

    def test_generate_response(self):
        with self.assertRaises(Exception):
            self.handler.generate_response("Hello, world!")

if __name__ == "__main__":
    unittest.main()
Logging for Debugging
Add logging to mistral_handler.py for better monitoring:

python
Copy code
import logging

class MistralHandler:
    def __init__(self, api_key, model_version="mistral-7B-v1"):
        self.client = MistralClient(api_key)
        self.model_version = model_version
        logging.basicConfig(level=logging.INFO)

    def generate_response(self, prompt, temperature=0.7, max_tokens=512, top_p=0.9):
        try:
            logging.info(f"Sending request to Mistral with prompt: {prompt}")
            response = self.client.generate(
                model=self.model_version,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )
            logging.info(f"Received response: {response}")
            return response["choices"][0]["text"].strip()
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            raise
