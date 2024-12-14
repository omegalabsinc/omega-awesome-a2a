# In ai-explorer/models/__init__.py
from .gpt2_medium.model import GPT2MediumModel

# Register the model
AVAILABLE_MODELS = {
    "gpt2-medium": GPT2MediumModel
}
