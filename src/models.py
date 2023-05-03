
import torchvision
from torchvision.models import list_models

# List available models
all_models = list_models()
classification_models = list_models(module=torchvision.models)

print(f"Total number of available models: {len(all_models)}")
print(f"Total number of classification models: {len(classification_models)}")

for model in classification_models:
    print(model)

for model in all_models:
    print(model)