# Lokr-Implementation
This repository is the implementation of a PEFT technique LoKr

# What is Lokr

LoKr shares similarities with LoRA and LoHa, primarily utilized in diffusion models, although it can be applied to other types of models as well. In LoKr, the conventional matrix product is replaced by the Kronecker product. This decomposition method generates a block matrix that retains the rank of the original weight matrix. Moreover, leveraging the Kronecker product allows for vectorization by stacking the matrix columns, which can enhance efficiency by circumventing the need for fully reconstructing ∆W.

## Model 
google/vit-base-patch16-224-in21k: The Vision Transformer (ViT) model has been pre-trained on ImageNet-21k, comprising 14 million images across 21,843 classes, with a resolution of 224x224. This model was initially presented in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al., and its first release was made available in a specific repository

## Dataset

food101: This dataset consists of 101 food categories, with 101'000 images. For each class, 250 manually reviewed test images are provided as well as 750 training images. On purpose, the training images were not cleaned, and thus still contain some amount of noise. This comes mostly in the form of intense colors and sometimes wrong labels. All images were rescaled to have a maximum side length of 512 pixels.

## Libraries Used

- peft: for model pruning and quantization
- transformers: transformers: For utilizing and fine-tuning the model.
- datasets: For handling and processing the data.
- numpy: For numerical computations.
- torch: For building and training neural networks.
- torchvision: used in PyTorch-based projects for computer vision tasks due to its comprehensive collection of utilities and pre-trained models, as well as its seamless integration with PyTorch.

## Hyperparameters 

Hyperparameters
learning_rate=5e-3
num_train_epochs=5
batch_size = 128
rank = 16

## Training 

| Epoch | Training Loss | Validation Loss |
|-------|---------------|-----------------|
| 1     | 0.826600      | 0.481831        |
| 2     | 0.743100      | 0.428663        |
| 3     | 0.726600      | 0.405388        |
| 4     | 0.656200      | 0.395885        |
| 5     | 0.647500      | 0.389642        |

## Usage

```
from peft import PeftConfig, PeftModel
from transformers import AutoImageProcessor
from PIL import Image
import requests

config = PeftConfig.from_pretrained("likhith231/vit-base-patch16-224-in21k-lokr_model")
model = AutoModelForImageClassification.from_pretrained(
    config.base_model_name_or_path,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,
)
model = PeftModel.from_pretrained(model, "likhith231/vit-base-patch16-224-in21k-lokr_model")

url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/beignets.jpeg"
image = Image.open(requests.get(url, stream=True).raw)

encoding = image_processor(image.convert("RGB"), return_tensors="pt")

with torch.no_grad():
    outputs = model(**encoding)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

```