# RASO: Recognize Any Surgical Object

RASO (Recognize Any Surgical Object) is a vision-language model for recognizing and detecting surgical instruments and objects in surgical images and videos.

## Installation

```bash
# Clone the repository
git clone https://github.com/ntlm1686/raso.git
cd recognize-any-surgical-object

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Model Weights

The pre-trained model weights need to be downloaded from Hugging Face:
https://huggingface.co/Mumon/raso

Download the model weights and place them in the `MODEL` directory:
- `MODEL/raso_zeroshot.pth`: Zero-shot recognition model
- `MODEL/raos_cholect50_ft.pth`: Model fine-tuned on the Cholec50 dataset

## Usage

### Closed-Set Inference

Use the standard `inference` helper to obtain closed-set predictions from the pretrained model.

```python
import torch
from PIL import Image
from raso.models import raso
from raso import inference, get_transform

# Load model
model = raso(pretrained='./MODEL/raso_zeroshot.pth',
             image_size=384,
             vit='swin_l')
model.eval()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
transform = get_transform(image_size=384)

# Load and preprocess image
image_path = "./examples/img_01.png"
image_pil = Image.open(image_path)
image = transform(image_pil).unsqueeze(0).to(device)

tags, logits = inference(image, model) 
print("Results with default threshold (0.65):", tags)
```

### Open-Set Inference

You can extend RASO to recognize custom vocabulary at inference time by pairing it with a CLIP text encoder and calling `inference_openset`. This keeps the closed-set predictions while adding scores for any extra tags you provide.

```python
from transformers import CLIPModel, CLIPProcessor

# Load the CLIP text encoder once (reuse across images)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Closed-set predictions from RASO
tags_closed, _ = inference(image, model)
print("Closed-set tags:", tags_closed)

# Add new vocabulary for open-set inference
extra_tags = ["hemostat", "laparoscopic grasper", "trocar 5mm", "new tag 1", "new tag 2"]

tags_open, open_logits, full_tags = inference_openset(
    image=image,
    raso_model=model,
    clip_model=clip_model,
    clip_tokenizer=clip_proc.tokenizer,
    extra_tags=extra_tags,
    threshold=0.68,  # adjust per your precision/recall needs
    return_tags=True,  # return the merged closed- and open-set tags
)

print("Open-set tags:", tags_open)
print("Number of open-set logits:", open_logits.shape)
```

`threshold` controls how confident the model must be to surface a new tag (default `0.68`). If you only need the open-set logits, drop `return_tags=True`.


## Citation

If you use RASO in your research, please cite the following papers:

```
@misc{li2025recognizesurgicalobjectunleashing,
      title={Recognize Any Surgical Object: Unleashing the Power of Weakly-Supervised Data}, 
      author={Jiajie Li and Brian R Quaranto and Chenhui Xu and Ishan Mishra and Ruiyang Qin and Dancheng Liu and Peter C W Kim and Jinjun Xiong},
      year={2025},
      eprint={2501.15326},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.15326}, 
}
```

## License

This project is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0) - see the LICENSE file for details.

## Acknowledgments

This project builds upon the [Recognize Anything](https://github.com/xinyu1205/recognize-anything) repository. We acknowledge and thank the authors for their foundational work on the Recognize Anything Model (RAM) architecture that made RASO possible.
