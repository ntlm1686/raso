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

### Basic Inference

```python
import torch
from PIL import Image
from raso.models import raso
from raso import inference_ram, get_transform

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

# Inference
results, logits = inference_ram(image, model, return_logits=True)
print("Detected objects:", results)
```

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