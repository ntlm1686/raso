'''
 * The Recognize Any Surgical Object (RASO)
 * which is based on the Inference of RAM and Tag2Text Models written by Xinyu Huang
'''
import torch

def inference_ram(image, model, return_logits=False):

    with torch.no_grad():
        tags, logits = model.generate_tag(image, return_logits=return_logits)

    return tags[0], logits


def inference_ram_openset(image, model):

    with torch.no_grad():
        tags = model.generate_tag_openset(image)

    return tags[0]
