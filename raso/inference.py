'''
 * The Recognize Any Surgical Object (RASO)
'''
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple


def inference(image, model, return_logits=False):

    with torch.no_grad():
        tags, logits = model.generate_tag(image, return_logits=return_logits)

    return tags[0], logits


@torch.no_grad()
def open_set_infer(
    image: torch.Tensor,                 # [B,3,H,W]，预处理与 raso 训练时一致
    raso,                                # 已实例化的 RASO 模型
    clip_model,                          # transformers.CLIPModel 实例（只用 text_model）
    clip_tokenizer,                      # 来自 CLIPProcessor.tokenizer
    extra_tags: Optional[List[str]] = None,
    default_threshold: Optional[float] = None,
    return_logits: bool = False,
) -> Tuple[List[str], Optional[torch.Tensor]]:
    """
    返回:
      - tags_str_per_image: List[str]，每张图的预测标签，用 ' | ' 连接；若无则为空字符串
      - logits_all (可选): [B, N_total]，拼接后的全部标签 logits（未 sigmoid）
    说明:
      - 不会修改 raso 的任何属性；所有扩展只在本函数内临时生效。
    """
    device = next(raso.parameters()).device
    image = image.to(device)

    # 1) 视觉编码 → 512 维
    image_embeds = raso.visual_encoder(image)      # [B,P,vision_width]
    image_embeds = raso.image_proj(image_embeds)   # [B,P,512]
    image_atts   = torch.ones(image_embeds.size()[:-1],
                               dtype=torch.long, device=device)

    # 2) 组装标签嵌入：原有 + 额外（CLIP 编码）
    base_tags = list(map(str, raso.tag_list.tolist()))
    label_embed_eff = raso.label_embed.data.to(device)      # [N,512]
    thr_eff = raso.class_threshold.to(device).float()       # [N]
    if default_threshold is None:
        default_threshold = float(raso.threshold)

    if extra_tags:
        # 去重（避免与已有标签重复）
        existed = set(base_tags)
        new_tags = [t for t in extra_tags if t not in existed]
        if len(new_tags) > 0:
            toks = clip_tokenizer(new_tags, padding=True, truncation=True,
                                  max_length=16, return_tensors="pt").to(device)
            # CLIP 文本编码，取 pooled 输出（ViT-B/32 → 512 维）
            txt = clip_model.text_model(
                input_ids=toks["input_ids"],
                attention_mask=toks["attention_mask"],
                return_dict=True
            ).pooler_output                                    # [M,512]
            txt = F.normalize(txt, dim=-1)                     # 稍稳一些
            label_embed_eff = torch.cat([label_embed_eff, txt], dim=0)  # [N+M,512]
            thr_new = torch.full((len(new_tags),), default_threshold, device=device)
            thr_eff = torch.cat([thr_eff, thr_new], dim=0)
            base_tags.extend(new_tags)

    # 3) 走原有打分链：wordvec_proj → tagging_head(mode='tagging') → fc → sigmoid
    label_embed_eff = F.relu(raso.wordvec_proj(label_embed_eff))        # [N_total, hidden]
    bs = image_embeds.size(0)
    label_embed_rep = label_embed_eff.unsqueeze(0).repeat(bs, 1, 1)     # [B,N_total,hidden]

    tagging_out = raso.tagging_head(
        encoder_embeds=label_embed_rep,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=False,
        mode='tagging',
    )
    logits = raso.fc(tagging_out[0]).squeeze(-1)  # [B, N_total]
    probs  = torch.sigmoid(logits)

    # 4) 阈值筛选并拼回字符串
    mask = (probs > thr_eff.unsqueeze(0))         # [B, N_total] bool
    tags_str_per_image = []
    for b in range(mask.size(0)):
        idx = torch.nonzero(mask[b], as_tuple=False).squeeze(-1).tolist()
        if isinstance(idx, int):
            idx = [idx]
        if not idx:
            tags_str_per_image.append("")
        else:
            tags = [base_tags[i] for i in idx]
            tags_str_per_image.append(" | ".join(tags))

    return tags_str_per_image, (logits if return_logits else None), base_tags


def inference_openset(image, raso_model, clip_model, clip_tokenizer,
                       extra_tags=None, threshold=None, return_logits=True, return_tags=False):
    """
    image: 已经 transform 并 to(device) 的张量 [B,3,H,W]
    raso_model: 已经实例化的 RASO 模型
    clip_model: CLIPModel
    clip_tokenizer: clip_processor.tokenizer
    extra_tags: list[str] 额外标签
    threshold: float，给新标签用的阈值（不传则用 raso_model.threshold）
    """
    tags, logits, base_tags = open_set_infer(
        image=image,
        raso=raso_model,
        clip_model=clip_model,
        clip_tokenizer=clip_tokenizer,
        extra_tags=extra_tags,
        default_threshold=threshold,
        return_logits=return_logits
    )
    if return_tags:
        return tags, logits, base_tags
    return tags[0], logits
