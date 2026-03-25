import torch
import torch.nn as nn
import clip
from transformers import AutoModelForCausalLM
import torch.nn.functional as F

class VideoQAModel(nn.Module):
    def __init__(self):
        super().__init__()

        # CLIP encoder (frozen)
        self.clip_model, _ = clip.load("ViT-B/32")
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # GPT-2 decoder
        self.decoder = AutoModelForCausalLM.from_pretrained("gpt2")

        # Projection
        self.video_proj = nn.Linear(512, 768)

    def encode_video(self, frames):
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)

        with torch.no_grad():
            feats = self.clip_model.encode_image(frames)

        feats = feats.view(B, T, -1)
        feats = feats.mean(dim=1)

        return feats

    def forward(self, frames, input_ids, attention_mask, labels):
        video_feats = self.encode_video(frames)
        video_feats = video_feats.float()
        video_feats = self.video_proj(video_feats)
        video_feats = video_feats.unsqueeze(1)
        text_embeds = self.decoder.transformer.wte(input_ids)

        inputs_embeds = torch.cat([video_feats, text_embeds], dim=1)

        # Attention mask
        video_mask = torch.ones((attention_mask.size(0), video_feats.size(1)), device=attention_mask.device)
        attention_mask = torch.cat([video_mask, attention_mask], dim=1)

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask
        )

        logits = outputs.logits

        # ---- KEY FIX ----
        # Remove video token logits before computing loss
        logits_text = logits[:, video_feats.size(1):, :]  # align with input_ids

        # Shift within text sequence only
        shift_logits = logits_text[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        return type("Output", (), {"loss": loss})

    def generate(self, frames, input_ids, attention_mask, max_new_tokens=20):
        video_feats = self.encode_video(frames)
        video_feats = self.video_proj(video_feats)

        text_embeds = self.decoder.transformer.wte(input_ids)

        inputs_embeds = torch.cat([video_feats, text_embeds], dim=1)

        batch_size = attention_mask.size(0)
        num_video_tokens = video_feats.size(1)

        video_mask = torch.ones((batch_size, num_video_tokens), device=attention_mask.device)
        attention_mask = torch.cat([video_mask, attention_mask], dim=1)

        outputs = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.2,
            pad_token_id=self.decoder.config.pad_token_id,
            eos_token_id=self.decoder.config.eos_token_id,
            early_stopping=True
        )

        return outputs