import torch
import torch.nn as nn
import clip
from transformers import AutoModelForCausalLM
import torch.nn.functional as F

class VideoQAModel(nn.Module):
    def __init__(self):
        super().__init__()

        # CLIP encoder
        self.clip_model, _ = clip.load("ViT-B/32")
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # GPT-2 decoder
        self.decoder = AutoModelForCausalLM.from_pretrained("gpt2")

        # Project CLIP → GPT hidden size
        self.video_proj = nn.Linear(512, 768)

    def encode_video(self, frames):
        B, T, C, H, W = frames.shape
        frames = frames.view(B*T, C, H, W)

        with torch.no_grad():
            feats = self.clip_model.encode_image(frames)

        feats = feats.view(B, T, -1)

        # Mean pool frames → 1 token
        feats = feats.mean(dim=1, keepdim=True)  # (B, 1, 512)

        return feats

    def forward(self, frames, input_ids, attention_mask):
        video_feats = self.encode_video(frames)          # (B, 1, 512)
        video_feats = self.video_proj(video_feats)       # (B, 1, 768)

        text_embeds = self.decoder.transformer.wte(input_ids)

        inputs_embeds = torch.cat([video_feats, text_embeds], dim=1)

        prefix_mask = torch.ones((attention_mask.size(0), 1)).to(attention_mask.device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False
        )

        logits = outputs.logits  # (B, T+1, V)

        # Manual shift (CORRECT WAY)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids.contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.decoder.config.pad_token_id
        )

        return type("Output", (), {"loss": loss})

    def generate(self, frames, input_ids, attention_mask, max_new_tokens=20):
        video_feats = self.encode_video(frames)
        video_feats = self.video_proj(video_feats)

        text_embeds = self.decoder.transformer.wte(input_ids)
        inputs_embeds = torch.cat([video_feats, text_embeds], dim=1)

        prefix_mask = torch.ones((attention_mask.size(0), 1)).to(attention_mask.device)
        attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        outputs = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=self.decoder.config.pad_token_id,
            eos_token_id=self.decoder.config.eos_token_id
        )

        return outputs