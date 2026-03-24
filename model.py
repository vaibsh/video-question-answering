import torch
import torch.nn as nn
import clip

class VideoQAModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=2):
        super().__init__()

        self.clip_model, _ = clip.load("ViT-B/32")
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size)

    def encode_video(self, frames):
        B, T, C, H, W = frames.shape
        frames = frames.view(B*T, C, H, W)
        with torch.no_grad():
            feats = self.clip_model.encode_image(frames)
        return feats.view(B, T, -1)

    def encode_text(self, questions):
        tokens = clip.tokenize(questions).to(next(self.parameters()).device)
        with torch.no_grad():
            text_feats = self.clip_model.encode_text(tokens)
        return text_feats.unsqueeze(1)

    def forward(self, frames, questions, answer_tokens):
        video_feats = self.encode_video(frames)
        text_feats = self.encode_text(questions)

        memory, _ = self.attention(text_feats, video_feats, video_feats)

        tgt = self.token_embed(answer_tokens)
        tgt = tgt.permute(1,0,2)
        memory = memory.permute(1,0,2)

        out = self.decoder(tgt, memory)
        out = out.permute(1,0,2)

        logits = self.output_head(out)
        return logits

    def generate(self, frames, questions, tokenizer, max_len=20, temperature=0.7):
        self.eval()
        with torch.no_grad():
            video_feats = self.encode_video(frames)
            text_feats = self.encode_text(questions)

            memory, _ = self.attention(text_feats, video_feats, video_feats)

            ys = torch.ones(frames.size(0), 1).long().to(frames.device)  # <SOS>

            eos_id = tokenizer.eos_token_id

            for _ in range(max_len):
                tgt = self.token_embed(ys).permute(1, 0, 2)
                out = self.decoder(tgt, memory.permute(1, 0, 2))
                logits = self.output_head(out[-1])  # (batch, vocab)

                # Temperature scaling
                logits = logits / temperature

                # Repetition penalty (penalize last generated token)
                last_tokens = ys[:, -1]  # (batch,)
                logits[torch.arange(logits.size(0)), last_tokens] -= 1.0

                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)

                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)

                ys = torch.cat([ys, next_token], dim=1)

                # Stop if EOS generated
                if (next_token == eos_id).all():
                    break

            return ys