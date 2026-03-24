import torch

def train(model, dataloader, val_loader,optimizer, criterion, device, pad_token_id, epochs=5):
    
    # ---- TRAIN ----
    model.train()

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}")
        
        total_loss = 0
        for frames, questions, answers in dataloader:
            frames = frames.to(device)
            answers = answers.to(device)

            input_tokens = answers[:, :-1]
            target_tokens = answers[:, 1:]
            target_tokens[target_tokens == pad_token_id] = -100
            logits = model(frames, list(questions), input_tokens)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(dataloader)

        # ---- VALIDATION ----
        val_loss = evaluate(model, val_loader, criterion, device, pad_token_id)
        print(f"Train Loss:, {train_loss:.4f}")
        print(f"Validation Loss:, {val_loss:.4f}")

    # Save final checkpoint
    torch.save(model.state_dict(), f"models/model_video-q-a.pt")

def evaluate(model, dataloader, criterion, device, pad_token_id):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for frames, questions, answers in dataloader:
            frames = frames.to(device)
            answers = answers.to(device)

            input_tokens = answers[:, :-1]
            target_tokens = answers[:, 1:]

            # mask padding
            target_tokens[target_tokens == pad_token_id] = -100

            logits = model(frames, list(questions), input_tokens)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_tokens.reshape(-1)
            )

            total_loss += loss.item()

    return total_loss / len(dataloader)