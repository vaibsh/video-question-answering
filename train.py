import torch

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for frames, input_ids, attention_mask, labels in dataloader:
            frames = frames.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(frames, input_ids, attention_mask, labels)
            total_loss += outputs.loss.item()

    return total_loss / len(dataloader)

def train(model, train_loader, val_loader, optimizer, device, num_epochs=5):
    # ✅ NEW: GradScaler for AMP
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for frames, input_ids, attention_mask, labels in train_loader:
            frames = frames.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            # ✅ NEW: autocast for mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(frames, input_ids, attention_mask, labels)
                loss = outputs.loss

            optimizer.zero_grad()

            # ✅ NEW: scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, val_loader, device) if val_loader else None

        if val_loss is not None:
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f}")