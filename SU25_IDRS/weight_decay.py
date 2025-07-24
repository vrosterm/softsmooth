# Training function
def epoch_weight_decay(loader, model, opt=None, lam=0.1, L=1):
    total_err, total_loss, n = 0, 0.0, 0
    model.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        yp = model(X)
        loss = nn.CrossEntropyLoss()(yp, y)

        # Spectral norm regularization
        spec_reg = 0.0
        for layer in model:
            if isinstance(layer, nn.Linear):
                spec_norm = torch.linalg.svdvals(layer.weight)[0]  # Largest singular value
                spec_reg += spec_norm

        loss = loss + lam * spec_reg

        if opt:
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Total number of linear layers
    num_linear_layers = sum(1 for layer in model if isinstance(layer, nn.Linear))

    # Normalize each layer’s weight
    normalized_weights = []
    for layer in model:
        if isinstance(layer, nn.Linear):
            weight = layer.weight.data
            spec_norm = torch.linalg.svdvals(weight)[0]

            if spec_norm > 0:
                scaling = (L ** (1 / num_linear_layers))
                norm_weight = scaling * weight / spec_norm
            else:
                norm_weight = weight.clone()

            normalized_weights.append(norm_weight.clone())

        total_err += (yp.max(dim=1)[1] != y).sum().item()
        total_loss += loss.item() * X.shape[0]
        n += X.shape[0]

    return total_err / n, total_loss / n
