import math
from contextlib import contextmanager

import torch


def _as_sigma_list(sigma, n_layers=2):
    if isinstance(sigma, torch.Tensor):
        sigma = sigma.detach().cpu().tolist()
    if isinstance(sigma, (int, float)):
        return [float(sigma)] * n_layers
    sigma = [float(s) for s in sigma]
    if len(sigma) == 1:
        return sigma * n_layers
    if len(sigma) != n_layers:
        raise ValueError(f"Expected {n_layers} sigma values, got {len(sigma)}")
    return sigma


def _norm_ord(p):
    if isinstance(p, str) and p.lower() in {"inf", "infty", "infinity"}:
        return float("inf")
    return float(p)


def _norm_label(p):
    p = _norm_ord(p)
    return "inf" if math.isinf(p) else str(int(p) if p.is_integer() else p)


def _dual_norm_ord(p):
    p = _norm_ord(p)
    if p == 1:
        return float("inf")
    if math.isinf(p):
        return 1.0
    return p / (p - 1)


def _torch_norm_ord(p):
    p = _norm_ord(p)
    return float("inf") if math.isinf(p) else p


def _vector_norm(x, p):
    return torch.linalg.vector_norm(x, ord=_torch_norm_ord(p))


def _induced_matrix_norm(matrix, p):
    """Section 2.1 notation: ||A||_p is the l_p-induced matrix norm."""
    p = _norm_ord(p)
    if p not in (1.0, 2.0, float("inf")):
        raise ValueError("Only induced matrix norms p=1, p=2, and p=inf are supported")
    return torch.linalg.matrix_norm(matrix, ord=_torch_norm_ord(p))


def _linear_layers(model):
    if hasattr(model, "linear_layers"):
        return model.linear_layers()
    if not hasattr(model, "fc3"):
        return [model.fc0, model.fc1, model.fc2]
    return [model.fc0, model.fc1, model.fc2, model.fc3]


def _regu_layers(model):
    if hasattr(model, "regu_layers"):
        return model.regu_layers()
    if not hasattr(model, "fc3"):
        return [model.fc0, model.fc1]
    return [model.fc0, model.fc1, model.fc2]


def _output_layer(model):
    if hasattr(model, "output_layer"):
        return model.output_layer()
    return model.fc3 if hasattr(model, "fc3") else model.fc2


def _row_noise_std(layer, sigma_l):
    """sqrt(diag(sigma_l^2 W W^T)); row L2 comes from the covariance diagonal."""
    return float(sigma_l) * torch.linalg.vector_norm(layer.weight, ord=2, dim=1)


def _row_norm_factor(layer, p):
    q = _dual_norm_ord(p)
    row_norms = torch.linalg.vector_norm(layer.weight, ord=_torch_norm_ord(q), dim=1)
    return _vector_norm(row_norms, p)


def _margin_from_scores(scores):
    best_scores = torch.topk(scores, 2)
    return best_scores.values[0] - best_scores.values[1], best_scores


def _clamp_prob(p, eps=1e-6):
    return torch.clamp(p, min=eps, max=1 - eps)


def phi_inverse(x, mu=0.0):
    x = torch.as_tensor(x)
    return mu + torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype)) * torch.erfinv(2 * x - 1)


@contextmanager
def _temporary_sigma(model, sigma):
    if not hasattr(model, "set_sigma"):
        yield
        return

    state = (
        list(getattr(model, "base_sigma", [])),
        list(getattr(model, "sigma", [])),
        list(getattr(model, "z", [])),
    )
    model.set_sigma(sigma)
    try:
        yield
    finally:
        model.base_sigma = state[0]
        model.sigma = state[1]
        model.z = state[2]


def _abs_weight_product(layers):
    if len(layers) == 0:
        return None

    product = torch.abs(layers[0].weight)
    for layer in layers[1:]:
        product = torch.abs(layer.weight) @ product
    return product


def compute_lipschitz_bound(model, p=2, bound_type="k_p", layers=None):
    """
    Compute Lipschitz upper bounds with induced matrix norms.

    K_p = product_l ||W_l||_p.
    K_abs,p = ||abs(W_L) ... abs(W_1)||_p is only used for p=1 or p=inf.
    The matrix p=2 case is intentionally unsupported; K_abs,inf is not assumed
    to be smaller than the other p-norm bounds.
    """
    p = _norm_ord(p)
    if layers is None:
        layers = _linear_layers(model)
    if len(layers) == 0:
        return 1.0

    if bound_type == "k_p":
        value = 1.0
        for layer in layers:
            value *= float(_induced_matrix_norm(layer.weight.detach(), p).item())
        return value

    if bound_type == "k_abs":
        if p not in (1.0, float("inf")):
            return None
        product = _abs_weight_product(layers)
        return float(_induced_matrix_norm(product.detach(), p).item())

    raise ValueError(f"Unknown Lipschitz bound_type: {bound_type}")


def _forward_last_regu_samples(model, X, sigma, n_samples=None):
    """
    Certification-only sampling for R_LWA_2.

    The model forward remains analytic ReGU; this samples only the input to the last
    ReGU layer and then applies that last ReGU with plain ReLU plus the output
    layer to estimate the last-layer RS radius.
    """
    regu_layers = _regu_layers(model)
    sigma = _as_sigma_list(sigma, n_layers=len(regu_layers))
    last_regu_layer = regu_layers[-1]
    output_layer = _output_layer(model)
    was_training = model.training
    model.eval()
    n_samples = int(n_samples or getattr(model, "n_samples", 1000))

    with torch.no_grad(), _temporary_sigma(model, sigma):
        h = model.features_to_last_regu_input(X)
        batch_size = h.size(0)
        h_samples = h.unsqueeze(1).repeat(1, n_samples, 1)
        h_samples = h_samples.view(batch_size * n_samples, -1)

        if sigma[-1] > 0:
            h_samples = h_samples + float(sigma[-1]) * torch.randn_like(h_samples)

        logits = output_layer(torch.relu(last_regu_layer(h_samples)))
        logits = logits.view(batch_size, n_samples, -1)

    if was_training:
        model.train()
    return logits


def _randomized_smoothing_radius_from_logits(sample_logits, sigma_l, y):
    sample_preds = torch.argmax(sample_logits, dim=-1)
    n_samples = sample_preds.numel()
    counts = torch.bincount(sample_preds.reshape(-1), minlength=sample_logits.size(-1)).float()
    top_counts = torch.topk(counts, 2)
    label = int(top_counts.indices[0].item())

    if label != int(y.item()):
        return label, 0.0

    pA = _clamp_prob(top_counts.values[0] / n_samples)
    pB = _clamp_prob(top_counts.values[1] / n_samples)
    radius = float(sigma_l) * (phi_inverse(pA, 0) - phi_inverse(pB, 0)) / 2
    return label, float(radius.item())


def affine_surrogate_parameters(model, sigma):
    """
    W_aff and b_aff for the current architecture.

    Hidden layers use ReGU tangent affine maps; the final classifier remains
    the plain class-score layer.
    """
    regu_layers = _regu_layers(model)
    sigma = _as_sigma_list(sigma, n_layers=len(regu_layers))
    output_layer = _output_layer(model)
    device = output_layer.weight.device
    dtype = output_layer.weight.dtype
    input_dim = regu_layers[0].in_features

    W_aff = torch.eye(input_dim, device=device, dtype=dtype)
    b_aff = torch.zeros(input_dim, device=device, dtype=dtype)

    for layer_idx, layer in enumerate(regu_layers):
        tangent_offset = _row_noise_std(layer, sigma[layer_idx]) / math.sqrt(2 * math.pi)
        W_aff = 0.5 * (layer.weight @ W_aff)
        b_aff = 0.5 * (layer.weight @ b_aff + layer.bias) + tangent_offset

    return output_layer.weight @ W_aff, output_layer.weight @ b_aff + output_layer.bias


def affine_surrogate_scores(model, X, sigma):
    W_aff, b_aff = affine_surrogate_parameters(model, sigma)
    return model.flatten(X) @ W_aff.T + b_aff


# Affine-deviation certification is disabled for now. The previous full-network
# implementation is kept here as comments until replacement math is settled.
#
# def affine_deviation_bound(model, X, sigma, p=2, eps=1e-12):
#     """
#     e_p(x) from Proposition 10, adapted to this model's ReGU hidden layers.
#     """
#     p = _norm_ord(p)
#     regu_layers = _regu_layers(model)
#     sigma = _as_sigma_list(sigma, n_layers=len(regu_layers))
#     output = _output_layer(model)
#     x_aff = model.flatten(X)
#     batch_bounds = []
#
#     for sample_idx in range(x_aff.size(0)):
#         activation = x_aff[sample_idx]
#         total = activation.new_tensor(0.0)
#
#         for layer_idx, layer in enumerate(regu_layers):
#             z_aff = layer(activation)
#             std = _row_noise_std(layer, sigma[layer_idx]).clamp_min(eps)
#             delta_l = z_aff.pow(2) / (2 * math.sqrt(2 * math.pi) * std)
#             delta_norm = _vector_norm(delta_l, p)
#
#             propagation = activation.new_tensor(1.0)
#             for later_layer in regu_layers[layer_idx + 1:]:
#                 propagation = propagation * _row_norm_factor(later_layer, p)
#             propagation = propagation * _row_norm_factor(output, p)
#
#             total = total + propagation * delta_norm
#             activation = 0.5 * z_aff + _row_noise_std(layer, sigma[layer_idx]) / math.sqrt(2 * math.pi)
#
#         batch_bounds.append(total)
#
#     return torch.stack(batch_bounds)


def certify_affine(X, y, model, sigma, p=2):
    """Pure r_aff,p certificate from Proposition 9."""
    p = _norm_ord(p)
    q = _dual_norm_ord(p)
    W_aff, b_aff = affine_surrogate_parameters(model, sigma)
    scores = (model.flatten(X) @ W_aff.T + b_aff)[0]
    label = torch.argmax(scores)

    if label != y.item():
        return label.item(), 0.0

    radii = []
    for class_idx in range(scores.numel()):
        if class_idx == label.item():
            continue
        margin = scores[label] - scores[class_idx]
        denom = _vector_norm(W_aff[label] - W_aff[class_idx], q).clamp_min(1e-12)
        radii.append(margin / denom)

    return label.item(), float(torch.stack(radii).min().clamp_min(0).item())


# def certify_affine_deviation(X, y, model, sigma, p=2):
#     """Deviation-corrected affine surrogate certificate from Proposition 11."""
#     p = _norm_ord(p)
#     q = _dual_norm_ord(p)
#
#     with _temporary_sigma(model, sigma):
#         scores_model = model(X)[0]
#
#     label = torch.argmax(scores_model)
#     if label != y.item():
#         return label.item(), 0.0
#
#     W_aff, b_aff = affine_surrogate_parameters(model, sigma)
#     scores_aff = (model.flatten(X) @ W_aff.T + b_aff)[0]
#     e_p = affine_deviation_bound(model, X, sigma, p=p)[0]
#
#     radii = []
#     for class_idx in range(scores_aff.numel()):
#         if class_idx == label.item():
#             continue
#         adjusted_margin = scores_aff[label] - scores_aff[class_idx] - 2 * e_p
#         denom = _vector_norm(W_aff[label] - W_aff[class_idx], q).clamp_min(1e-12)
#         radii.append(adjusted_margin / denom)
#
#     return label.item(), float(torch.stack(radii).min().clamp_min(0).item())


def _certify_lwa_lipschitz(X, y, model, sigma, p=2, bound_type="k_p", n_samples=None):
    p = _norm_ord(p)
    sigma = _as_sigma_list(sigma, n_layers=len(_regu_layers(model)))

    with _temporary_sigma(model, sigma):
        scores = model(X)[0]

    label = torch.argmax(scores)
    if label != y.item():
        return label.item(), 0.0, {"R_LWA_1": 0.0, "R_LWA_2": 0.0}

    K_full = compute_lipschitz_bound(model, p=p, bound_type=bound_type, layers=_linear_layers(model))
    if K_full is None:
        return label.item(), None, {"R_LWA_1": None, "R_LWA_2": None}

    margin, _ = _margin_from_scores(scores)
    R_LWA_1 = float((margin / (2 * max(K_full, 1e-12))).item())

    sample_logits = _forward_last_regu_samples(model, X, sigma, n_samples=n_samples)[0]
    _, R_last_layer = _randomized_smoothing_radius_from_logits(sample_logits, sigma[-1], y)
    K_up_to_last = compute_lipschitz_bound(
        model,
        p=p,
        bound_type=bound_type,
        layers=_regu_layers(model)[:-1],
    )
    R_LWA_2 = None if K_up_to_last is None else R_last_layer / (2 * max(K_up_to_last, 1e-12))

    valid_radii = [r for r in (R_LWA_1, R_LWA_2) if r is not None]
    radius = max(valid_radii) if valid_radii else None
    return label.item(), radius, {"R_LWA_1": R_LWA_1, "R_LWA_2": R_LWA_2}


def certify_lwa_kp(X, y, model, sigma, p=2, n_samples=None):
    return _certify_lwa_lipschitz(X, y, model, sigma, p=p, bound_type="k_p", n_samples=n_samples)


def certify_lwa_kabs(X, y, model, sigma, p=1, n_samples=None):
    return _certify_lwa_lipschitz(X, y, model, sigma, p=p, bound_type="k_abs", n_samples=n_samples)


def certify_all_radius_results(X, y, model, sigma, norms=(1, 2, "inf"), n_samples=None):
    results = {}
    for p in norms:
        label = _norm_label(p)
        _, lwa_k_p, kp_parts = certify_lwa_kp(X, y, model, sigma, p=p, n_samples=n_samples)
        _, lwa_k_abs, kabs_parts = certify_lwa_kabs(X, y, model, sigma, p=p, n_samples=n_samples)
        # Affine-deviation certification is disabled for now.
        # _, aff_dev = certify_affine_deviation(X, y, model, sigma, p=p)
        _, aff = certify_affine(X, y, model, sigma, p=p)

        results[label] = {
            "lwa_k_p": lwa_k_p,
            "lwa_k_abs": lwa_k_abs,
            # "affine_deviation": aff_dev,
            "affine": aff,
            "parts": {
                "k_p": kp_parts,
                "k_abs": kabs_parts,
            },
        }
    return results
