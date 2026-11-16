"""Gradients through the pH model (mbrl/grad): right, and the same with the memory-saving blocks."""

import torch

from mbrl.grad.env import GradStand, NA


def window_cost(env, actions):
    """Cost of a window: running cost (term 1) + cost at the end (term 2), for fixed actions (K, B, 22)."""
    env.reset()
    env.detach()
    total = 0.0
    for a in actions:
        r, _ = env.step(a)
        total = total - r.sum()
    return total + env.terminal_cost().sum()


def make(checkpoint):
    return GradStand(2, seed=3, checkpoint=checkpoint)


def test_gradient_matches_finite_differences():
    env = make(False)
    torch.manual_seed(0)
    actions = (0.05 * torch.randn(2, 2, NA, dtype=torch.float64)).requires_grad_(True)
    (grad,) = torch.autograd.grad(window_cost(env, actions), actions)
    assert torch.isfinite(grad).all()
    flat = grad.flatten()
    eps = 1e-6
    for j in flat.abs().topk(3).indices.tolist():
        d = torch.zeros_like(flat)
        d[j] = eps
        d = d.reshape(actions.shape)
        with torch.no_grad():
            env = make(False)
            up = window_cost(env, actions.detach() + d)
            env = make(False)
            down = window_cost(env, actions.detach() - d)
        fd = float((up - down) / (2 * eps))
        assert abs(fd - float(flat[j])) <= 1e-4 * max(abs(fd), 1e-9), (j, fd, float(flat[j]))


def test_checkpointed_gradient_is_the_same():
    torch.manual_seed(0)
    actions = 0.05 * torch.randn(3, 2, NA, dtype=torch.float64)
    grads = []
    for ck in (False, True):
        a = actions.clone().requires_grad_(True)
        (g,) = torch.autograd.grad(window_cost(make(ck), a), a)
        grads.append(g)
    assert torch.allclose(grads[0], grads[1], rtol=1e-9, atol=1e-12)
