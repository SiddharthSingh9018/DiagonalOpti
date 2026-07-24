from __future__ import annotations

import torch


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if weight_decay:
                    p.mul_(1 - lr * weight_decay)
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                p.add_(update.sign(), alpha=-lr)
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


class SophiaG(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, betas=(0.965, 0.99), rho=0.04, eps=1e-12, weight_decay=0.0, update_interval=10):
        defaults = dict(lr=lr, betas=betas, rho=rho, eps=eps, weight_decay=weight_decay, update_interval=update_interval)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            rho = group["rho"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            interval = group["update_interval"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian"] = torch.zeros_like(p)
                state["step"] += 1
                exp_avg = state["exp_avg"]
                hessian = state["hessian"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                if state["step"] % interval == 1:
                    hessian.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if weight_decay:
                    p.mul_(1 - lr * weight_decay)
                update = (exp_avg / (rho * hessian.add(eps))).clamp_(-1, 1)
                p.add_(update, alpha=-lr)
        return loss


class MuonFallback(torch.optim.Optimizer):
    def __init__(self, params, lr=2e-4, momentum=0.95, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay))

    @staticmethod
    def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
        if g.ndim < 2:
            return g.sign()
        x = g.float()
        transposed = x.size(0) > x.size(1)
        if transposed:
            x = x.T
        x = x / (x.norm() + eps)
        a, b, c = 3.4445, -4.7750, 2.0315
        for _ in range(steps):
            xx_t = x @ x.T
            x = a * x + (b * xx_t + c * xx_t @ xx_t) @ x
        if transposed:
            x = x.T
        return x.to(g.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if weight_decay:
                    p.mul_(1 - lr * weight_decay)
                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(p.grad)
                update = self.zeropower_via_newtonschulz5(buf, ns_steps) if p.ndim >= 2 else buf.sign()
                p.add_(update, alpha=-lr)
        return loss


class DiagonalOptimiser(torch.optim.Optimizer):
    """PyTorch benchmark variant of the repository's diagonal-curvature idea.

    The repository optimizer is NumPy-based and estimates curvature with finite
    differences. This PyTorch optimizer keeps that cost visible by optionally
    perturbing parameters and calling a closure during curvature refreshes.
    """

    def __init__(
        self,
        params,
        lr=3e-4,
        beta=0.99,
        eps=1e-8,
        weight_decay=0.0,
        max_update=1.0,
        curvature_interval=10,
        fd_eps=1e-3,
    ):
        defaults = dict(
            lr=lr,
            beta=beta,
            eps=eps,
            weight_decay=weight_decay,
            max_update=max_update,
            curvature_interval=curvature_interval,
            fd_eps=fd_eps,
        )
        super().__init__(params, defaults)
        self.curvature_updates = 0
        self.hvp_probes = 0

    @torch.no_grad()
    def _update_from_current_grads(self, group):
        beta = group["beta"]
        for p in group["params"]:
            if p.grad is None:
                continue
            state = self.state[p]
            if not state:
                state["step"] = 0
                state["diag_curvature"] = torch.zeros_like(p)
            state["diag_curvature"].mul_(beta).addcmul_(p.grad, p.grad, value=1 - beta)

    def _finite_difference_curvature_refresh(self, group, closure):
        if closure is None:
            self._update_from_current_grads(group)
            self.curvature_updates += 1
            return
        fd_eps = group["fd_eps"]
        with torch.no_grad():
            params = [p for p in group["params"] if p.grad is not None]
            base_grads = [p.grad.detach().clone() for p in params]
            directions = [torch.sign(g) for g in base_grads]
            for p, direction in zip(params, directions):
                p.add_(direction, alpha=fd_eps)
        closure()
        with torch.no_grad():
            for p, base_grad in zip(params, base_grads):
                hvp_diag = (p.grad - base_grad).abs() / fd_eps
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["diag_curvature"] = torch.zeros_like(p)
                state["diag_curvature"].mul_(group["beta"]).add_(hvp_diag, alpha=1 - group["beta"])
            for p, direction, base_grad in zip(params, directions, base_grads):
                p.add_(direction, alpha=-fd_eps)
                p.grad.copy_(base_grad)
        self.curvature_updates += 1
        self.hvp_probes += 1

    def step(self, closure=None):
        loss = None
        for group in self.param_groups:
            first_param = next((p for p in group["params"] if p.grad is not None), None)
            if first_param is None:
                continue
            state = self.state[first_param]
            group_step = state.get("step", 0) + 1 if state else 1
            if group_step == 1 or group_step % group["curvature_interval"] == 0:
                self._finite_difference_curvature_refresh(group, closure)
            else:
                self._update_from_current_grads(group)
                self.curvature_updates += 1

            with torch.no_grad():
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    state["step"] = group_step
                    if group["weight_decay"]:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    diag = state["diag_curvature"]
                    update = p.grad / (diag.sqrt() + group["eps"])
                    update.clamp_(min=-group["max_update"], max=group["max_update"])
                    p.add_(update, alpha=-group["lr"])
        return loss


def make_optimizer(name: str, model: torch.nn.Module, learning_rates: dict[str, float], weight_decay: float):
    lr = learning_rates[name]
    if name == "AdamW":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "Lion":
        return Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "Sophia":
        return SophiaG(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "Muon":
        return MuonFallback(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "DiagonalOptimiser":
        return DiagonalOptimiser(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def optimizer_stats(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    return {
        "curvature_updates": float(getattr(optimizer, "curvature_updates", 0)),
        "hvp_probes": float(getattr(optimizer, "hvp_probes", 0)),
    }
