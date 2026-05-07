from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Literal

import dimod
import numpy as np
import torch
import torch.nn as nn


# -----------------------------
# Init-scale helpers (your current strength)
# -----------------------------
def get_uniform_scale_for_linear(num_bits: int, input_var: float, labels_var: float) -> float:
    return math.sqrt(3.0 * labels_var / (2.0 * num_bits * input_var))


def get_uniform_scale_for_quad(
    num_edges: float,
    k: float,
    input_mean: float,
    input_var: float,
    labels_var: float,
) -> float:
    return (
        9.0 * labels_var
        / (2.0 * k * num_edges * input_var * (input_var + 2.0 * input_mean * input_mean))
    ) ** 0.25


def compute_init_scales(x_np: np.ndarray, y_np: np.ndarray, k: int) -> tuple[float, float]:
    num_bits = x_np.shape[1]
    num_edges = num_bits * (num_bits - 1) / 2.0

    input_mean = float(x_np.mean())
    input_var = float(x_np.var())
    labels_var = float(y_np.var())

    if input_var == 0 or labels_var == 0 or num_edges == 0:
        return 0.01, 0.01

    scale_w = get_uniform_scale_for_linear(num_bits, input_var, labels_var)
    scale_v = get_uniform_scale_for_quad(num_edges, float(k), input_mean, input_var, labels_var)
    return scale_w, scale_v


# -----------------------------
# Torch FM
# -----------------------------
class TorchFM(nn.Module):
    def __init__(self, d: int, k: int, scale_w: float = 0.1, scale_v: float = 0.1):
        super().__init__()
        self.d = d
        self.k = k
        self.w = nn.Parameter(torch.empty(d).uniform_(-scale_w, scale_w))
        self.v = nn.Parameter(torch.empty(d, k).uniform_(-scale_v, scale_v))
        self.w0 = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_linear = torch.matmul(x, self.w) + self.w0
        out_1 = torch.matmul(x, self.v).pow(2).sum(1)
        out_2 = torch.matmul(x.pow(2), self.v.pow(2)).sum(1)
        out_quadratic = 0.5 * (out_1 - out_2)
        return out_linear + out_quadratic

    def get_parameters(self):
        np_v = self.v.detach().cpu().numpy().copy()   # (d,k)
        np_w = self.w.detach().cpu().numpy().copy()   # (d,)
        np_w0 = self.w0.detach().cpu().numpy().copy() # scalar
        return np_v, np_w, float(np_w0)


def build_optimizer(model: TorchFM, base_lr: float) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [{"params": [model.v, model.w, model.w0]}],
        lr=base_lr,
        weight_decay=0.01,
    )


def train_fm(
    x_np: np.ndarray,
    y_np: np.ndarray,
    model: TorchFM,
    optimizer: torch.optim.Optimizer,
    epochs: int = 1000,
    patience: int | None = 50,
) -> None:
    X = torch.from_numpy(x_np).float()
    Y = torch.from_numpy(y_np).float()

    loss_fn = nn.MSELoss()
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stall = 0

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()

        cur_loss = loss.item()
        if cur_loss < best_loss - 1e-9:
            best_loss = cur_loss
            best_state = copy.deepcopy(model.state_dict())
            stall = 0
        else:
            stall += 1
            if patience is not None and stall >= patience:
                break

    model.load_state_dict(best_state)


# -----------------------------
# FM params -> QUBO helpers
# -----------------------------
def _torchfm_to_qubo_raw(model: TorchFM) -> tuple[np.ndarray, float]:
    """
    Returns:
      Q: full square matrix (d,d), upper/lower symmetric (diag holds linear)
      b: offset
    Energy in binary x∈{0,1}^d: E(x) = b + x^T Q x
    """
    v, w, w0 = model.get_parameters()
    d = len(w)

    # Pair terms from FM: sum_{i<j} <v_i,v_j> x_i x_j
    # In x^T Q x, off-diagonal contributes 2*Q_ij x_i x_j if symmetric.
    # So set Q_ij = 0.5 * <v_i,v_j> for i!=j in both [i,j],[j,i].
    Q = np.zeros((d, d), dtype=np.float64)
    for i in range(d):
        Q[i, i] = float(w[i])
        for j in range(i + 1, d):
            cij = float(np.dot(v[i], v[j]))
            Q[i, j] = 0.5 * cij
            Q[j, i] = 0.5 * cij

    b = float(w0)
    return Q, b


def _binary_qubo_to_ising(Q: np.ndarray, b: float) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Convert binary QUBO E(x)=b+x^TQx, x∈{0,1}
    to Ising E(s)=bI + h·s + s^T J s, s∈{-1,+1}
    with x=(s+1)/2.
    """
    d = Q.shape[0]
    q_sym = 0.5 * (Q + Q.T)
    h = np.zeros(d, dtype=np.float64)
    J = np.zeros((d, d), dtype=np.float64)

    # offdiag
    for i in range(d):
        for j in range(i + 1, d):
            qij = q_sym[i, j]
            J[i, j] = qij / 4.0
            J[j, i] = qij / 4.0

    # linear
    for i in range(d):
        h[i] = q_sym[i, i] / 2.0 + np.sum(q_sym[i, :]) / 4.0 - q_sym[i, i] / 4.0

    # offset
    bI = b + np.trace(q_sym) / 2.0
    for i in range(d):
        for j in range(i + 1, d):
            bI += q_sym[i, j] / 2.0

    return h, J, float(bI)


def _scale_ising(h: np.ndarray, J: np.ndarray, b: float) -> tuple[np.ndarray, np.ndarray, float]:
    m = max(float(np.max(np.abs(h))), float(np.max(np.abs(J))), 1e-12)
    return h / m, J / m, b / m


def _scale_qubo(Q: np.ndarray, b: float) -> tuple[np.ndarray, float]:
    m = max(float(np.max(np.abs(Q))), 1e-12)
    return Q / m, b / m


@dataclass
class TorchFMBQMConfig:
    rank: int = 8
    lr: float = 1e-2
    epochs: int = 1000
    patience: int | None = 50


class TorchFMBQM:
    def __init__(
        self,
        input_size: int,
        k: int = 8,
        lr: float = 1e-2,
        epochs: int = 1000,
        patience: int | None = 50,
        scale_w: float | None = None,
        scale_v: float | None = None,
    ):
        self.input_size = input_size
        self.k = k
        self.lr = lr
        self.epochs = epochs
        self.patience = patience

        sw = 0.1 if scale_w is None else scale_w
        sv = 0.1 if scale_v is None else scale_v
        self.model = TorchFM(d=input_size, k=k, scale_w=sw, scale_v=sv)

    @classmethod
    def from_data(
        cls,
        x: np.ndarray,
        y: np.ndarray,
        k: int = 8,
        lr: float = 1e-2,
        epochs: int = 1000,
        patience: int | None = 50,
        auto_scale: bool = True,
    ) -> "TorchFMBQM":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if x.ndim != 2:
            raise ValueError("x must be 2D array (n_samples, n_bits)")
        if y.ndim != 1:
            raise ValueError("y must be 1D array (n_samples,)")
        if x.shape[0] != y.shape[0]:
            raise ValueError("x and y must have same number of samples")
        if not np.all((x == 0) | (x == 1)):
            raise ValueError("x must be binary (0/1) for this FM->QUBO mapping")

        if auto_scale:
            scale_w, scale_v = compute_init_scales(x, y, k=k)
        else:
            scale_w, scale_v = 0.1, 0.1

        obj = cls(
            input_size=x.shape[1],
            k=k,
            lr=lr,
            epochs=epochs,
            patience=patience,
            scale_w=scale_w,
            scale_v=scale_v,
        )
        obj.train(x, y)
        return obj

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        lr: float | None = None,
        epochs: int | None = None,
        patience: int | None = None,
    ) -> None:
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        _lr = self.lr if lr is None else lr
        _epochs = self.epochs if epochs is None else epochs
        _patience = self.patience if patience is None else patience

        optimizer = build_optimizer(self.model, base_lr=_lr)
        train_fm(x, y, self.model, optimizer, epochs=_epochs, patience=_patience)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        X = torch.from_numpy(x).float()
        self.model.eval()
        with torch.no_grad():
            return self.model(X).cpu().numpy()

    def to_qubo(self, scaling: bool = True) -> tuple[dict[tuple[int, int], float], float]:
        Q, b = _torchfm_to_qubo_raw(self.model)
        if scaling:
            Q, b = _scale_qubo(Q, b)

        d = Q.shape[0]
        Qdict: dict[tuple[int, int], float] = {}
        for i in range(d):
            for j in range(d):
                if i == j or Q[i, j] != 0.0:
                    Qdict[(i, j)] = float(Q[i, j])
        return Qdict, float(b)

    def to_ising(
        self, scaling: bool = True
    ) -> tuple[dict[int, float], dict[tuple[int, int], float], float]:
        Q, b = _torchfm_to_qubo_raw(self.model)
        h, J, bI = _binary_qubo_to_ising(Q, b)
        if scaling:
            h, J, bI = _scale_ising(h, J, bI)

        hdict = {i: float(h[i]) for i in range(len(h))}
        Jdict = {}
        for i in range(len(h)):
            for j in range(i + 1, len(h)):
                if J[i, j] != 0.0:
                    Jdict[(i, j)] = float(J[i, j])
        return hdict, Jdict, float(bI)

    def to_bqm(
        self,
        vartype: Literal["BINARY", "SPIN"] = "BINARY",
        scaling: bool = True,
    ) -> dimod.BinaryQuadraticModel:
        if vartype == "BINARY":
            Q, b = self.to_qubo(scaling=scaling)
            return dimod.BinaryQuadraticModel.from_qubo(Q, offset=b)
        elif vartype == "SPIN":
            h, J, b = self.to_ising(scaling=scaling)
            return dimod.BinaryQuadraticModel.from_ising(h, J, offset=b)
        raise ValueError("vartype must be 'BINARY' or 'SPIN'")
