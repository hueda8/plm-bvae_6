from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import dimod
import numpy as np
import torch
import torch.nn as nn


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
        np_v = self.v.detach().cpu().numpy().copy()
        np_w = self.w.detach().cpu().numpy().copy()
        np_w0 = self.w0.detach().cpu().numpy().copy()
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
    patience: int | None = None,
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


def torchfm_to_bqm(model: TorchFM) -> dimod.BinaryQuadraticModel:
    v, w, w0 = model.get_parameters()
    linear = {i: float(w[i]) for i in range(len(w))}
    quadratic = {}

    for i in range(len(w)):
        for j in range(i + 1, len(w)):
            quadratic[(i, j)] = float(np.dot(v[i], v[j]))

    return dimod.BinaryQuadraticModel(linear, quadratic, float(w0), vartype=dimod.BINARY)


@dataclass
class TorchFMBQMConfig:
    rank: int = 8
    lr: float = 1e-2
    epochs: int = 1000
    patience: int | None = 50


class TorchFMBQM:
    """
    fmqaライクな薄いラッパー:
      - from_data(...)
      - train(...)
      - predict(...)
      - to_bqm(...)
    """

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
            out = self.model(X).cpu().numpy()
        return out

    def to_bqm(self) -> dimod.BinaryQuadraticModel:
        return torchfm_to_bqm(self.model)
