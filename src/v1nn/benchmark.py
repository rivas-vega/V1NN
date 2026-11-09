"""Development benchmark. Validation is for tuning, never a final test set."""
import hashlib
import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .graph import Graph, rewire
from .model import ConnectomeRNN, FixedRetina
from .synthetic import moving_textures

DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1),
              (1, 1), (1, -1), (-1, 1), (-1, -1))
MODELS = ("static", "linear_history", "gru", "temporal_cnn",
          "connectome", "no_edges", "reset", "reservoir", "degree_null")


@dataclass
class PilotConfig:
    model: str = "gru"
    seed: int = 7
    data_seed: int = 20261109
    steps: int = 300
    batch: int = 64
    train_size: int = 2048
    validation_size: int = 512
    frames: int = 8
    image_size: int = 16
    grid: int = 4
    hidden: int = 64
    lr: float = 0.003
    device: str = "cpu"
    max_seconds: float = 600.0

    def validate(self):
        if self.model not in MODELS:
            raise ValueError(f"Unknown model: {self.model}")
        if min(self.steps, self.batch, self.train_size, self.validation_size,
               self.grid, self.hidden) < 1 or self.frames < 2 or self.image_size < 5:
            raise ValueError("Invalid pilot dimensions")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("Learning rate must be finite and positive")
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("A finite positive time limit is required")


def split_seed(data_seed, split):
    if split not in ("train", "validation", "counterfactual"):
        raise ValueError("Pilot supports development splits only")
    payload = f"v1nn-pilot-v1:{data_seed}:{split}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63)


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def paired_histories(batch=8, steps=8, size=16, seed=0):
    """Every texture has all eight velocities and exactly the same final frame."""
    generator = torch.Generator().manual_seed(seed)
    final = torch.rand(batch, 1, size, size, generator=generator)
    clips, labels = [], []
    for texture in final:
        for dx, dy in DIRECTIONS:
            clips.append(torch.stack([
                torch.roll(texture, ((t-steps+1)*dy, (t-steps+1)*dx), (-2, -1))
                for t in range(steps)]))
            labels.append((dx, dy))
    return torch.stack(clips), torch.tensor(labels, dtype=torch.float32)


def pixel_motion_reference(clips):
    """Analytic task check using raw last-two frames; not a matched neural model."""
    previous, current = clips[:, -2], clips[:, -1]
    errors = torch.stack([
        (torch.roll(previous, (dy, dx), (-2, -1))-current).square().mean((1, 2, 3))
        for dx, dy in DIRECTIONS], dim=1)
    return clips.new_tensor(DIRECTIONS)[errors.argmin(1)]


def prepare_data(config):
    config.validate()
    retina = FixedRetina(config.grid)
    data, manifest = {}, {}
    with torch.no_grad():
        for split, count in (("train", config.train_size),
                             ("validation", config.validation_size)):
            seed = split_seed(config.data_seed, split)
            clips, _, target = moving_textures(batch=count, steps=config.frames,
                                              size=config.image_size, seed=seed)
            features = retina(clips)
            data[split] = (features, target)
            manifest[split] = {"seed": seed, "clips": count,
                               "features_sha256": tensor_hash(features),
                               "targets_sha256": tensor_hash(target)}
            if split == "validation":
                manifest["pixel_reference"] = metrics(pixel_motion_reference(clips), target)
        seed = split_seed(config.data_seed, "counterfactual")
        clips, target = paired_histories(16, config.frames, config.image_size, seed)
        data["counterfactual"] = (retina(clips), target)
        manifest["counterfactual"] = {
            "seed": seed, "textures": 16, "clips": len(clips),
            "features_sha256": tensor_hash(data["counterfactual"][0]),
            "targets_sha256": tensor_hash(target),
            "same_final_frame_per_eight_directions": True}
        # Fixed affine calibration fitted on training frames only, shared by all models.
        mean = data["train"][0].mean((0, 1))
        scale = data["train"][0].std((0, 1)).clamp_min(1e-6)
        for split, (features, target) in data.items():
            data[split] = ((features-mean)/scale, target)
        manifest["normalization"] = {"fit_split": "train", "mean": mean.tolist(),
                                      "std": scale.tolist(), "trainable": False}
    return data, manifest


class Reference(nn.Module):
    def __init__(self, kind, features, frames, hidden):
        super().__init__()
        self.kind = kind
        if kind in ("static", "linear_history"):
            self.core = nn.Linear(features if kind == "static" else features*frames, 2)
        elif kind == "gru":
            self.core = nn.GRU(features, hidden, batch_first=True)
            self.readout = nn.Linear(hidden, 2)
        elif kind == "temporal_cnn":
            self.core = nn.Sequential(nn.Conv1d(features, hidden, 3), nn.GELU())
            self.readout = nn.Linear(hidden, 2)
        else:
            raise ValueError(kind)

    def forward(self, x):
        if self.kind == "static":
            return self.core(x[:, -1])
        if self.kind == "linear_history":
            return self.core(x.flatten(1))
        if self.kind == "gru":
            states, _ = self.core(x)
            return self.readout(states[:, -1])
        return self.readout(self.core(x.transpose(1, 2)).mean(-1))


class GraphPredictor(nn.Module):
    def __init__(self, core, reset=False):
        super().__init__()
        self.core, self.reset = core, reset

    def forward(self, x):
        prediction, _ = self.core(x, reset_each_frame=self.reset)
        return prediction[:, -1]


def build_model(config, graph=None):
    torch.manual_seed(config.seed)
    features = 2*config.grid**2
    if config.model in ("static", "linear_history", "gru", "temporal_cnn"):
        if config.model == "temporal_cnn" and config.frames < 3:
            raise ValueError("Temporal CNN needs at least three frames")
        return Reference(config.model, features, config.frames, config.hidden), {
            "graph": None, "hidden": config.hidden}
    if graph is None:
        rng = np.random.default_rng(12345)
        mask = rng.random((config.hidden, config.hidden)) < 0.08
        np.fill_diagonal(mask, False)
        src, dst = np.nonzero(mask)
        graph = Graph(config.hidden, src, dst,
                      np.where(np.arange(config.hidden) % 5 == 0, -1, 1))
        provenance = "synthetic; NOT MICrONS"
    else:
        provenance = "user_supplied_graph; see source hash and adjacent provenance"
    info = {"graph": provenance, "nodes": graph.n, "original_edges": len(graph.src)}
    if config.model == "degree_null":
        graph, audit = rewire(graph, swaps=5*len(graph.src), seed=config.seed)
        info["rewiring"] = audit
        info["null_constraints"] = "directed degrees only; not type/distance matched"
    if config.model == "no_edges":
        graph = Graph(graph.n, np.array([], dtype=int), np.array([], dtype=int), graph.signs)
    core = ConnectomeRNN(graph, features, 2, routing_seed=config.seed)
    if config.model == "reservoir":
        for name, parameter in core.named_parameters():
            if not name.startswith("readout."):
                parameter.requires_grad_(False)
    info["edges"] = len(graph.src)
    return GraphPredictor(core, config.model == "reset"), info


def metrics(prediction, target):
    squared = (prediction-target).square().sum(-1)
    return {"mean_squared_vector_error": float(squared.mean()),
            "mean_endpoint_error": float(squared.sqrt().mean())}


def config_dict(config):
    return asdict(config)
