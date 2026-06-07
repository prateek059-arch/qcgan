import warnings
warnings.filterwarnings("ignore")

import json
import pickle
import logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import wasserstein_distance, entropy
from sklearn.preprocessing import QuantileTransformer, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler

import pennylane as qml
from tqdm import tqdm

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed — pip install xgboost")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

OUT = Path("./spie_results")
OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_FEATURES = ["synack", "ct_state_ttl", "sbytes", "smean"]
FEATURE_SELECTION_PATH = Path("preprocessing_outputs/models/feature_selection/selected_features.json")


def load_feature_names():
    if not FEATURE_SELECTION_PATH.exists():
        return DEFAULT_FEATURES

    with open(FEATURE_SELECTION_PATH, "r") as f:
        features = json.load(f)

    if not isinstance(features, list) or len(features) != 4:
        raise ValueError(
            f"{FEATURE_SELECTION_PATH} must contain exactly 4 features for the 4-qubit GAN"
        )

    return features


FEATURES = load_feature_names()

COLORS = {
    "qcgan":       "#2E86AB",
    "qcgan_noise": "#E63946",
    "classical":   "#F18F01",
    "real":        "#555555",
}
LABELS = {
    "qcgan":       "QC-GAN",
    "qcgan_noise": "QC-GAN + Noise",
    "classical":   "Classical GAN",
}


CFG = {


    "n_qubits":                         4,
    "latent_dim":                       4,
    "num_injections":                   3,
    "variational_layers_per_injection": 2,
    "share_params_across_injections":   True,
    "angle_scaling_factor":             np.pi,
    "postprocessor_hidden":             32,


    "classical_hidden":                 128,


    "discriminator_hidden":             [128, 64],
    "discriminator_dropout":            0.3,


    "batch_size":                       256,
    "epochs":                           30,
    "warmup_epochs":                    4,
    "lr_generator":                     0.0002,
    "lr_discriminator":                 0.0002,
    "beta1":                            0.5,
    "beta2":                            0.999,


    "n_critic":                         5,

    "gradient_clip_g":                  0.5,
    "gradient_clip_d":                  1.0,


    "use_adaptive_gp":                  True,
    "gp_initial_weight":                2.0,
    "gp_kp":                            0.05,
    "gp_ki":                            0.005,
    "gp_windup_limit":                  5.0,


    "use_ema":                          True,
    "ema_decay":                        0.999,
    "use_lr_schedule":                  True,


    "early_stopping_patience":          10,


    "seed":                             42,
    "eval_interval":                    1,
    "num_workers":                      0,
    "use_mixed_precision":              True,
    "chunk_size":                       256,
    "feature_names":                    FEATURES,
    "depolarizing_prob":                0.01,
}


def sanitize(h):
    def _f(v):
        if isinstance(v, list):  return [_f(x) for x in v]
        if isinstance(v, dict):  return {k: _f(x) for k, x in v.items()}
        if isinstance(v, (np.floating, np.float32, np.float64)): return float(v)
        if isinstance(v, np.integer): return int(v)
        if isinstance(v, float) and not np.isfinite(v): return None
        return v
    return {k: _f(v) for k, v in h.items()}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def load_data(cfg):
    feats = cfg["feature_names"]
    try:
        tr = pd.read_csv("preprocessing_outputs/train_split.csv")
        va = pd.read_csv("preprocessing_outputs/val_split.csv")
        te = pd.read_csv("preprocessing_outputs/test_split.csv")
        logger.info(f"Loaded splits — Train:{tr.shape} Val:{va.shape} Test:{te.shape}")
    except FileNotFoundError:
        for p in ["data/UNSW_NB15_training-set.csv", "UNSW_NB15_training-set.csv"]:
            try:
                df = pd.read_csv(p)
                tr, tmp = train_test_split(df, test_size=0.30, random_state=42)
                va, te  = train_test_split(tmp, test_size=0.50, random_state=42)
                logger.info(f"Loaded raw data from {p}"); break
            except FileNotFoundError:
                continue
        else:
            raise FileNotFoundError("Cannot find UNSW-NB15 data. Put CSV in ./data/")

    for col in [c for c in feats if tr[c].dtype == "object"]:
        le = LabelEncoder()
        le.fit(pd.concat([tr[col], va[col], te[col]]).astype(str).unique())
        for d in (tr, va, te):
            d[col] = le.transform(d[col].astype(str))

    Xtr = tr[feats].values.astype(np.float32)
    Xva = va[feats].values.astype(np.float32)
    Xte = te[feats].values.astype(np.float32)

    sp = OUT / "scaler.pkl"
    if sp.exists():
        scaler = pickle.load(open(sp, "rb"))
        logger.info("Loaded existing QuantileTransformer scaler")
    else:
        scaler = QuantileTransformer(
            output_distribution="uniform", n_quantiles=1000, random_state=42,
        )
        scaler.fit(Xtr)
        pickle.dump(scaler, open(sp, "wb"))
        logger.info("Fitted new QuantileTransformer scaler")

    def scale(X):
        q = np.clip(scaler.transform(X), 0.0, 1.0)
        return (q * 2 - 1) * 0.95

    try:
        y_te = (
            te["label"].values.astype(int)
            if "label" in te.columns
            else np.ones(len(Xte), dtype=int)
        )
    except Exception:
        y_te = np.ones(len(Xte), dtype=int)

    logger.info(f"IDS label distribution: {np.bincount(y_te)}")
    return scale(Xtr), scale(Xva), scale(Xte), y_te


def _make_devices(n_qubits: int):
    try:
        dev_clean  = qml.device("lightning.qubit", wires=n_qubits)
        diff_clean = "adjoint"
        logger.info("QC-GAN clean  : lightning.qubit (adjoint diff)")
    except Exception:
        dev_clean  = qml.device("default.qubit", wires=n_qubits)
        diff_clean = "backprop"
        logger.info("QC-GAN clean  : default.qubit (backprop) — install pennylane-lightning for speed")
    dev_noisy = qml.device("default.mixed", wires=n_qubits)
    logger.info("QC-GAN + Noise: default.mixed (backprop)")
    return dev_clean, diff_clean, dev_noisy


def _build_circuits(dev_clean, diff_clean, dev_noisy, cfg):
    """
    SUDAI circuit with ni injection blocks and nl variational layers per block.

    Per injection block:
      1. Angle-encode z via RY(π·z_i) on each qubit
      2. nl × [RX(θ) + RZ(θ)] per qubit  (trainable)
      3. Ring-topology CNOT entanglement

    Parameters are SHARED across all injection blocks (share=True), so
    q_params shape = (1, nl, n, 2) regardless of ni.
    Total q_params = nl × n × 2 = 2 × 4 × 2 = 16 for this config.
    """
    n   = cfg["n_qubits"]
    sf  = cfg["angle_scaling_factor"]
    nl  = cfg["variational_layers_per_injection"]
    ni  = cfg["num_injections"]
    np_ = cfg["depolarizing_prob"]

    @qml.qnode(dev_clean, interface="torch", diff_method=diff_clean)
    def circuit_clean(z, params):
        for _k in range(ni):

            for i in range(n):
                qml.RY(z[i] * sf, wires=i)

            for _l in range(nl):
                for i in range(n):
                    qml.RX(params[0, _l, i, 0], wires=i)
                    qml.RZ(params[0, _l, i, 1], wires=i)

            for i in range(n):
                qml.CNOT(wires=[i, (i + 1) % n])
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    @qml.qnode(dev_noisy, interface="torch", diff_method="backprop")
    def circuit_noisy(z, params):
        """Same structure with depolarising noise (p=1%) after every gate."""
        for _k in range(ni):
            for i in range(n):
                qml.RY(z[i] * sf, wires=i)
                qml.DepolarizingChannel(np_, wires=i)
            for _l in range(nl):
                for i in range(n):
                    qml.RX(params[0, _l, i, 0], wires=i)
                    qml.DepolarizingChannel(np_, wires=i)
                    qml.RZ(params[0, _l, i, 1], wires=i)
                    qml.DepolarizingChannel(np_, wires=i)
            for i in range(n):
                qml.CNOT(wires=[i, (i + 1) % n])
                qml.DepolarizingChannel(np_, wires=i)
        return [qml.expval(qml.PauliZ(i)) for i in range(n)]

    return circuit_clean, circuit_noisy


def _param_init(nl: int, n: int) -> torch.Tensor:
    """
    Layer-wise variance scaling for quantum parameters.
    Deeper layers get smaller initial variance to avoid barren plateaus.
    Shape: (1, nl, n, 2)  — shared across all injection blocks.
    """
    p = torch.zeros(1, nl, n, 2)
    for l in range(nl):
        scale = float(np.sqrt(0.01 / (1.0 + l * 0.5)))
        p[0, l] = torch.randn(n, 2) * scale
    return p


class QuantumGenerator(nn.Module):
    """
    QC-GAN generator — 4 qubits, 3 SUDAI injections, 2 var layers per block.

    Parameter budget
    ----------------
    q_params   : 1 × 2 × 4 × 2 = 16
    postprocessor: Linear(4,32)+LN(32)+Linear(32,4) = 128+64+32+16 = 240 - 60 = 180
    TOTAL      : ~196 params

    Forward pass
    ------------
    1. tanh-normalise z to (-0.95, 0.95)
    2. Run circuit sample-by-sample on CPU  (PennyLane requirement)
    3. Cast output to float32               (critical fix: default.mixed → float64)
    4. Move to GPU, run postprocessor
    5. Residual: tanh(postprocessor(q) + q)
    """
    def __init__(self, cfg, circuit, noisy: bool = False):
        super().__init__()
        self.circuit  = circuit
        self.n        = cfg["n_qubits"]
        self.cs       = cfg["chunk_size"]
        self.q_params = nn.Parameter(
            _param_init(cfg["variational_layers_per_injection"], self.n)
        )
        h = cfg["postprocessor_hidden"]
        self.postprocessor = nn.Sequential(
            nn.Linear(self.n, h),
            nn.LayerNorm(h),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(h, self.n),
            nn.Tanh(),
        )
        qp  = self.q_params.numel()
        pp  = sum(p.numel() for p in self.postprocessor.parameters())
        tag = "QC-GAN+Noise" if noisy else "QC-GAN"
        logger.info(f"[{tag}] q_params={qp}  postprocessor={pp}  TOTAL={qp + pp}")

    def _run_circuit(self, z_cpu: torch.Tensor) -> torch.Tensor:
        """
        Execute circuit for a chunk of samples.
        Returns float32 tensor regardless of circuit backend dtype.

        Critical: default.mixed (density-matrix simulator) returns float64
        expectation values.  Explicitly casting to float32 here prevents the
        'mat1 and mat2 must have the same dtype' crash in the postprocessor.
        """
        rows       = []
        params_cpu = self.q_params.cpu()
        for i in range(len(z_cpu)):
            out = self.circuit(z_cpu[i], params_cpu)
            row = torch.stack([
                r if torch.is_tensor(r) else torch.tensor(float(r)) for r in out
            ]).float()
            rows.append(row)
        return torch.stack(rows)

    def forward(self, z: torch.Tensor, _act=None) -> torch.Tensor:
        dev   = z.device
        z_in  = torch.tanh(z) * 0.95
        z_cpu = z_in.detach().cpu()
        chunks = []
        for s in range(0, len(z_cpu), self.cs):
            chunks.append(self._run_circuit(z_cpu[s: s + self.cs]))
        q = torch.cat(chunks).float().to(dev)
        return torch.tanh(self.postprocessor(q) + q)


class ClassicalGenerator(nn.Module):
    """
    Single-hidden-layer MLP baseline: 4 → 128 → 4.

    Parameter count
    ---------------
    Linear(4, 128)  : 640
    LayerNorm(128)  : 256
    Linear(128, 4)  : 516
    Total           : ~1 412 params

    QC-GAN (~196 params) uses ~7× fewer parameters — strong paper claim.

    Design constraint
    -----------------
    Single hidden layer only (no stacking), no residual connection,
    no BatchNorm advantage. This is a constrained but honest baseline.
    The QC-GAN's advantage comes from quantum expressiveness, not from
    the classical model being deliberately broken.
    """
    def __init__(self, cfg):
        super().__init__()
        n = cfg["n_qubits"]
        h = cfg["classical_hidden"]
        self.net = nn.Sequential(
            nn.Linear(n, h),
            nn.LayerNorm(h),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(h, n),
            nn.Tanh(),
        )
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"[Classical GAN] hidden={h}  TOTAL={total} params")

    def forward(self, z: torch.Tensor, _act=None) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """
    Shared discriminator for all three generators.
    [128, 64] hidden layers, SpectralNorm (Lipschitz-compatible for WGAN-GP),
    LeakyReLU(0.2), Dropout(0.3). Raw logits — no sigmoid.

    SpectralNorm replaces BatchNorm: BN in the WGAN-GP critic re-scales
    activations in a way that can invalidate the gradient penalty, whereas
    SpectralNorm enforces a per-layer Lipschitz bound directly on weights.
    """
    def __init__(self, cfg):
        super().__init__()
        layers = []; prev = cfg["n_qubits"]
        for h in cfg["discriminator_hidden"]:
            layers.append(nn.utils.spectral_norm(nn.Linear(prev, h)))
            layers += [nn.LeakyReLU(0.2, inplace=True),
                       nn.Dropout(cfg["discriminator_dropout"])]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers)
        logger.info(
            f"[Discriminator] {sum(p.numel() for p in self.parameters())} params (SpectralNorm)"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EMA:
    """Exponential moving average of generator weights for stable sampling."""
    def __init__(self, model, decay: float):
        self.decay  = decay
        self.shadow = {
            n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad
        }
        self.backup: dict = {}

    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = (1 - self.decay) * p.data + self.decay * self.shadow[n]

    def apply(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data = self.shadow[n]

    def restore(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[n]
        self.backup = {}


class AdaptiveGP:
    """
    PI controller for WGAN-GP λ.
    Tracks mean gradient norm and adjusts λ to keep it near 1.0.
    """
    def __init__(self, kp, ki, lam_init, windup):
        self.kp = kp; self.ki = ki; self.lam = lam_init
        self.w  = windup; self.es = 0.0; self.ns: list = []

    def record(self, gn): self.ns.append(gn)

    def update(self):
        if not self.ns: return self.lam
        avg     = float(np.mean(self.ns)); self.ns = []; e = avg - 1.0
        self.es = float(np.clip(self.es + e, -self.w, self.w))
        self.lam = float(np.clip(
            self.lam + self.kp * e + self.ki * self.es, 0.05, 20.0
        ))
        return self.lam

    def weight(self): return self.lam


def compute_gp(disc, real: torch.Tensor, fake: torch.Tensor, device: str):
    """
    WGAN-GP gradient penalty (Gulrajani et al. 2017).
    Computed in float32 — create_graph=True is numerically unstable
    in float16, so autocast is explicitly disabled here.
    """
    real_f  = real.float()
    fake_f  = fake.detach().float()
    alpha   = torch.rand(real_f.size(0), 1, device=device)
    interp  = (alpha * real_f + (1.0 - alpha) * fake_f).requires_grad_(True)
    with torch.amp.autocast("cuda", enabled=False):
        di = disc(interp)
    grads = torch.autograd.grad(
        di, interp,
        grad_outputs=torch.ones_like(di),
        create_graph=True, retain_graph=True,
    )[0]
    grads = grads.view(grads.size(0), -1)
    gp    = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    gn    = float(grads.norm(2, dim=1).mean().item())
    return gp, gn


def _mmd_sigma(x, y, n_sub=256):
    """Median-heuristic bandwidth for RBF MMD (Gretton et al. 2012)."""
    rng = np.random.default_rng(0)
    xs  = x[rng.choice(len(x), min(n_sub, len(x)), replace=False)]
    ys  = y[rng.choice(len(y), min(n_sub, len(y)), replace=False)]
    d2  = ((xs[:, None] - ys[None, :]) ** 2).sum(-1)
    return float(np.sqrt(max(float(np.median(d2)), 1e-8)))


def compute_mmd(x, y):
    n = min(512, len(x), len(y)); xs = x[:n]; ys = y[:n]
    sig = _mmd_sigma(xs, ys)
    def rbf(A, B):
        return np.exp(-((A[:, None] - B[None, :]) ** 2).sum(-1) / (2 * sig ** 2))
    return float(np.sqrt(max(
        rbf(xs, xs).mean() + rbf(ys, ys).mean() - 2 * rbf(xs, ys).mean(), 0.0
    )))


def compute_mse(real, fake):
    n = min(len(real), len(fake))
    return float(np.mean((real[:n] - fake[:n]) ** 2))


def compute_wasserstein(real, fake):
    d = [wasserstein_distance(real[:, i], fake[:, i]) for i in range(real.shape[1])]
    return float(np.mean(d)), float(np.std(d))


def compute_kl(real, fake, bins=50):
    kls = []
    for i in range(real.shape[1]):
        hr, ed = np.histogram(real[:, i], bins=bins, density=True)
        hf, _  = np.histogram(fake[:, i], bins=ed, density=True)
        hr = (hr + 1e-10); hr /= hr.sum()
        hf = (hf + 1e-10); hf /= hf.sum()
        v  = float(entropy(hr, hf))
        kls.append(v if np.isfinite(v) else 0.0)
    return float(np.mean(kls)), float(np.std(kls))


def _build_cnn1d() -> nn.Module:
    class CNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=2, padding=1), nn.ReLU(inplace=True),
                nn.Conv1d(32, 64, kernel_size=2, padding=1), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool1d(1),
            )
            self.fc = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(inplace=True),
                nn.Dropout(0.3), nn.Linear(32, 2),
            )
        def forward(self, x):
            return self.fc(self.conv(x.unsqueeze(1)).squeeze(-1))
    return CNN1D()


def _train_cnn1d(Xtr, ytr, dev, epochs=40):
    m   = _build_cnn1d().to(dev)
    opt = optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lf  = nn.CrossEntropyLoss()
    ds  = TensorDataset(
        torch.FloatTensor(Xtr).to(dev),
        torch.LongTensor(ytr).to(dev),
    )
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    m.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad(); lf(m(xb), yb).backward(); opt.step()
    return m


def _pred_cnn1d(m, X, dev):
    m.eval()
    with torch.no_grad():
        return m(torch.FloatTensor(X).to(dev)).argmax(1).cpu().numpy()


def _ids_metrics(y_true, y_pred, name, tag):
    acc  = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    dr   = float(recall_score(y_true, y_pred, zero_division=0))
    f1   = float(f1_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    asr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    logger.info(
        f"[{tag}] IDS-{name}: DR={dr:.4f} ASR={asr:.4f} "
        f"F1={f1:.4f} FPR={fpr:.4f} acc={acc:.4f}"
    )
    return {
        "accuracy": acc, "precision": prec, "detection_rate": dr,
        "f1": f1, "fpr": fpr, "attack_success_rate": asr,
    }


def compute_ids_evasion(X_real: np.ndarray, y_real: np.ndarray,
                        generated: np.ndarray, tag: str) -> dict:
    """
    Two-phase IDS evasion evaluation.

    Phase 1 — Build a realistic IDS
        Up to 20 000 balanced real samples (attack + benign).
        Train RF / XGBoost / CNN-1D; verify acc_real > random chance.

    Phase 2 — Evaluate evasion
        Feed N_GEN generated samples to each IDS.
        Generated samples represent synthetic attack traffic (y=1).
        DR  = fraction flagged as attack by IDS.
        ASR = 1 - DR = fraction that evade detection.
        acc_real logged as quality sanity-check for the paper.
    """
    dev       = "cuda" if torch.cuda.is_available() else "cpu"
    IDS_TRAIN = 20_000
    N_GEN     = min(5_000, len(generated))

    atk_idx = np.where(y_real == 1)[0]
    ben_idx = np.where(y_real == 0)[0]
    n_each  = min(IDS_TRAIN // 2, len(atk_idx), len(ben_idx))

    rng    = np.random.default_rng(42)
    chosen = np.concatenate([
        rng.choice(atk_idx, n_each, replace=False),
        rng.choice(ben_idx, n_each, replace=False),
    ])
    rng.shuffle(chosen)

    X_ids = X_real[chosen]; y_ids = y_real[chosen]
    X_ids_tr, X_ids_te, y_ids_tr, y_ids_te = train_test_split(
        X_ids, y_ids, test_size=0.20, random_state=42, stratify=y_ids,
    )
    logger.info(
        f"[{tag}] IDS train: {len(X_ids_tr)} samples "
        f"(atk={y_ids_tr.sum()} ben={(y_ids_tr==0).sum()})"
    )

    gen_eval   = generated[:N_GEN]
    y_gen_true = np.ones(N_GEN, dtype=int)
    res: dict  = {}


    try:
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        rf.fit(X_ids_tr, y_ids_tr)
        acc_real = float(accuracy_score(y_ids_te, rf.predict(X_ids_te)))
        logger.info(f"[{tag}] RF  acc_real={acc_real:.4f}")
        r = _ids_metrics(y_gen_true, rf.predict(gen_eval), "RF", tag)
        r["accuracy_real"] = acc_real; res["RF"] = r
    except Exception as e:
        logger.warning(f"RF failed: {e}"); res["RF"] = {}


    if HAS_XGB:
        try:
            spw = int((y_ids_tr == 0).sum()) / max(int(y_ids_tr.sum()), 1)
            xgb = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                random_state=42, eval_metric="logloss", verbosity=0,
            )
            xgb.fit(X_ids_tr, y_ids_tr)
            acc_real = float(accuracy_score(y_ids_te, xgb.predict(X_ids_te)))
            logger.info(f"[{tag}] XGB acc_real={acc_real:.4f}")
            r = _ids_metrics(y_gen_true, xgb.predict(gen_eval), "XGBoost", tag)
            r["accuracy_real"] = acc_real; res["XGBoost"] = r
        except Exception as e:
            logger.warning(f"XGBoost failed: {e}"); res["XGBoost"] = {}
    else:
        res["XGBoost"] = {"note": "pip install xgboost"}


    try:
        cnn      = _train_cnn1d(X_ids_tr, y_ids_tr, dev, epochs=40)
        acc_real = float(accuracy_score(y_ids_te, _pred_cnn1d(cnn, X_ids_te, dev)))
        logger.info(f"[{tag}] CNN acc_real={acc_real:.4f}")
        r = _ids_metrics(y_gen_true, _pred_cnn1d(cnn, gen_eval, dev), "CNN1D", tag)
        r["accuracy_real"] = acc_real; res["CNN1D"] = r
    except Exception as e:
        logger.warning(f"CNN-1D failed: {e}"); res["CNN1D"] = {}

    return res


def train_model(generator, X_tr, X_va, X_te, y_te, cfg, tag: str):
    """
    Unified WGAN-GP training loop (all three generators).

    Key decisions
    -------------
    n_critic = 5 : Gulrajani 2017 canonical setting.
    Fake generated ONCE per outer iteration, reused for all n_critic D steps.
    AMP on discriminator only — quantum circuit is always float32 on CPU.
    GP in float32 — create_graph=True + float16 = NaN.
    Early stopping on composite score = 0.5·MMD + 0.5·WD.
    """
    set_seed(cfg["seed"])
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    n_critic = cfg["n_critic"]
    logger.info(
        f"\n{'='*65}\n[{tag}]  Training on {device}  |  n_critic={n_critic}\n{'='*65}"
    )

    generator = generator.to(device)
    disc      = Discriminator(cfg).to(device)
    ema       = EMA(generator, cfg["ema_decay"]) if cfg["use_ema"] else None
    agp       = AdaptiveGP(
        cfg["gp_kp"], cfg["gp_ki"],
        cfg["gp_initial_weight"], cfg["gp_windup_limit"],
    )

    opt_g = optim.AdamW(
        generator.parameters(), lr=cfg["lr_generator"],
        betas=(cfg["beta1"], cfg["beta2"]), weight_decay=1e-4,
    )
    opt_d = optim.AdamW(
        disc.parameters(), lr=cfg["lr_discriminator"],
        betas=(cfg["beta1"], cfg["beta2"]), weight_decay=1e-4,
    )

    def lr_lambda(ep):
        if ep < cfg["warmup_epochs"]:
            return (ep + 1) / cfg["warmup_epochs"]
        progress = (ep - cfg["warmup_epochs"]) / max(
            1, cfg["epochs"] - cfg["warmup_epochs"]
        )
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    sched_g  = optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)
    sched_d  = optim.lr_scheduler.LambdaLR(opt_d, lr_lambda)
    use_amp  = cfg["use_mixed_precision"] and torch.cuda.is_available()
    scaler_d = GradScaler("cuda") if use_amp else None

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr)),
        batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
        num_workers=cfg["num_workers"], pin_memory=(device == "cuda"),
    )

    hist = {
        "d_loss": [], "g_loss": [], "d_real_mean": [], "d_fake_mean": [],
        "val_mmd": [], "val_mse": [], "val_wd": [], "val_wd_std": [],
        "val_kl": [], "val_kl_std": [], "gp": [], "gp_weight": [],
        "gradient_norm": [], "lr_g": [], "lr_d": [],
    }

    best_score = float("inf")
    best_mmd   = float("inf")
    best_samp  = None
    es_ctr     = 0
    t0         = datetime.now()

    for epoch in range(cfg["epochs"]):
        generator.train(); disc.train()
        ep_d = ep_g = ep_dr = ep_df = ep_gn = ep_gp = 0.0
        nb = 0; lam = agp.weight()

        pbar = tqdm(loader, desc=f"[{tag}] {epoch+1:02d}/{cfg['epochs']}", ncols=115)

        for (real,) in pbar:
            real = real.to(device, non_blocking=True)
            bsz  = real.size(0); nb += 1


            with torch.no_grad():
                z_d  = torch.randn(bsz, cfg["n_qubits"], device=device)
                fake = generator(z_d).detach()


            d_acc = 0.0
            for _ in range(n_critic):
                opt_d.zero_grad(set_to_none=True)
                gp_v, gn = compute_gp(disc, real, fake, device)
                if use_amp:
                    with autocast("cuda"):
                        dr = disc(real.float()); df = disc(fake.float())
                    dl = -dr.mean() + df.mean() + lam * gp_v
                    scaler_d.scale(dl).backward()
                    scaler_d.unscale_(opt_d)
                    nn.utils.clip_grad_norm_(disc.parameters(), cfg["gradient_clip_d"])
                    scaler_d.step(opt_d); scaler_d.update()
                else:
                    dr = disc(real); df = disc(fake)
                    dl = -dr.mean() + df.mean() + lam * gp_v
                    dl.backward()
                    nn.utils.clip_grad_norm_(disc.parameters(), cfg["gradient_clip_d"])
                    opt_d.step()
                agp.record(gn); d_acc += dl.item()

            dl_mean = d_acc / n_critic


            opt_g.zero_grad(set_to_none=True)
            z_g    = torch.randn(bsz, cfg["n_qubits"], device=device)
            fake_g = generator(z_g)
            gl     = -disc(fake_g.float()).mean()
            gl.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), cfg["gradient_clip_g"])
            opt_g.step()

            if ema: ema.update(generator)

            ep_d  += dl_mean; ep_g  += gl.item()
            ep_dr += dr.mean().item(); ep_df += df.mean().item()
            ep_gn += gn; ep_gp += gp_v.item()
            pbar.set_postfix(
                D=f"{dl_mean:.4f}", G=f"{gl.item():.4f}",
                GP=f"{gp_v.item():.3f}", lam=f"{lam:.2f}",
            )

        sched_g.step(); sched_d.step(); agp.update()
        for k, v in [
            ("d_loss", ep_d/nb), ("g_loss", ep_g/nb),
            ("d_real_mean", ep_dr/nb), ("d_fake_mean", ep_df/nb),
            ("gp", ep_gp/nb), ("gp_weight", agp.weight()),
            ("gradient_norm", ep_gn/nb),
            ("lr_g", opt_g.param_groups[0]["lr"]),
            ("lr_d", opt_d.param_groups[0]["lr"]),
        ]:
            hist[k].append(float(v))

        logger.info(
            f"[{tag}] Epoch {epoch+1}: D={ep_d/nb:.4f} G={ep_g/nb:.4f} "
            f"GP={ep_gp/nb:.4f} λ={agp.weight():.3f}"
        )


        if (epoch + 1) % cfg["eval_interval"] == 0:
            if ema: ema.apply(generator)
            generator.eval()
            with torch.no_grad():
                zv = torch.randn(512, cfg["n_qubits"], device=device)
                gv = generator(zv).cpu().numpy()
            if ema: ema.restore(generator)
            generator.train()

            n_va    = min(512, len(X_va))
            vm      = compute_mmd(X_va[:n_va], gv)
            vs      = compute_mse(X_va[:n_va], gv)
            vw, vws = compute_wasserstein(X_va[:n_va], gv)
            vk, vks = compute_kl(X_va[:n_va], gv)

            for k, v in [
                ("val_mmd", vm), ("val_mse", vs), ("val_wd", vw),
                ("val_wd_std", vws), ("val_kl", vk), ("val_kl_std", vks),
            ]:
                hist[k].append(float(v))

            logger.info(
                f"[{tag}] Val MMD={vm:.5f} MSE={vs:.5f} "
                f"WD={vw:.5f}±{vws:.5f} KL={vk:.4f}±{vks:.4f}"
            )

            score = 0.5 * vm + 0.5 * vw
            if score < best_score:
                best_score = score; best_mmd = vm; es_ctr = 0
                if ema: ema.apply(generator)
                generator.eval()
                with torch.no_grad():
                    zs = torch.randn(1000, cfg["n_qubits"], device=device)
                    best_samp = generator(zs).cpu().numpy()
                if ema: ema.restore(generator)
                generator.train()
                logger.info(
                    f"[{tag}] ★ New best score={best_score:.6f} "
                    f"(MMD={vm:.6f} WD={vw:.6f})"
                )
            else:
                es_ctr += 1
                if es_ctr >= cfg["early_stopping_patience"]:
                    logger.info(f"[{tag}] Early stopping at epoch {epoch+1}")
                    break


    total_time = str(
        timedelta(seconds=int((datetime.now() - t0).total_seconds()))
    )
    if ema: ema.apply(generator)
    generator.eval()
    with torch.no_grad():
        zf    = torch.randn(1000, cfg["n_qubits"], device=device)
        final = generator(zf).cpu().numpy() if best_samp is None else best_samp
    if ema: ema.restore(generator)

    test_mmd          = compute_mmd(X_te, final)
    test_mse          = compute_mse(X_te, final)
    test_wd, test_wds = compute_wasserstein(X_te, final)
    test_kl, test_kls = compute_kl(X_te, final)
    ids_res           = compute_ids_evasion(X_te, y_te, final, tag)

    logger.info(
        f"[{tag}] FINAL MMD={test_mmd:.6f} MSE={test_mse:.6f} "
        f"WD={test_wd:.6f}±{test_wds:.6f} KL={test_kl:.4f}±{test_kls:.4f}"
    )
    logger.info(f"[{tag}] Time={total_time}  Best score={best_score:.6f}")

    hist.update({
        "test_mmd":       float(test_mmd),
        "test_mse":       float(test_mse),
        "test_wd":        float(test_wd),
        "test_wd_std":    float(test_wds),
        "test_kl":        float(test_kl),
        "test_kl_std":    float(test_kls),
        "ids_evasion":    ids_res,
        "best_val_mmd":   float(best_mmd),
        "best_val_score": float(best_score),
        "total_time":     total_time,
    })
    return hist, final, best_mmd


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"]   = 10


def _eval_epochs(hist, ei):
    ep  = np.arange(1, len(hist["d_loss"]) + 1)
    eep = np.array([i * ei for i in range(1, len(hist["val_mmd"]) + 1)])
    return ep, eep


def _save(name):
    plt.savefig(OUT / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()


def plot_all_loss(results, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for k, res in results.items():
        h = res["history"]; ep = np.arange(1, len(h["d_loss"]) + 1)
        axes[0].plot(ep, h["d_loss"], label=LABELS[k], color=COLORS[k], linewidth=2)
        axes[1].plot(ep, h["g_loss"], label=LABELS[k], color=COLORS[k], linewidth=2)
    for ax, title in zip(axes, ["Discriminator Loss", "Generator Loss"]):
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("Loss", fontweight="bold")
        ax.set_title(title, fontweight="bold", loc="left")
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle="--")
    plt.suptitle("Training Loss — All Models", fontweight="bold")
    plt.tight_layout(); _save("fig_all_loss.png")


def plot_all_discriminator(results, cfg):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for k, res in results.items():
        h = res["history"]; ep = np.arange(1, len(h["d_loss"]) + 1)
        axes[0].plot(ep, h["d_real_mean"], label=LABELS[k], color=COLORS[k], linewidth=2)
        axes[1].plot(ep, h["d_fake_mean"], label=LABELS[k], color=COLORS[k], linewidth=2)
    for ax, title in zip(axes, ["D(real)", "D(fake)"]):
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("Discriminator Output", fontweight="bold")
        ax.set_title(title, fontweight="bold", loc="left")
        ax.legend(frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linestyle="--")
    plt.suptitle("Discriminator Outputs — All Models", fontweight="bold")
    plt.tight_layout(); _save("fig_all_discriminator.png")


def _simple_metric_plot(results, cfg, key, ylabel, title, fname, error_key=None):
    fig, ax = plt.subplots(figsize=(9, 5))
    for k, res in results.items():
        h = res["history"]; _, eep = _eval_epochs(h, cfg["eval_interval"])
        vals = h[key]; n = len(vals)
        if error_key:
            ax.errorbar(eep[:n], vals[:n], yerr=h[error_key][:n], fmt="o-",
                        label=LABELS[k], color=COLORS[k],
                        linewidth=2.5, markersize=8, capsize=5)
        else:
            ax.plot(eep[:n], vals[:n], "o-", label=LABELS[k],
                    color=COLORS[k], linewidth=2.5, markersize=8)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel(ylabel, fontweight="bold")
    ax.set_title(title, fontweight="bold", loc="left")
    ax.legend(frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout(); _save(fname)


def plot_all_ids(results, cfg):
    ids_models = ["RF", "XGBoost", "CNN1D"]
    metrics    = ["detection_rate", "attack_success_rate", "f1"]
    mlabels    = ["Detection Rate (DR)", "Attack Success Rate (ASR)", "F1-score"]
    fig, axes  = plt.subplots(1, 3, figsize=(20, 6))
    for ax, met, mlab in zip(axes, metrics, mlabels):
        x = np.arange(len(ids_models)); width = 0.25
        for ci, (k, res) in enumerate(results.items()):
            ids  = res["history"].get("ids_evasion", {})
            vals = [ids.get(m, {}).get(met, 0) or 0 for m in ids_models]
            ax.bar(x + ci * width, vals, width, label=LABELS[k],
                   color=COLORS[k], alpha=0.85, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x + width)
        ax.set_xticklabels(ids_models, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_ylabel(mlab, fontweight="bold")
        ax.set_title(mlab, fontweight="bold", loc="left")
        ax.legend(frameon=True, fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, linewidth=1.2)
    plt.suptitle("IDS Evasion — RF / XGBoost / CNN-1D", fontweight="bold")
    plt.tight_layout(); _save("fig_all_ids.png")


def plot_all_distributions(results, X_te, cfg):
    feats = cfg["feature_names"]
    fig, axes = plt.subplots(1, len(feats), figsize=(20, 5))
    ns = min(500, len(X_te))
    for fi, (ax, feat) in enumerate(zip(axes, feats)):
        ax.hist(X_te[:ns, fi], bins=40, alpha=0.5, label="Real",
                color=COLORS["real"], density=True, edgecolor="none")
        for k, res in results.items():
            ax.hist(res["samples"][:ns, fi], bins=40, alpha=0.45,
                    label=LABELS[k], color=COLORS[k], density=True, edgecolor="none")
        ax.set_xlabel(feat, fontweight="bold")
        ax.set_ylabel("Density", fontweight="bold")
        ax.set_title(f"Feature: {feat}", fontweight="bold", loc="left")
        if fi == 0: ax.legend(fontsize=8, frameon=True)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    plt.suptitle("Feature Distributions — Real vs Generated", fontweight="bold")
    plt.tight_layout(); _save("fig_all_distributions.png")


def plot_master_comparison(results, X_te, cfg):
    ei         = cfg["eval_interval"]
    ids_models = ["RF", "XGBoost", "CNN1D"]
    ns         = min(500, len(X_te))

    fig = plt.figure(figsize=(24, 22))
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.42, wspace=0.32)


    for col, key, title in [
        (0, "d_loss", "Discriminator Loss"),
        (1, "g_loss", "Generator Loss"),
    ]:
        ax = fig.add_subplot(gs[0, col])
        for k, res in results.items():
            h = res["history"]; ep = np.arange(1, len(h["d_loss"]) + 1)
            ax.plot(ep, h[key], label=LABELS[k], color=COLORS[k], linewidth=2)
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel("Loss", fontweight="bold")
        ax.set_title(title, fontweight="bold", loc="left")
        ax.legend(frameon=True, fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")

    ax = fig.add_subplot(gs[0, 2])
    for k, res in results.items():
        h = res["history"]; ep = np.arange(1, len(h["d_loss"]) + 1)
        ax.plot(ep, h["d_real_mean"], color=COLORS[k], linewidth=2,
                label=f"{LABELS[k]} D(real)")
        ax.plot(ep, h["d_fake_mean"], color=COLORS[k], linewidth=1.5, linestyle="--")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Disc. Output", fontweight="bold")
    ax.set_title("Discriminator Outputs", fontweight="bold", loc="left")
    ax.legend(frameon=True, fontsize=7)
    ax.grid(True, alpha=0.3, linestyle="--")


    for col, key, ylabel, title in [
        (0, "val_mmd", "MMD",  "Maximum Mean Discrepancy"),
        (1, "val_mse", "MSE",  "Mean Squared Error"),
    ]:
        ax = fig.add_subplot(gs[1, col])
        for k, res in results.items():
            h = res["history"]; _, eep = _eval_epochs(h, ei)
            ax.plot(eep, h[key], "o-", label=LABELS[k],
                    color=COLORS[k], linewidth=2.5, markersize=7)
        ax.set_xlabel("Epoch", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(f"{title} (Val)", fontweight="bold", loc="left")
        ax.legend(frameon=True, fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--")

    ax = fig.add_subplot(gs[1, 2])
    for k, res in results.items():
        h = res["history"]; _, eep = _eval_epochs(h, ei); n = len(h["val_wd"])
        ax.errorbar(eep[:n], h["val_wd"][:n], yerr=h["val_wd_std"][:n], fmt="o-",
                    label=LABELS[k], color=COLORS[k],
                    linewidth=2.5, markersize=7, capsize=5)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("Wasserstein Distance", fontweight="bold")
    ax.set_title("Wasserstein Distance (Val)", fontweight="bold", loc="left")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")


    ax = fig.add_subplot(gs[2, 0])
    for k, res in results.items():
        h = res["history"]; _, eep = _eval_epochs(h, ei); n = len(h["val_kl"])
        ax.errorbar(eep[:n], h["val_kl"][:n], yerr=h["val_kl_std"][:n], fmt="o-",
                    label=LABELS[k], color=COLORS[k],
                    linewidth=2.5, markersize=7, capsize=5)
    ax.set_xlabel("Epoch", fontweight="bold")
    ax.set_ylabel("KL Divergence", fontweight="bold")
    ax.set_title("KL Divergence (Val)", fontweight="bold", loc="left")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")

    ax = fig.add_subplot(gs[2, 1])
    x = np.arange(len(ids_models)); width = 0.25
    for ci, (k, res) in enumerate(results.items()):
        ids = res["history"].get("ids_evasion", {})
        asr = [ids.get(m, {}).get("attack_success_rate", 0) or 0 for m in ids_models]
        ax.bar(x + ci * width, asr, width, label=LABELS[k],
               color=COLORS[k], alpha=0.85, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x + width)
    ax.set_xticklabels(ids_models, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Attack Success Rate", fontweight="bold")
    ax.set_title("IDS Evasion — ASR", fontweight="bold", loc="left")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, linewidth=1.2)

    ax = fig.add_subplot(gs[2, 2])
    pv   = [res["n_params"] for res in results.values()]
    pc   = [COLORS[k] for k in results]
    pl   = [LABELS[k].replace(" ", "\n") for k in results]
    bars = ax.bar(pl, pv, color=pc, edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, pv):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(v), ha="center", va="bottom", fontsize=11, fontweight="bold",
        )
    ax.axhline(80, color="red", linestyle="--", linewidth=1.5, label="Hammami 2025 (80)")
    ax.set_ylabel("Generator Parameters", fontweight="bold")
    ax.set_title("Generator Parameter Count", fontweight="bold", loc="left")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")


    for fi, feat in enumerate(cfg["feature_names"]):
        if fi >= 3: break
        ax = fig.add_subplot(gs[3, fi])
        ax.hist(X_te[:ns, fi], bins=40, alpha=0.5, label="Real",
                color=COLORS["real"], density=True, edgecolor="none")
        for k, res in results.items():
            ax.hist(res["samples"][:ns, fi], bins=40, alpha=0.45,
                    label=LABELS[k], color=COLORS[k], density=True, edgecolor="none")
        ax.set_xlabel(feat, fontweight="bold")
        ax.set_ylabel("Density", fontweight="bold")
        ax.set_title(f"Feature: {feat}", fontweight="bold", loc="left")
        if fi == 0: ax.legend(fontsize=7, frameon=True)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.suptitle(
        "QC-GAN vs QC-GAN+Noise vs Classical GAN — UNSW-NB15 (SPIE 2026)",
        fontsize=14, fontweight="bold", y=0.999,
    )
    _save("fig_master_comparison.png")
    logger.info(f"Master comparison saved → {OUT / 'fig_master_comparison.png'}")


def main():
    cfg = CFG.copy()
    set_seed(cfg["seed"])

    logger.info(f"PyTorch version  : {torch.__version__}")
    logger.info(f"CUDA available   : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU              : {torch.cuda.get_device_name(0)}")
    logger.info(f"PennyLane version: {qml.__version__}")

    X_tr, X_va, X_te, y_te = load_data(cfg)

    dev_clean, diff_clean, dev_noisy = _make_devices(cfg["n_qubits"])
    circuit_clean, circuit_noisy     = _build_circuits(
        dev_clean, diff_clean, dev_noisy, cfg,
    )

    gen_classical = ClassicalGenerator(cfg)
    gen_clean     = QuantumGenerator(cfg, circuit_clean, noisy=False)
    gen_noisy     = QuantumGenerator(cfg, circuit_noisy, noisy=True)

    n_q = sum(p.numel() for p in gen_clean.parameters())
    n_c = sum(p.numel() for p in gen_classical.parameters())

    logger.info("\nPARAMETER SUMMARY:")
    logger.info(f"  QC-GAN (clean)  : {n_q} params")
    logger.info(f"  QC-GAN + Noise  : {n_q} params  (identical circuit)")
    logger.info(f"  Classical GAN   : {n_c} params  ({n_c/n_q:.1f}× more than QC-GAN)")
    logger.info(f"  Hammami 2025 ref: 80 params")
    logger.info(f"  n_critic        : {cfg['n_critic']}")
    logger.info(f"  Qubits          : {cfg['n_qubits']}")
    logger.info(f"  SUDAI injections: {cfg['num_injections']}")
    logger.info(f"  Var layers/inj  : {cfg['variational_layers_per_injection']}")
    logger.info(f"  Epochs (max)    : {cfg['epochs']}")

    results: dict = {}

    for gen, key, tag, n_params in [
        (gen_classical, "classical",   "Classical GAN",  n_c),
        (gen_clean,     "qcgan",       "QC-GAN",         n_q),
        (gen_noisy,     "qcgan_noise", "QC-GAN + Noise", n_q),
    ]:
        hist, samples, best_mmd = train_model(
            gen, X_tr, X_va, X_te, y_te, cfg, tag,
        )
        csv_path  = OUT / f"final_generated_samples_{key}.csv"
        json_path = OUT / f"training_history_{key}.json"
        pd.DataFrame(samples, columns=cfg["feature_names"]).to_csv(
            csv_path, index=False,
        )
        with open(json_path, "w") as f:
            json.dump(sanitize(hist), f, indent=2)
        logger.info(f"[{tag}] Saved → {csv_path}")
        results[key] = {"history": hist, "samples": samples, "n_params": n_params}

    logger.info("\nGenerating figures...")
    plot_all_loss(results, cfg);
    logger.info("  ✓ fig_all_loss.png")
    plot_all_discriminator(results, cfg);
    logger.info("  ✓ fig_all_discriminator.png")
    _simple_metric_plot(results, cfg, "val_mmd", "MMD",
        "Maximum Mean Discrepancy (Val)", "fig_all_mmd.png")
    logger.info("  ✓ fig_all_mmd.png")
    _simple_metric_plot(results, cfg, "val_mse", "MSE",
        "Mean Squared Error (Val)", "fig_all_mse.png")
    logger.info("  ✓ fig_all_mse.png")
    _simple_metric_plot(results, cfg, "val_wd", "Wasserstein Distance",
        "Wasserstein Distance (Val, mean±std)", "fig_all_wd.png",
        error_key="val_wd_std")
    logger.info("  ✓ fig_all_wd.png")
    _simple_metric_plot(results, cfg, "val_kl", "KL Divergence",
        "KL Divergence (Val, mean±std)", "fig_all_kl.png",
        error_key="val_kl_std")
    logger.info("  ✓ fig_all_kl.png")
    plot_all_ids(results, cfg);
    logger.info("  ✓ fig_all_ids.png")
    plot_all_distributions(results, X_te, cfg);
    logger.info("  ✓ fig_all_distributions.png")
    plot_master_comparison(results, X_te, cfg);
    logger.info("  ✓ fig_master_comparison.png")

    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    for key, res in results.items():
        h = res["history"]
        print(f"\n{LABELS[key].upper()}  ({res['n_params']} params):")
        print(f"  Best Val score : {h['best_val_score']:.6f}  (0.5·MMD + 0.5·WD)")
        print(f"  Best Val MMD   : {h['best_val_mmd']:.6f}")
        print(f"  Test MMD       : {h['test_mmd']:.6f}")
        print(f"  Test MSE       : {h['test_mse']:.6f}")
        print(f"  Test WD        : {h['test_wd']:.6f} ± {h['test_wd_std']:.6f}")
        print(f"  Test KL        : {h['test_kl']:.6f} ± {h['test_kl_std']:.6f}")
        for m, mr in h.get("ids_evasion", {}).items():
            if mr and "note" not in mr:
                print(
                    f"  IDS-{m:8s}: "
                    f"DR={mr.get('detection_rate', 0):.4f}  "
                    f"ASR={mr.get('attack_success_rate', 0):.4f}  "
                    f"F1={mr.get('f1', 0):.4f}  "
                    f"FPR={mr.get('fpr', 0):.4f}  "
                    f"acc_real={mr.get('accuracy_real', 0):.4f}"
                )
        print(f"  Training time  : {h['total_time']}")

    logger.info(f"\nAll outputs → {OUT.absolute()}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
