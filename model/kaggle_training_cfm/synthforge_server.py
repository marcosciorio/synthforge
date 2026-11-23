#!/usr/bin/env python3
"""Inference server for the SynthForge plugin (CLAP, text adapter and CFM model).

Local TCP server on 127.0.0.1:51027 with a line-based JSON protocol:
one JSON request per line, one JSON response per line.
"""

import json
import math
import os
import re
import socketserver
import sys
import threading

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

if not hasattr(np, "_core"):
    import numpy.core
    sys.modules["numpy._core"] = numpy.core
    for _sub in ("multiarray", "numeric", "umath", "_multiarray_umath"):
        if hasattr(numpy.core, _sub):
            sys.modules[f"numpy._core.{_sub}"] = getattr(numpy.core, _sub)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(SCRIPT_DIR, "cfm_final.pt")
CLAP_CKPT = os.path.join(SCRIPT_DIR, "music_audioset_epoch_15_esc_90.14.pt")
ADAPTER_PATH = os.path.join(SCRIPT_DIR, "adapter_aug_space.pt")
EMB_NPZ = os.path.join(SCRIPT_DIR, "adapter_embeddings.npz")
JSON_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "Dataset_TAL_48kHz", "not_in_pool", "json")
CAPTION_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "Dataset_TAL_48kHz", "not_in_pool", "captions")

HOST, PORT = "127.0.0.1", 51027
CLAP_SR = 48000

DEFAULT_STEPS = 50
DEFAULT_S_DIRETTA = 1.0
DEFAULT_S_ADAPTER = 2.0
DEFAULT_S_RETRIEVAL = 3.0
DEFAULT_T_STAR = 0.5
DEFAULT_TAU = None
DEFAULT_S_MEMORY = DEFAULT_S_RETRIEVAL
DEFAULT_S_AUDIO = DEFAULT_S_ADAPTER

CATEGORY_VOCAB = ["arp", "bass", "chord", "drum", "fx", "keyboard", "lead", "pad", "pluck"]

_mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
DEVICE = "mps" if _mps_ok else ("cuda" if torch.cuda.is_available() else "cpu")

CATEGORICAL = {
    "osc1waveform":      list(np.round(np.linspace(0, 1, 3), 7)),
    "osc2waveform":      list(np.round(np.linspace(0, 1, 5), 7)),
    "filtertype":        list(np.round(np.linspace(0, 1, 12), 7)),
    "lfo1destination":   list(np.round(np.linspace(0, 1, 8), 7)),
    "lfo2destination":   list(np.round(np.linspace(0, 1, 8), 7)),
    "lfo1waveform":      list(np.round(np.linspace(0, 1, 6), 7)),
    "lfo2waveform":      list(np.round(np.linspace(0, 1, 6), 7)),
    "freeaddestination": list(np.round(np.linspace(0, 1, 6), 7)),
}
ORDINAL_49 = ["osc1tune", "osc2tune"]
BINARY = ["lfo1keytrigger", "lfo2keytrigger", "oscsync",
          "chorus1enable", "chorus2enable", "delaysync", "delayx2l", "delayx2r"]
BIPOLAR_NEUTRAL = {
    "lfo1amount", "lfo2amount", "osc1finetune", "osc2finetune",
    "osc1pw", "filtercontour", "freeadamount", "masterdetune",
}


class Param:
    def __init__(self, name, kind, **kw):
        self.name = name
        self.kind = kind
        self.__dict__.update(kw)

    def __len__(self):
        return len(self.values) if self.kind == "cat" else 1

    def decode(self, seg):
        if self.kind in ("cont", "bipolar"):
            return float(np.clip((seg[0] + 1.0) / 2.0, 0, 1))
        if self.kind == "bin":
            return 1.0 if seg[0] >= 0.0 else 0.0
        if self.kind == "ord49":
            v = float(np.clip((seg[0] + 1.0) / 2.0, 0, 1))
            return round(int(round(v * 48)) / 48, 7)
        return self.values[int(np.argmax(seg))]


class ParamSpec:
    def __init__(self, param_names):
        self.params = []
        for n in param_names:
            if n in CATEGORICAL:
                self.params.append(Param(n, "cat", values=CATEGORICAL[n]))
            elif n in ORDINAL_49:
                self.params.append(Param(n, "ord49"))
            elif n in BINARY:
                self.params.append(Param(n, "bin"))
            elif n in BIPOLAR_NEUTRAL:
                self.params.append(Param(n, "bipolar"))
            else:
                self.params.append(Param(n, "cont"))
        self.layout = []
        i = 0
        for p in self.params:
            k = len(p)
            self.layout.append((p, slice(i, i + k)))
            i += k
        self.dim = i

    def decode(self, x):
        return {p.name: p.decode(x[sl]) for p, sl in self.layout}


class TextAdapter(nn.Module):

    def __init__(self, dim: int = 512):
        super().__init__()
        self.W = nn.Parameter(torch.eye(dim))
        self.b = nn.Parameter(torch.zeros(dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return F.normalize(t @ self.W.T + self.b, dim=-1)


class TimeEmbed(nn.Module):
    def __init__(self, d_cond, n_freq=128, max_freq=1000.0):
        super().__init__()
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(max_freq), n_freq))
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq, d_cond), nn.SiLU(), nn.Linear(d_cond, d_cond))

    def forward(self, t):
        ang = 2 * math.pi * t * self.freqs[None]
        return self.mlp(torch.cat([ang.sin(), ang.cos()], -1))


class DiTBlock(nn.Module):
    def __init__(self, d, n_heads, d_cond):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_cond, 6 * d))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, z):
        sa, ga, ha, sf, gf, hf = self.ada(z).unsqueeze(1).chunk(6, -1)
        h = self.n1(x) * (1 + ga) + sa
        a, _ = self.attn(h, h, h)
        x = x + ha * a
        h = self.n2(x) * (1 + gf) + sf
        return x + hf * self.ff(h)


class FinalLayer(nn.Module):
    def __init__(self, d, d_cond):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_cond, 2 * d))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, z):
        s, g = self.ada(z).unsqueeze(1).chunk(2, -1)
        return self.norm(x) * (1 + g) + s


class VectorField(nn.Module):
    def __init__(self, spec, d_token, n_heads, n_layers, clap_dim, d_cond, n_freq_time=128):
        super().__init__()
        self.spec, self.d_token = spec, d_token
        self.in_p = nn.ModuleList([nn.Linear(len(p), d_token) for p, _ in spec.layout])
        self.out_p = nn.ModuleList([nn.Linear(d_token, len(p)) for p, _ in spec.layout])
        for lin in self.out_p:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.p_emb = nn.Embedding(len(spec.params), d_token)
        self.t_emb = TimeEmbed(d_cond, n_freq_time)
        self.clap_mlp = nn.Sequential(nn.Linear(clap_dim, d_cond), nn.SiLU(), nn.Linear(d_cond, d_cond))
        self.cfg_token = nn.Parameter(torch.randn(1, clap_dim) / math.sqrt(clap_dim))
        self.blocks = nn.ModuleList([DiTBlock(d_token, n_heads, d_cond) for _ in range(n_layers)])
        self.final = FinalLayer(d_token, d_cond)

    def _tokenize(self, x):
        toks = torch.stack([self.in_p[i](x[:, sl]) for i, (_, sl) in enumerate(self.spec.layout)], 1)
        return toks + self.p_emb(torch.arange(len(self.spec.params), device=x.device))

    def forward(self, x, t, cond):
        B = x.shape[0]
        if cond is None:
            cond = self.cfg_token.expand(B, -1)
        z = self.t_emb(t) + self.clap_mlp(cond)
        toks = self._tokenize(x)
        for blk in self.blocks:
            toks = blk(toks, z)
        toks = self.final(toks, z)
        return torch.cat([self.out_p[i](toks[:, i]) for i in range(len(self.spec.layout))], 1)


def _call_cfg(vf, x, t, cond, s):
    if s == 1.0 or cond is None:
        return vf(x, t, cond)
    return (1 - s) * vf(x, t, None) + s * vf(x, t, cond)


def _rk4(vf, x, t, dt, cond, s):
    f = lambda xx, tt: _call_cfg(vf, xx, tt, cond, s)
    k1 = f(x, t)
    k2 = f(x + dt * k1 / 2, t + dt / 2)
    k3 = f(x + dt * k2 / 2, t + dt / 2)
    k4 = f(x + dt * k3, t + dt)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


@torch.no_grad()
def sample_cfm(vf, cond, n_dim, steps=50, s=2.0):
    dev = next(vf.parameters()).device
    x = torch.randn(cond.shape[0], n_dim, device=dev)
    t = torch.zeros(cond.shape[0], 1, device=dev)
    dt = 1.0 / steps
    for _ in range(steps):
        x = _rk4(vf, x, t, dt, cond, s)
        t = t + dt
    return x


@torch.no_grad()
def sample_from_t(vf, x1, cond, t_star, steps=50, s=2.0):
    dev = next(vf.parameters()).device
    t_star = float(np.clip(t_star, 0.01, 0.99))
    noise = torch.randn_like(x1)
    x = (1.0 - t_star) * noise + t_star * x1
    dt = 1.0 / steps
    remaining = max(1, int(round(steps * (1.0 - t_star))))
    t = torch.full((x1.shape[0], 1), t_star, device=dev)
    for _ in range(remaining):
        x = _rk4(vf, x, t, dt, cond, s)
        t = t + dt
    return x


def encode_preset(spec, params: dict) -> np.ndarray:
    x = np.zeros(spec.dim, dtype=np.float32)
    for p, sl in spec.layout:
        v = float(params.get(p.name, 0.5))
        if p.kind in ("cont", "bipolar", "ord49"):
            x[sl] = np.array([2.0 * float(np.clip(v, 0.0, 1.0)) - 1.0])
        elif p.kind == "bin":
            x[sl] = np.array([1.0 if v >= 0.5 else -1.0])
        elif p.kind == "cat":
            seg = np.full(len(p), -1.0, dtype=np.float32)
            diffs = [abs(v - cv) for cv in p.values]
            seg[int(np.argmin(diffs))] = 1.0
            x[sl] = seg
    return x


def _int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def _float32_to_int16(x):
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def load_audio_48k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != CLAP_SR:
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=CLAP_SR)
        except ImportError:
            n = int(len(audio) * CLAP_SR / sr)
            audio = np.interp(np.linspace(0, len(audio) - 1, n),
                               np.arange(len(audio)), audio)
    audio = _int16_to_float32(_float32_to_int16(audio))
    return audio.reshape(1, -1).astype(np.float32)


def l2norm_np(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def get_audio_embedding(path: str) -> np.ndarray:
    with torch.no_grad():
        raw = load_audio_48k(path)
        emb = CLAP_MODEL.get_audio_embedding_from_data(x=raw, use_tensor=False)[0]
    return l2norm_np(emb.astype(np.float32))


_EPS = 0.05


def enforce_audio_logic(params):
    def g(k, d=0.0):
        return float(params.get(k, d))

    if g("lfo1destination") == 0.0:
        for p in ["lfo1waveform", "lfo1rate", "lfo1phase", "lfo1keytrigger"]:
            if p in params:
                params[p] = 0.0
        if "lfo1amount" in params:
            params["lfo1amount"] = 0.5
    if abs(g("lfo1amount") - 0.5) < _EPS:
        for p in ["lfo1waveform", "lfo1rate", "lfo1phase", "lfo1keytrigger"]:
            if p in params:
                params[p] = 0.0
        if "lfo1destination" in params:
            params["lfo1destination"] = 0.0
    if g("lfo2destination") == 0.0:
        for p in ["lfo2waveform", "lfo2rate", "lfo2phase", "lfo2keytrigger"]:
            if p in params:
                params[p] = 0.0
        if "lfo2amount" in params:
            params["lfo2amount"] = 0.5
    if abs(g("lfo2amount") - 0.5) < _EPS:
        for p in ["lfo2waveform", "lfo2rate", "lfo2phase", "lfo2keytrigger"]:
            if p in params:
                params[p] = 0.0
        if "lfo2destination" in params:
            params["lfo2destination"] = 0.0
    if g("osc1waveform") != 0.5:
        if "osc1pw" in params:
            params["osc1pw"] = 0.5
    if g("osc2volume") < _EPS and g("osc2fm") < _EPS:
        if "oscsync" in params:
            params["oscsync"] = 0.0
    if g("filtertype") == 1.0:
        for p in ["filtercutoff", "filterresonance", "filterattack", "filterdecay",
                  "filtersustain", "filterrelease", "filterdrive"]:
            if p in params:
                params[p] = 0.0
        if "filtercontour" in params:
            params["filtercontour"] = 0.5
    if abs(g("filtercontour") - 0.5) < _EPS:
        for p in ["filterattack", "filterdecay", "filtersustain", "filterrelease"]:
            if p in params:
                params[p] = 0.0
    if g("freeaddestination") == 0.0:
        for p in ["freeadattack", "freeaddecay"]:
            if p in params:
                params[p] = 0.0
        if "freeadamount" in params:
            params["freeadamount"] = 0.5
    if abs(g("freeadamount") - 0.5) < _EPS:
        for p in ["freeadattack", "freeaddecay"]:
            if p in params:
                params[p] = 0.0
        if "freeaddestination" in params:
            params["freeaddestination"] = 0.0
    if g("reverbwet") < _EPS:
        for p in ["reverbdecay", "reverbpredelay", "reverbhighcut", "reverblowcut"]:
            if p in params:
                params[p] = 0.0
    if g("delaywet") < _EPS:
        for p in ["delaytime", "delaysync", "delayhighshelf", "delaylowshelf",
                  "delayfeedback", "delayx2l", "delayx2r"]:
            if p in params:
                params[p] = 0.0
    return params


_TIMESHAPE_FORCE = {
    "pulsing": {
        "lfo1amount": 0.9, "lfo1rate": 0.6, "lfo1destination": 0.1428571,
    },
    "evolving": {
        "lfo1amount": 0.85, "lfo1rate": 0.15, "lfo1destination": 0.1428571,
        "lfo2amount": 0.0, "lfo2destination": 0.0,
        "delaysync": 0.0,
    },
    "slow": {
        "ampattack": 0.6,
        "lfo1amount": 0.0, "lfo1destination": 0.0,
        "lfo2amount": 0.0, "lfo2destination": 0.0,
        "delaysync": 0.0,
        "freeadamount": 0.0, "freeaddestination": 0.0,
    },
    "long": {
        "ampattack": 0.1, "ampsustain": 0.85, "amprelease": 0.8,
        "lfo1amount": 0.0, "lfo1destination": 0.0,
        "lfo2amount": 0.0, "lfo2destination": 0.0,
        "delaysync": 0.0,
        "freeadamount": 0.0, "freeaddestination": 0.0,
        "filterattack": 0.3,
    },
    "fading": {
        "ampattack": 0.1, "ampdecay": 0.5, "ampsustain": 0.1, "amprelease": 0.45,
        "lfo1amount": 0.0, "lfo1destination": 0.0,
        "lfo2amount": 0.0, "lfo2destination": 0.0,
        "delaysync": 0.0,
        "freeadamount": 0.0, "freeaddestination": 0.0,
    },
    "short": {
        "ampattack": 0.1, "ampsustain": 0.1, "amprelease": 0.1,
        "lfo1amount": 0.0, "lfo1destination": 0.0,
        "lfo2amount": 0.0, "lfo2destination": 0.0,
        "delaysync": 0.0,
        "freeadamount": 0.0, "freeaddestination": 0.0,
    },
}

_SPACE_FORCE = {
    "cavernous": {"reverbwet": 0.6, "reverbdecay": 0.8},
    "echoey":    {"delaywet": 0.4, "delaytime": 0.3, "reverbwet": 0.1},
    "spacious":  {"reverbwet": 0.3, "reverbdecay": 0.3, "delaywet": 0.1},
    "chorused":  {"chorus1enable": 1.0, "reverbwet": 0.1, "delaywet": 0.1},
    "dry":       {"reverbwet": 0.0, "delaywet": 0.0, "chorus1enable": 0.0, "chorus2enable": 0.0},
}


def apply_force(params: dict, force_time_shape=None, force_space=None) -> dict:
    for value, table in ((force_time_shape, _TIMESHAPE_FORCE), (force_space, _SPACE_FORCE)):
        if not value:
            continue
        patch = table.get(str(value).strip().lower())
        if not patch:
            continue
        for k, v in patch.items():
            if k in params:
                params[k] = v
    return params


MODEL_TO_VST3 = {
    "osc1waveform":      "osc_1_waveform",
    "osc1tune":          "osc_1_tune",
    "osc1finetune":      "osc_1_fine_tune",
    "osc1volume":        "osc_1_volume",
    "osc1pw":            "osc_1_pw",
    "osc1phase":         "osc_1_phase",
    "osc2waveform":      "osc_2_waveform",
    "osc2tune":          "osc_2_tune",
    "osc2finetune":      "osc_2_fine_tune",
    "osc2volume":        "osc_2_volume",
    "osc2fm":            "osc_2_fm",
    "osc2phase":         "osc_2_phase",
    "osc3volume":        "osc_3_volume",
    "ringmodulation":    "ringmodulation",
    "oscsync":           "osc_sync",
    "oscbitcrusher":     "osc_bitcrusher",
    "vintagenoise":      "vintage_noise",
    "filtertype":        "filter_type",
    "filtercutoff":      "filter_cutoff",
    "filterresonance":   "filter_resonance",
    "filtercontour":     "filter_contour",
    "filterdrive":       "filter_drive",
    "filterattack":      "filter_attack",
    "filterdecay":       "filter_decay",
    "filtersustain":     "filter_sustain",
    "filterrelease":     "filter_release",
    "ampattack":         "amp_attack",
    "ampdecay":          "amp_decay",
    "ampsustain":        "amp_sustain",
    "amprelease":        "amp_release",
    "lfo1destination":   "lfo_1_destination",
    "lfo1waveform":      "lfo_1_waveform",
    "lfo1rate":          "lfo_1_rate",
    "lfo1amount":        "lfo_1_amount",
    "lfo1phase":         "lfo_1_phase",
    "lfo1keytrigger":    "lfo_1_keytrigger",
    "lfo2destination":   "lfo_2_destination",
    "lfo2waveform":      "lfo_2_waveform",
    "lfo2rate":          "lfo_2_rate",
    "lfo2amount":        "lfo_2_amount",
    "lfo2phase":         "lfo_2_phase",
    "lfo2keytrigger":    "lfo_2_keytrigger",
    "freeaddestination": "free_ad_destination",
    "freeadamount":      "free_ad_amount",
    "freeadattack":      "free_ad_attack",
    "freeaddecay":       "free_ad_decay",
    "reverbwet":         "reverb_wet",
    "reverbdecay":       "reverb_decay",
    "reverbpredelay":    "reverb_pre_delay",
    "reverbhighcut":     "reverb_high_cut",
    "reverblowcut":      "reverb_low_cut",
    "delaywet":          "delay_wet",
    "delaytime":         "delay_time",
    "delaysync":         "delay_sync",
    "delayhighshelf":    "delay_high_shelf",
    "delaylowshelf":     "delay_low_shelf",
    "delayfeedback":     "delay_feedback",
    "delayx2l":          "delay_x2_l",
    "delayx2r":          "delay_x2_r",
    "chorus1enable":     "chorus_1_enable",
    "chorus2enable":     "chorus_2_enable",
    "masterdetune":      "master_detune",
    "masterhighpass":    "master_high_pass",
}

VF = SPEC = CLAP_MODEL = ADAPTER = None
PRESET_TEXT_EMBS = None
PRESET_AUDIO_EMBS = None
PRESET_NAMES = None
PRESET_PARAMS = None
PRESET_CATEGORIES = None


def load_models():
    global VF, SPEC, CLAP_MODEL, ADAPTER
    global PRESET_TEXT_EMBS, PRESET_AUDIO_EMBS, PRESET_NAMES, PRESET_PARAMS, PRESET_CATEGORIES

    print("[..] Loading the CFM model...")
    ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    spec = ParamSpec(ck["param_names"])
    c = ck["cfg"]
    vf = VectorField(spec,
                      c.get("d_token", 256), c.get("n_heads", 8), c.get("n_layers", 8),
                      c.get("clap_dim", 512), c.get("d_cond", 256),
                      c.get("n_freq_time", 128)).to(DEVICE)
    vf.load_state_dict(ck.get("ema", ck["vf"]))
    vf.eval()
    VF, SPEC = vf, spec

    print("[..] Loading CLAP...")
    import laion_clap
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=DEVICE)
    clap.load_ckpt(CLAP_CKPT if os.path.isfile(CLAP_CKPT) else None)
    CLAP_MODEL = clap

    if os.path.isfile(ADAPTER_PATH):
        print("[..] Loading the text adapter...")
        ck = torch.load(ADAPTER_PATH, map_location=DEVICE, weights_only=False)
        adapter = TextAdapter(ck.get("dim", 512)).to(DEVICE)
        adapter.load_state_dict(ck["state_dict"])
        adapter.eval()
        ADAPTER = adapter
        print(f"[OK] Adapter loaded  "
              f"(CV: {ck.get('cv_mean', float('nan')):.4f} +/- "
              f"{ck.get('cv_std', float('nan')):.4f}  "
              f"baseline: {ck.get('baseline', float('nan')):.4f})")
    else:
        print("[--] adapter.pt not found: using raw CLAP text embeddings (no adapter).")

    print("[..] Loading the retrieval presets...")
    if os.path.isfile(EMB_NPZ):
        data = np.load(EMB_NPZ)
        PRESET_TEXT_EMBS = data["T"].astype(np.float32)
        PRESET_AUDIO_EMBS = data["A"].astype(np.float32)
        PRESET_NAMES = [str(n) for n in data["names"]]
        PRESET_PARAMS = {}
        missing = 0
        for name in PRESET_NAMES:
            p = os.path.join(JSON_DIR, name + ".json")
            if os.path.isfile(p):
                with open(p) as f:
                    raw = json.load(f)
                params = raw.get("parameters", raw) if isinstance(raw, dict) else {}
                PRESET_PARAMS[name] = {k: float(v) for k, v in params.items()
                                        if isinstance(v, (int, float))}
            else:
                missing += 1
        PRESET_CATEGORIES = {}
        for name in PRESET_NAMES:
            cp = os.path.join(CAPTION_DIR, name + ".json")
            if os.path.isfile(cp):
                with open(cp) as f:
                    cap = json.load(f)
                PRESET_CATEGORIES[name] = (cap.get("choices") or {}).get("category", "").lower()
        print(f"[OK] Retrieval ready: {len(PRESET_NAMES)} presets ({missing} JSON files missing).")
    else:
        print("[--] adapter_embeddings.npz not found: retrieval disabled.")


def _params_to_vst3(preset: dict) -> dict:
    return {MODEL_TO_VST3[k]: v for k, v in preset.items() if k in MODEL_TO_VST3}


_CATEGORY_RE = {c: re.compile(r"\b" + re.escape(c) + r"\b") for c in CATEGORY_VOCAB}


def category_from_text(text: str) -> str:
    low = (text or "").lower()
    for c in CATEGORY_VOCAB:
        if _CATEGORY_RE[c].search(low):
            return c
    return ""


def text_embedding(prompt: str, use_adapter: bool = True) -> torch.Tensor:
    emb = CLAP_MODEL.get_text_embedding([prompt], use_tensor=True)
    emb = F.normalize(emb.float(), dim=-1).to(DEVICE)
    if use_adapter and ADAPTER is not None:
        with torch.no_grad():
            emb = F.normalize(ADAPTER(emb), dim=-1)
    return emb


def generate(prompt: str, steps: int, cfg: float, force_time_shape=None, force_space=None,
             use_adapter: bool = True):
    emb = text_embedding(prompt, use_adapter)
    x = sample_cfm(VF, emb, SPEC.dim, int(steps), float(cfg))
    x1_np = x.cpu().numpy()[0]
    preset = enforce_audio_logic(SPEC.decode(x1_np))
    preset = apply_force(preset, force_time_shape, force_space)
    return _params_to_vst3(preset), x1_np.tolist()


def retrieve_and_edit(prompt: str, audio_path: str, steps: int, cfg: float,
                       t_star: float, tau, category,
                       force_time_shape=None, force_space=None):
    if PRESET_AUDIO_EMBS is None:
        raise RuntimeError("retrieval unavailable: adapter_embeddings.npz is missing")

    prompt = (prompt or "").strip()
    use_audio_query = bool(audio_path)

    if use_audio_query:
        query_emb = get_audio_embedding(audio_path)
    else:
        if not prompt:
            raise ValueError("a prompt or an audio file is required")
        query_emb = text_embedding(prompt, use_adapter=True).detach().cpu().numpy()[0]
    sims_all = PRESET_AUDIO_EMBS @ query_emb

    if category is None:
        category = category_from_text(prompt)
    category = (category or "").strip().lower()

    ok = np.array([bool(PRESET_PARAMS.get(n)) for n in PRESET_NAMES], dtype=bool)
    sims = np.where(ok, sims_all, -np.inf)
    if category and PRESET_CATEGORIES:
        mask = ok & np.array([PRESET_CATEGORIES.get(n, "") == category for n in PRESET_NAMES], dtype=bool)
        j_star = int(np.argmax(np.where(mask, sims, -np.inf))) if mask.any() else int(np.argmax(sims))
    else:
        j_star = int(np.argmax(sims))

    best_name = PRESET_NAMES[j_star]
    best_sim = float(sims[j_star])

    if prompt:
        cond_emb = text_embedding(prompt, use_adapter=True)
    else:
        cond_emb = torch.from_numpy(query_emb).unsqueeze(0).to(DEVICE)

    if tau is not None and best_sim < float(tau):
        x = sample_cfm(VF, cond_emb, SPEC.dim, int(steps), float(cfg))
        matched_name, matched_sim = None, best_sim
    else:
        if best_name not in PRESET_PARAMS or not PRESET_PARAMS[best_name]:
            raise RuntimeError(f"parameters not found for {best_name}")
        x1_np = encode_preset(SPEC, PRESET_PARAMS[best_name])
        x1_tensor = torch.from_numpy(x1_np).unsqueeze(0).to(DEVICE)
        x = sample_from_t(VF, x1_tensor, cond_emb, float(t_star), int(steps), float(cfg))
        matched_name, matched_sim = best_name, best_sim

    x1_out = x.cpu().numpy()[0]
    preset = enforce_audio_logic(SPEC.decode(x1_out))
    preset = apply_force(preset, force_time_shape, force_space)
    return _params_to_vst3(preset), x1_out.tolist(), matched_name, matched_sim, category


def memory_edit(prompt: str, steps: int, cfg: float, t_star: float, previous_x1,
                 force_time_shape=None, force_space=None):
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("empty prompt")
    if not isinstance(previous_x1, list) or len(previous_x1) != SPEC.dim:
        raise ValueError(f"previous_x1 must have {SPEC.dim} values")

    emb = text_embedding(prompt, use_adapter=True)

    x1_tensor = torch.tensor(previous_x1, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    x = sample_from_t(VF, x1_tensor, emb, float(t_star), int(steps), float(cfg))
    x1_out = x.cpu().numpy()[0]
    preset = enforce_audio_logic(SPEC.decode(x1_out))
    preset = apply_force(preset, force_time_shape, force_space)
    return _params_to_vst3(preset), x1_out.tolist()


def generate_from_audio(audio_path: str, steps: int, cfg: float, force_time_shape=None, force_space=None):
    query_emb = get_audio_embedding(audio_path)
    cond = torch.from_numpy(query_emb).unsqueeze(0).to(DEVICE)
    x = sample_cfm(VF, cond, SPEC.dim, int(steps), float(cfg))
    x1_out = x.cpu().numpy()[0]
    preset = enforce_audio_logic(SPEC.decode(x1_out))
    preset = apply_force(preset, force_time_shape, force_space)
    return _params_to_vst3(preset), x1_out.tolist()


_state_lock = threading.Lock()
_active_clients = 0
_shutdown_timer = None
_SHUTDOWN_GRACE_SECONDS = 5.0


def _cancel_shutdown_timer_locked():
    global _shutdown_timer
    if _shutdown_timer is not None:
        _shutdown_timer.cancel()
        _shutdown_timer = None


def _shutdown_if_still_idle():
    with _state_lock:
        _cancel_shutdown_timer_locked()
        if _active_clients <= 0:
            print("[--] No active plugin instance: shutting down the server.")
            os._exit(0)


def register_client():
    global _active_clients
    with _state_lock:
        _active_clients += 1
        _cancel_shutdown_timer_locked()
        print(f"[..] Plugin instance registered (active: {_active_clients}).")


def unregister_client():
    global _active_clients, _shutdown_timer
    with _state_lock:
        _active_clients = max(0, _active_clients - 1)
        print(f"[..] Plugin instance unregistered (active: {_active_clients}).")
        if _active_clients == 0:
            _cancel_shutdown_timer_locked()
            _shutdown_timer = threading.Timer(_SHUTDOWN_GRACE_SECONDS, _shutdown_if_still_idle)
            _shutdown_timer.daemon = True
            _shutdown_timer.start()


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        for raw_line in self.rfile:
            line = raw_line.strip()
            if not line:
                continue
            try:
                req = json.loads(line.decode("utf-8"))
                action = req.get("action")
                steps = req.get("steps", DEFAULT_STEPS)

                def cfg_or(default):
                    v = req.get("cfg")
                    return default if v is None else v

                force_time_shape = (req.get("force_time_shape") or "").strip() or None
                force_space = (req.get("force_space") or "").strip() or None

                if action == "register":
                    register_client()
                    response = {"ok": True}

                elif action == "unregister":
                    unregister_client()
                    response = {"ok": True}

                elif action == "generate":
                    prompt = str(req.get("prompt", "")).strip()
                    if not prompt:
                        raise ValueError("empty prompt")
                    use_adapter = bool(req.get("use_adapter", True))
                    cfg = cfg_or(DEFAULT_S_ADAPTER if use_adapter else DEFAULT_S_DIRETTA)
                    params, x1 = generate(prompt, steps, cfg, force_time_shape, force_space,
                                          use_adapter=use_adapter)
                    response = {"ok": True, "params": params, "x1": x1,
                                "settings": {"steps": steps, "cfg": cfg, "use_adapter": use_adapter}}

                elif action == "retrieve_and_edit":
                    cfg = cfg_or(DEFAULT_S_RETRIEVAL)
                    t_star = req.get("t_star", DEFAULT_T_STAR)
                    category = req.get("category", None)
                    params, x1, matched_name, sim, cat_used = retrieve_and_edit(
                        req.get("prompt", ""), req.get("audio_path", ""),
                        steps, cfg, t_star, req.get("tau", DEFAULT_TAU),
                        category, force_time_shape, force_space)
                    response = {"ok": True, "params": params, "x1": x1,
                                "matched_preset": matched_name, "similarity": sim,
                                "category": cat_used,
                                "settings": {"steps": steps, "cfg": cfg, "t_star": t_star}}

                elif action == "memory_edit":
                    cfg = cfg_or(DEFAULT_S_MEMORY)
                    params, x1 = memory_edit(
                        req.get("prompt", ""), steps, cfg,
                        req.get("t_star", DEFAULT_T_STAR), req.get("previous_x1"),
                        force_time_shape, force_space)
                    response = {"ok": True, "params": params, "x1": x1}

                elif action == "generate_from_audio":
                    audio_path = str(req.get("audio_path", "")).strip()
                    if not audio_path:
                        raise ValueError("audio_path is missing")
                    cfg = cfg_or(DEFAULT_S_AUDIO)
                    params, x1 = generate_from_audio(audio_path, steps, cfg, force_time_shape, force_space)
                    response = {"ok": True, "params": params, "x1": x1}

                else:
                    response = {"ok": False, "error": f"azione sconosciuta: {action!r}"}
            except Exception as exc:
                response = {"ok": False, "error": str(exc)}
            self.wfile.write((json.dumps(response) + "\n").encode("utf-8"))
            self.wfile.flush()


class Server(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    print("[..] Loading models (this can take a few seconds)...")
    load_models()
    print(f"[OK] Adapter: {os.path.basename(ADAPTER_PATH)}  |  default: steps={DEFAULT_STEPS}, "
          f"s direct={DEFAULT_S_DIRETTA}, s adapter={DEFAULT_S_ADAPTER}, "
          f"s retrieval={DEFAULT_S_RETRIEVAL}, t*={DEFAULT_T_STAR}")
    print(f"[OK] Models ready. Listening on {HOST}:{PORT}")
    with Server((HOST, PORT), Handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
