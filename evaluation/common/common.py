"""Shared code for the evaluation: model and CLAP loading, generation,
TAL-NoiseMaker rendering, metrics and preset pool.
"""

import os, sys, math, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if not hasattr(np, "_core"):
    import numpy.core
    sys.modules["numpy._core"] = numpy.core
    for _sub in ("multiarray", "numeric", "umath", "_multiarray_umath"):
        if hasattr(numpy.core, _sub):
            sys.modules[f"numpy._core.{_sub}"] = getattr(numpy.core, _sub)

import librosa
from scipy.stats import wasserstein_distance


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT       = os.path.dirname(os.path.dirname(SCRIPT_DIR))

CFM_CKPT     = os.path.join(ROOT, "model", "kaggle_training_cfm", "cfm_final.pt")
CLAP_CKPT    = os.path.join(ROOT, "model", "kaggle_training_cfm", "music_audioset_epoch_15_esc_90.14.pt")
ADAPTER_PATH = os.path.join(ROOT, "model", "kaggle_training_cfm", "adapter_aug.pt")
EMB_NPZ      = os.path.join(ROOT, "model", "kaggle_training_cfm", "adapter_embeddings.npz")
VST3_PATH    = "/Library/Audio/Plug-Ins/VST3/TAL-NoiseMaker.vst3"

IN_POOL_DIR  = os.path.join(ROOT, "data", "Dataset_TAL_48kHz", "in_pool")
CAPTION_DIR  = os.path.join(IN_POOL_DIR, "captions")
AUDIO_DIR    = os.path.join(IN_POOL_DIR, "audio")
JSON_DIR     = os.path.join(IN_POOL_DIR, "json")

NOT_IN_POOL_DIR = os.path.join(ROOT, "data", "Dataset_TAL_48kHz", "not_in_pool")
NIP_JSON_DIR    = os.path.join(NOT_IN_POOL_DIR, "json")
NIP_CAPTION_DIR = os.path.join(NOT_IN_POOL_DIR, "captions")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RENDER_SR  = 48000
RENDER_DUR = 4.0
NOTE_DUR   = 2.0
MIDI_NOTE  = 60
MIDI_VEL   = 100

_mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
DEVICE  = "mps" if _mps_ok else ("cuda" if torch.cuda.is_available() else "cpu")

CATEGORY_VOCAB = ["arp", "bass", "chord", "drum", "fx", "keyboard", "lead", "pad", "pluck"]


CATEGORICAL = {
    "osc1waveform":      list(np.round(np.linspace(0,1,3),7)),
    "osc2waveform":      list(np.round(np.linspace(0,1,5),7)),
    "filtertype":        list(np.round(np.linspace(0,1,12),7)),
    "lfo1destination":   list(np.round(np.linspace(0,1,8),7)),
    "lfo2destination":   list(np.round(np.linspace(0,1,8),7)),
    "lfo1waveform":      list(np.round(np.linspace(0,1,6),7)),
    "lfo2waveform":      list(np.round(np.linspace(0,1,6),7)),
    "freeaddestination": list(np.round(np.linspace(0,1,6),7)),
}
ORDINAL_49      = ["osc1tune", "osc2tune"]
BINARY          = ["lfo1keytrigger","lfo2keytrigger","oscsync",
                   "chorus1enable","chorus2enable","delaysync","delayx2l","delayx2r"]
BIPOLAR_NEUTRAL = {"lfo1amount","lfo2amount","osc1finetune","osc2finetune",
                   "osc1pw","filtercontour","freeadamount","masterdetune"}

MODEL_TO_VST3 = {
    "osc1waveform":"osc_1_waveform","osc1tune":"osc_1_tune",
    "osc1finetune":"osc_1_fine_tune","osc1volume":"osc_1_volume",
    "osc1pw":"osc_1_pw","osc1phase":"osc_1_phase",
    "osc2waveform":"osc_2_waveform","osc2tune":"osc_2_tune",
    "osc2finetune":"osc_2_fine_tune","osc2volume":"osc_2_volume",
    "osc2fm":"osc_2_fm","osc2phase":"osc_2_phase",
    "osc3volume":"osc_3_volume","ringmodulation":"ringmodulation",
    "oscsync":"osc_sync","oscbitcrusher":"osc_bitcrusher",
    "vintagenoise":"vintage_noise","filtertype":"filter_type",
    "filtercutoff":"filter_cutoff","filterresonance":"filter_resonance",
    "filtercontour":"filter_contour","filterdrive":"filter_drive",
    "filterattack":"filter_attack","filterdecay":"filter_decay",
    "filtersustain":"filter_sustain","filterrelease":"filter_release",
    "ampattack":"amp_attack","ampdecay":"amp_decay",
    "ampsustain":"amp_sustain","amprelease":"amp_release",
    "lfo1destination":"lfo_1_destination","lfo1waveform":"lfo_1_waveform",
    "lfo1rate":"lfo_1_rate","lfo1amount":"lfo_1_amount",
    "lfo1phase":"lfo1Phase","lfo1keytrigger":"lfo_1_keytrigger",
    "lfo2destination":"lfo_2_destination","lfo2waveform":"lfo_2_waveform",
    "lfo2rate":"lfo_2_rate","lfo2amount":"lfo_2_amount",
    "lfo2phase":"lfo2Phase","lfo2keytrigger":"lfo_2_keytrigger",
    "freeaddestination":"freeaddestination","freeadamount":"freeadamount",
    "freeadattack":"freeadattack","freeaddecay":"freeaddecay",
    "reverbwet":"reverb_wet","reverbdecay":"reverb_decay",
    "reverbpredelay":"reverb_predelay","reverbhighcut":"reverb_high_cut",
    "reverblowcut":"reverb_low_cut","delaywet":"delay_wet",
    "delaytime":"delay_time","delaysync":"delay_sync",
    "delayhighshelf":"delay_high_shelf","delaylowshelf":"delay_low_shelf",
    "delayfeedback":"delay_feedback","delayx2l":"delay_x2_l",
    "delayx2r":"delay_x2_r","chorus1enable":"chorus_1_enable",
    "chorus2enable":"chorus_2_enable","masterdetune":"master_detune",
    "masterhighpass":"master_high_pass",
}

_EPS = 0.05

class Param:
    def __init__(self, name, kind, **kw):
        self.name = name; self.kind = kind; self.__dict__.update(kw)
    def __len__(self):
        return len(self.values) if self.kind == "cat" else 1
    def decode(self, seg):
        if self.kind in ("cont","bipolar"):
            return float(np.clip((seg[0]+1.0)/2.0, 0, 1))
        if self.kind == "bin":
            return 1.0 if seg[0] >= 0.0 else 0.0
        if self.kind == "ord49":
            v = float(np.clip((seg[0]+1.0)/2.0, 0, 1))
            return round(int(round(v*48))/48, 7)
        return self.values[int(np.argmax(seg))]

class ParamSpec:
    def __init__(self, param_names):
        self.params = []
        for n in param_names:
            if n in CATEGORICAL:       self.params.append(Param(n,"cat",values=CATEGORICAL[n]))
            elif n in ORDINAL_49:      self.params.append(Param(n,"ord49"))
            elif n in BINARY:          self.params.append(Param(n,"bin"))
            elif n in BIPOLAR_NEUTRAL: self.params.append(Param(n,"bipolar"))
            else:                      self.params.append(Param(n,"cont"))
        self.layout = []; i = 0
        for p in self.params:
            k = len(p); self.layout.append((p, slice(i,i+k))); i += k
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
        self.mlp = nn.Sequential(nn.Linear(2*n_freq,d_cond), nn.SiLU(), nn.Linear(d_cond,d_cond))
    def forward(self, t):
        ang = 2*math.pi*t*self.freqs[None]
        return self.mlp(torch.cat([ang.sin(), ang.cos()], -1))

class DiTBlock(nn.Module):
    def __init__(self, d, n_heads, d_cond):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d,d*4), nn.GELU(), nn.Linear(d*4,d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_cond,6*d))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)
    def forward(self, x, z):
        sa,ga,ha,sf,gf,hf = self.ada(z).unsqueeze(1).chunk(6,-1)
        h = self.n1(x)*(1+ga)+sa; a,_ = self.attn(h,h,h)
        x = x+ha*a; h = self.n2(x)*(1+gf)+sf
        return x+hf*self.ff(h)

class FinalLayer(nn.Module):
    def __init__(self, d, d_cond):
        super().__init__()
        self.norm = nn.LayerNorm(d, elementwise_affine=False)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_cond,2*d))
        nn.init.zeros_(self.ada[-1].weight); nn.init.zeros_(self.ada[-1].bias)
    def forward(self, x, z):
        s,g = self.ada(z).unsqueeze(1).chunk(2,-1)
        return self.norm(x)*(1+g)+s

class VectorField(nn.Module):
    def __init__(self, spec, d_token, n_heads, n_layers, clap_dim, d_cond, n_freq_time=128):
        super().__init__()
        self.spec, self.d_token = spec, d_token
        self.in_p  = nn.ModuleList([nn.Linear(len(p),d_token) for p,_ in spec.layout])
        self.out_p = nn.ModuleList([nn.Linear(d_token,len(p)) for p,_ in spec.layout])
        for lin in self.out_p:
            nn.init.zeros_(lin.weight); nn.init.zeros_(lin.bias)
        self.p_emb    = nn.Embedding(len(spec.params), d_token)
        self.t_emb    = TimeEmbed(d_cond, n_freq_time)
        self.clap_mlp = nn.Sequential(nn.Linear(clap_dim,d_cond), nn.SiLU(), nn.Linear(d_cond,d_cond))
        self.cfg_token = nn.Parameter(torch.randn(1,clap_dim)/math.sqrt(clap_dim))
        self.blocks   = nn.ModuleList([DiTBlock(d_token,n_heads,d_cond) for _ in range(n_layers)])
        self.final    = FinalLayer(d_token, d_cond)
    def _tokenize(self, x):
        toks = torch.stack([self.in_p[i](x[:,sl]) for i,(_,sl) in enumerate(self.spec.layout)], 1)
        return toks+self.p_emb(torch.arange(len(self.spec.params), device=x.device))
    def forward(self, x, t, cond):
        B = x.shape[0]
        if cond is None: cond = self.cfg_token.expand(B,-1)
        z = self.t_emb(t)+self.clap_mlp(cond)
        toks = self._tokenize(x)
        for blk in self.blocks: toks = blk(toks,z)
        toks = self.final(toks,z)
        return torch.cat([self.out_p[i](toks[:,i]) for i in range(len(self.spec.layout))], 1)


def _call_cfg(vf, x, t, cond, s):
    if s == 1.0 or cond is None: return vf(x,t,cond)
    return (1-s)*vf(x,t,None) + s*vf(x,t,cond)

def _rk4(vf, x, t, dt, cond, s):
    f = lambda xx,tt: _call_cfg(vf,xx,tt,cond,s)
    k1=f(x,t); k2=f(x+dt*k1/2,t+dt/2)
    k3=f(x+dt*k2/2,t+dt/2); k4=f(x+dt*k3,t+dt)
    return x+(dt/6)*(k1+2*k2+2*k3+k4)

@torch.no_grad()
def sample_cfm(vf, cond, n_dim, steps=50, s=2.0):
    x = torch.randn(cond.shape[0], n_dim, device=DEVICE)
    t = torch.zeros(cond.shape[0],1,device=DEVICE); dt = 1.0/steps
    for _ in range(steps):
        x = _rk4(vf,x,t,dt,cond,s); t = t+dt
    return x

@torch.no_grad()
def sample_from_t(vf, x1, cond, t_star, steps=50, s=2.0):
    t_star = float(np.clip(t_star, 0.01, 0.99))
    noise  = torch.randn_like(x1)
    x      = (1.0 - t_star) * noise + t_star * x1
    dt     = 1.0 / steps
    remaining = max(1, int(round(steps * (1.0 - t_star))))
    t = torch.full((x1.shape[0], 1), t_star, device=DEVICE)
    for _ in range(remaining):
        x = _rk4(vf, x, t, dt, cond, s); t = t + dt
    return x


def enforce_audio_logic(params):
    def g(k, d=0.0): return float(params.get(k,d))
    if g("lfo1destination")==0.0:
        for p in ["lfo1waveform","lfo1rate","lfo1phase","lfo1keytrigger"]:
            if p in params: params[p]=0.0
        if "lfo1amount" in params: params["lfo1amount"]=0.5
    if abs(g("lfo1amount")-0.5)<_EPS:
        for p in ["lfo1waveform","lfo1rate","lfo1phase","lfo1keytrigger"]:
            if p in params: params[p]=0.0
        if "lfo1destination" in params: params["lfo1destination"]=0.0
    if g("lfo2destination")==0.0:
        for p in ["lfo2waveform","lfo2rate","lfo2phase","lfo2keytrigger"]:
            if p in params: params[p]=0.0
        if "lfo2amount" in params: params["lfo2amount"]=0.5
    if abs(g("lfo2amount")-0.5)<_EPS:
        for p in ["lfo2waveform","lfo2rate","lfo2phase","lfo2keytrigger"]:
            if p in params: params[p]=0.0
        if "lfo2destination" in params: params["lfo2destination"]=0.0
    if g("osc1waveform")!=0.5:
        if "osc1pw" in params: params["osc1pw"]=0.5
    if g("osc2volume")<_EPS and g("osc2fm")<_EPS:
        if "oscsync" in params: params["oscsync"]=0.0
    if g("filtertype")==1.0:
        for p in ["filtercutoff","filterresonance","filterattack","filterdecay",
                  "filtersustain","filterrelease","filterdrive"]:
            if p in params: params[p]=0.0
        if "filtercontour" in params: params["filtercontour"]=0.5
    if abs(g("filtercontour")-0.5)<_EPS:
        for p in ["filterattack","filterdecay","filtersustain","filterrelease"]:
            if p in params: params[p]=0.0
    if g("freeaddestination")==0.0:
        for p in ["freeadattack","freeaddecay"]:
            if p in params: params[p]=0.0
        if "freeadamount" in params: params["freeadamount"]=0.5
    if abs(g("freeadamount")-0.5)<_EPS:
        for p in ["freeadattack","freeaddecay"]:
            if p in params: params[p]=0.0
        if "freeaddestination" in params: params["freeaddestination"]=0.0
    if g("reverbwet")<_EPS:
        for p in ["reverbdecay","reverbpredelay","reverbhighcut","reverblowcut"]:
            if p in params: params[p]=0.0
    if g("delaywet")<_EPS:
        for p in ["delaytime","delaysync","delayhighshelf","delaylowshelf",
                  "delayfeedback","delayx2l","delayx2r"]:
            if p in params: params[p]=0.0
    return params

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
            seg[int(np.argmin([abs(v-cv) for cv in p.values]))] = 1.0
            x[sl] = seg
    return x


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        return audio.mean(axis=1)
    return audio

def _align_lengths(a1, a2):
    n = min(len(a1), len(a2))
    return a1[:n], a2[:n]

FFT_SIZES = [2048, 1024, 512, 256, 128, 64]

def metric_mss(ref: np.ndarray, gen: np.ndarray, sr: int) -> float:
    ref, gen = _align_lengths(ref, gen)
    total = 0.0
    for n_fft in FFT_SIZES:
        hop = n_fft // 4
        s_ref = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop))
        s_gen = np.abs(librosa.stft(gen, n_fft=n_fft, hop_length=hop))
        log_ref = np.log(s_ref + 1e-7)
        log_gen = np.log(s_gen + 1e-7)
        t = min(log_ref.shape[1], log_gen.shape[1])
        total += float(np.mean(np.abs(log_ref[:, :t] - log_gen[:, :t])))
    return total / len(FFT_SIZES)

def metric_wmfcc(ref: np.ndarray, gen: np.ndarray, sr: int, n_mfcc: int = 20) -> float:
    ref, gen = _align_lengths(ref, gen)
    m_ref = librosa.feature.mfcc(y=ref, sr=sr, n_mfcc=n_mfcc)
    m_gen = librosa.feature.mfcc(y=gen, sr=sr, n_mfcc=n_mfcc)
    D, _ = librosa.sequence.dtw(m_ref, m_gen, metric="euclidean")
    return float(D[-1, -1]) / float(D.shape[0] + D.shape[1])

def metric_sot(ref: np.ndarray, gen: np.ndarray, sr: int, n_fft: int = 1024) -> float:
    ref, gen = _align_lengths(ref, gen)
    hop = n_fft // 4
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    freqs_norm = freqs / (freqs.max() + 1e-8)
    s_ref = np.abs(librosa.stft(ref, n_fft=n_fft, hop_length=hop))
    s_gen = np.abs(librosa.stft(gen, n_fft=n_fft, hop_length=hop))
    T = min(s_ref.shape[1], s_gen.shape[1])
    dists = []
    for t in range(T):
        p = s_ref[:, t]; q = s_gen[:, t]
        ps, qs = p.sum(), q.sum()
        if ps < 1e-12 or qs < 1e-12:
            continue
        p_norm = p / ps; q_norm = q / qs
        dists.append(wasserstein_distance(freqs_norm, freqs_norm, p_norm, q_norm))
    return float(np.mean(dists)) if dists else 0.0

def metric_rms_cosine(ref: np.ndarray, gen: np.ndarray, sr: int) -> float:
    ref, gen = _align_lengths(ref, gen)
    r_ref = librosa.feature.rms(y=ref, frame_length=2048, hop_length=512)[0]
    r_gen = librosa.feature.rms(y=gen, frame_length=2048, hop_length=512)[0]
    n = min(len(r_ref), len(r_gen))
    r_ref, r_gen = r_ref[:n], r_gen[:n]
    denom = np.linalg.norm(r_ref) * np.linalg.norm(r_gen)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(r_ref, r_gen) / denom)

def embed_audio_clap(clap_model, audio: np.ndarray, sr: int) -> np.ndarray:
    if sr != RENDER_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=RENDER_SR)
    audio_in = audio.reshape(1, -1).astype(np.float32)
    emb = clap_model.get_audio_embedding_from_data(x=audio_in, use_tensor=False)[0]
    return emb / (np.linalg.norm(emb) + 1e-8)

def metric_clap_scores(clap_model, t_emb_raw_np, t_emb_adapted_np,
                       gen_audio: np.ndarray, gen_sr: int) -> tuple:
    a_emb = embed_audio_clap(clap_model, gen_audio, gen_sr)
    return (
        float(np.dot(t_emb_raw_np,     a_emb)),
        float(np.dot(t_emb_adapted_np, a_emb)),
    )

def compute_all_metrics(ref_mono, gen_mono, sr, clap_model, t_emb_raw_np, t_emb_adapted_np):
    clap_raw, clap_adapted = metric_clap_scores(
        clap_model, t_emb_raw_np, t_emb_adapted_np, gen_mono, sr)
    return {
        "MSS":                metric_mss(ref_mono, gen_mono, sr),
        "wMFCC":              metric_wmfcc(ref_mono, gen_mono, sr),
        "SOT":                metric_sot(ref_mono, gen_mono, sr),
        "RMS_cosine":         metric_rms_cosine(ref_mono, gen_mono, sr),
        "CLAP_score_raw":     clap_raw,
        "CLAP_score_adapted": clap_adapted,
    }


def apply_to_plugin(plugin, preset):
    for model_name, value in preset.items():
        vst_name = MODEL_TO_VST3.get(model_name)
        if vst_name and vst_name in plugin.parameters:
            try:
                plugin.parameters[vst_name].raw_value = float(value)
            except Exception:
                pass

def render_preset(plugin, preset: dict) -> np.ndarray:
    apply_to_plugin(plugin, preset)
    if "master_volume" in plugin.parameters:
        plugin.parameters["master_volume"].raw_value = 0.8
    midi_msgs = [
        (bytes([0x90, MIDI_NOTE, MIDI_VEL]), 0.0),
        (bytes([0x80, MIDI_NOTE, 0]),         NOTE_DUR),
    ]
    audio = plugin.process(midi_msgs, duration=RENDER_DUR,
                           sample_rate=RENDER_SR, reset=True)
    peak = float(np.max(np.abs(audio)))
    if peak > 1e-4:
        audio = audio * (10 ** (-3.0 / 20.0) / peak)
    mono = _to_mono(audio.T)
    return mono.astype(np.float32)


_CATEGORY_WORD_RE_CACHE = {}

def resolve_prompt_category(prompt_text: str, cap_data: dict,
                            vocab=CATEGORY_VOCAB) -> str:
    import re
    text_low = (prompt_text or "").lower()
    for cat in vocab:
        pattern = _CATEGORY_WORD_RE_CACHE.get(cat)
        if pattern is None:
            pattern = re.compile(r"\b" + re.escape(cat) + r"\b")
            _CATEGORY_WORD_RE_CACHE[cat] = pattern
        if pattern.search(text_low):
            return cat
    return (cap_data.get("choices") or {}).get("category", "").lower()

def retrieve_random_cat(prompt_category: str, pool_names: list,
                        pool_categories: dict, rng: np.random.Generator) -> str:
    cat_names = [n for n in pool_names if pool_categories.get(n, "") == prompt_category]
    pool = cat_names if (prompt_category and cat_names) else pool_names
    return pool[int(rng.integers(len(pool)))]


class Bundle:
    def __init__(self):
        self.vf = self.spec = self.clap = self.adapter = self.plugin = None
        self.pool_names = []
        self.pool_categories = {}
        self.pool_audio_embs = None
        self.pool_params = {}

def load_all(load_plugin_: bool = True) -> Bundle:
    b = Bundle()

    print("[..] Loading the CFM model...")
    ck   = torch.load(CFM_CKPT, map_location=DEVICE, weights_only=False)
    spec = ParamSpec(ck["param_names"])
    c    = ck["cfg"]
    vf   = VectorField(spec, c.get("d_token",256), c.get("n_heads",8),
                       c.get("n_layers",8), c.get("clap_dim",512),
                       c.get("d_cond",256), c.get("n_freq_time",128)).to(DEVICE)
    vf.load_state_dict(ck.get("ema", ck["vf"]))
    vf.eval()
    b.vf, b.spec = vf, spec
    print(f"    OK — dim={spec.dim}")

    print("[..] Loading CLAP...")
    import laion_clap
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=DEVICE)
    clap.load_ckpt(CLAP_CKPT)
    clap.eval()
    b.clap = clap
    print("    OK")

    print("[..] Loading the adapter...")
    adapter_ck = torch.load(ADAPTER_PATH, map_location=DEVICE, weights_only=False)
    adapter = TextAdapter(adapter_ck.get("dim", 512)).to(DEVICE)
    adapter.load_state_dict(adapter_ck["state_dict"])
    adapter.eval()
    b.adapter = adapter
    print(f"    OK — CV {adapter_ck.get('cv_mean',float('nan')):.4f} ± "
          f"{adapter_ck.get('cv_std',float('nan')):.4f}")

    if load_plugin_:
        print("[..] Loading the TAL-NoiseMaker plugin...")
        from pedalboard import load_plugin
        b.plugin = load_plugin(VST3_PATH)
        print(f"    OK — {b.plugin.name}")

    print("[..] Loading the retrieval pool (not_in_pool)...")
    emb_data = np.load(EMB_NPZ, allow_pickle=True)
    b.pool_audio_embs = emb_data["A"].astype(np.float32)
    b.pool_names      = [str(n) for n in emb_data["names"]]

    for name in b.pool_names:
        cp = os.path.join(NIP_CAPTION_DIR, name + ".json")
        if os.path.isfile(cp):
            with open(cp) as f:
                cd = json.load(f)
            b.pool_categories[name] = (cd.get("choices") or {}).get("category", "").lower()

    for name in b.pool_names:
        jp = os.path.join(NIP_JSON_DIR, name + ".json")
        if os.path.isfile(jp):
            with open(jp) as f:
                raw = json.load(f)
            b.pool_params[name] = {k: float(v)
                                   for k, v in (raw.get("parameters", raw) or {}).items()
                                   if isinstance(v, (int, float))}
    print(f"    OK — {len(b.pool_names)} presets in the pool")

    return b
