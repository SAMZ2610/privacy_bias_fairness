"""fmpriv.py - Encoding vs. fairness vs. membership-inference leakage in frozen medical foundation-model pipelines.

Input : meta.csv (one row per image) + emb.npy (N_images x d, same row order, frozen foundation-model embeddings)
        meta columns: patient_id, sex, race, age, insurance, + CheXpert-style label columns (1/0, -1/NaN = ignored)
Design: audit population of patients; N target heads, each trained on a random 50% of patients (balanced: every patient is a
        member of exactly N/2 models).  Record-level LiRA AUC in closed form (Knolle et al., Nature 2026, Eq.1-2),
        patient-level score = max over the patient's records.  Compared per patient / per group with (i) demographic
        encoding strength (out-of-fold linear probes) and (ii) fairness (FNR / underdiagnosis on held-out test patients).
Usage : python fmpriv.py --meta meta.csv --emb emb_raddino.npy --name raddino_linear --head linear
        python fmpriv.py --selftest
"""
import os, sys, json, math, time, logging, platform, argparse, tempfile
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score
from sklearn.model_selection import KFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

LABELS = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
          'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
NONLABEL = {"patient_id", "path", "rel", "sex", "age", "race", "insurance", "view", "age_bin", "strength", "dicom_id"}
EXCL = {"NAN", "UNKNOWN", "OTHER", "UNK", "", "NONE"}           # groups excluded from group-level tables
EXCL_PROBE = {"NAN", "UNKNOWN", "UNK", "", "NONE"}              # values excluded when fitting encoding probes
L = logging.getLogger("fmpriv")


@dataclass
class Cfg:
    name: str = "run"; out: str = "outputs"; meta: str = "meta.csv"; emb: str = "emb.npy"
    head: str = "linear"; hidden: int = 512              # 'linear' | 'mlp'
    n_models: int = 200; epochs: int = 30; bs: int = 2048; lr: float = 1e-3; wd: float = 1e-2
    seed: int = 0; val_frac: float = 0.10; test_frac: float = 0.20
    max_audit_patients: Optional[int] = None
    attrs: tuple = ("sex", "race", "age_bin", "insurance")
    min_group: int = 100; tail_q: float = 99.0; n_boot: int = 500; min_pos: int = 20; n_util_models: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------- utils
def setup(cfg):
    d = f"{cfg.out}/{cfg.name}"; os.makedirs(d, exist_ok=True)
    L.handlers.clear(); L.setLevel(logging.INFO); L.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(f"{d}/log.txt", mode="w")):
        h.setFormatter(fmt); L.addHandler(h)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    env = dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
               gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    json.dump(dict(cfg=asdict(cfg), env=env), open(f"{d}/config.json", "w"), indent=1)
    L.info(f"CONFIG {asdict(cfg)}"); L.info(f"ENV {env}")
    return d


def md(df, nd=3):
    f = lambda v: f"{v:.{nd}g}" if isinstance(v, (float, np.floating)) else str(v)
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] +
                     ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)])


def savefig(d, name):
    plt.tight_layout(); plt.savefig(f"{d}/{name}.png", dpi=200); plt.savefig(f"{d}/{name}.pdf"); plt.close()


def jdump(obj, path):
    json.dump(obj, open(path, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


class Timer:
    def __init__(s, msg): s.msg = msg
    def __enter__(s): s.t = time.time(); L.info(f"[start] {s.msg}")
    def __exit__(s, *a): L.info(f"[done ] {s.msg} ({time.time() - s.t:.1f}s)")


# ----------------------------------------------------------------------------- data
def load_data(cfg):
    meta = pd.read_csv(cfg.meta); X = np.load(cfg.emb).astype(np.float32)
    assert len(meta) == len(X), f"meta rows {len(meta)} != emb rows {len(X)}"
    assert np.isfinite(X).all(), "non-finite embeddings"
    labs = [c for c in meta.columns if c not in NONLABEL]; assert labs, "no label columns found"
    Y = meta[labs].to_numpy(np.float32); Y[Y < 0] = np.nan                    # uncertain (-1) -> ignored
    for a in cfg.attrs:
        if a != "age_bin": assert a in meta.columns, f"attribute '{a}' not in meta.csv - set attrs=... in Cfg"; meta[a] = meta[a].astype(str).str.upper()
    pc, pu = pd.factorize(meta.patient_id.astype(str))
    pdf = meta.groupby(pc).agg(age=("age", "mean"), n_rec=("age", "size"), **{a: (a, "first") for a in cfg.attrs if a != "age_bin"}).reset_index(drop=True)
    pdf.insert(0, "patient_id", pu.astype(str)); pdf["age_bin"] = pd.cut(pdf.age, [0, 40, 60, 80, 500], labels=["<40", "40-60", "60-80", "80+"], right=False).astype(str)
    nf = [i for i, l in enumerate(labs) if l != "No Finding"]
    pdf["n_findings"] = pd.Series((Y[:, nf] == 1).sum(1)).groupby(pc).mean().to_numpy()
    L.info(f"DATA records={len(meta)} patients={len(pdf)} d={X.shape[1]} labels={len(labs)} "
           f"records/patient={len(meta) / len(pdf):.2f}")
    return X, Y, labs, pc, pdf


def make_splits(pc, P, Y, cfg):
    rng = np.random.default_rng(cfg.seed); perm = rng.permutation(P)
    nt, nv = int(P * cfg.test_frac), int(P * cfg.val_frac)
    test, val, aud = perm[:nt], perm[nt:nt + nv], perm[nt + nv:]
    if cfg.max_audit_patients: aud = aud[:cfg.max_audit_patients]
    sp = np.full(P, -1); sp[aud] = 0; sp[val] = 1; sp[test] = 2
    assert not (set(aud) & set(val) | set(aud) & set(test) | set(val) & set(test)), "patient leakage between splits"
    rs = sp[pc]; has = (~np.isnan(Y)).any(1)
    ra, rv, rt = [np.where((rs == s) & has)[0] for s in (0, 1, 2)]
    L.info(f"SPLIT patients audit/val/test = {len(aud)}/{len(val)}/{len(test)} ; records = {len(ra)}/{len(rv)}/{len(rt)}")
    return aud, val, test, ra, rv, rt


# ----------------------------------------------------------------------------- batched heads (N independent models at once)
def init_head(kind, N, d, k, H, dev):
    r = lambda *s: torch.randn(*s, device=dev)
    z = lambda *s: torch.zeros(*s, device=dev)
    P = [r(N, d, k) / d ** .5, z(N, 1, k)] if kind == "linear" else \
        [r(N, d, H) / d ** .5, z(N, 1, H), r(N, H, k) / H ** .5, z(N, 1, k)]
    return [p.requires_grad_(True) for p in P]


def fwd(P, x, kind):                                   # x (B,d) -> (N,B,k)
    if kind == "linear": return torch.einsum("bd,ndk->nbk", x, P[0]) + P[1]
    h = F.gelu(torch.einsum("bd,ndh->nbh", x, P[0]) + P[1])
    return torch.bmm(h, P[2]) + P[3]


def train_heads(X, Y, V, M, cfg, tag):
    """X (R,d), Y (R,k) with nan->0, V (R,k) valid mask, M (N,R) bool membership. Every model sees only its member records."""
    N, R = M.shape; dev = X.device
    P = init_head(cfg.head, N, X.shape[1], Y.shape[1], cfg.hidden, dev)
    opt = torch.optim.AdamW(P, lr=cfg.lr, weight_decay=cfg.wd)
    steps = cfg.epochs * math.ceil(R / cfg.bs); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Mf = M.float()
    for s in range(steps):
        idx = torch.randint(0, R, (cfg.bs,), device=dev)
        lg = fwd(P, X[idx], cfg.head)
        l = F.binary_cross_entropy_with_logits(lg, Y[idx].expand(N, -1, -1), reduction="none")
        w = Mf[:, idx, None] * V[idx][None]
        loss = ((l * w).sum((1, 2)) / w.sum((1, 2)).clamp(min=1)).sum()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sch.step()
        if s % max(1, steps // 8) == 0 or s == steps - 1:
            L.info(f"  [{tag}] step {s}/{steps} mean member-loss/model={loss.item() / N:.4f}")
    assert math.isfinite(loss.item()), "training diverged"
    return [p.detach() for p in P]


@torch.no_grad()
def logits_of(P, X, kind, chunk=2048):
    return torch.cat([fwd(P, X[i:i + chunk], kind) for i in range(0, len(X), chunk)], 1)    # (N,R,k)


@torch.no_grad()
def margins(P, X, Y, V, kind, chunk=2048):
    """Record statistic = mean over valid labels of the signed logit margin (= LiRA logit-scaled confidence for binary labels)."""
    out = []
    for i in range(0, len(X), chunk):
        lg = fwd(P, X[i:i + chunk], kind); sg = 2 * Y[i:i + chunk] - 1; v = V[i:i + chunk]
        out.append(((lg * sg[None]) * v[None]).sum(2) / v.sum(1).clamp(min=1)[None])
    return torch.cat(out, 1)                                                                  # (N,R)


def macro_auc(lg, Y, V):
    a = []
    for j in range(lg.shape[1]):
        m = V[:, j] > 0
        if m.sum() > 10 and 0 < Y[m, j].sum() < m.sum(): a.append(roc_auc_score(Y[m, j], lg[m, j]))
    return float(np.mean(a)) if a else float("nan")


# ----------------------------------------------------------------------------- membership inference (record/patient level)
Phi = lambda z: 0.5 * (1 + torch.erf(z / 2 ** .5))


def lira_record_auc(S, M, chunk=50000):
    """Closed-form record-level MIA AUC  theta = Phi((mu_in-mu_out)/sqrt(var_in+var_out)) and Hanley-McNeil SE (Nature Eq.1-2)."""
    th, se = [], []
    for i in range(0, S.shape[1], chunk):
        s, m = S[:, i:i + chunk], M[:, i:i + chunk].float()
        ni, no = m.sum(0), (1 - m).sum(0)
        mi, mo = (s * m).sum(0) / ni, (s * (1 - m)).sum(0) / no
        vi, vo = (((s - mi) ** 2) * m).sum(0) / (ni - 1), (((s - mo) ** 2) * (1 - m)).sum(0) / (no - 1)
        t = Phi((mi - mo) / torch.sqrt(vi + vo + 1e-12)); q1, q2 = t / (2 - t), 2 * t ** 2 / (1 + t)
        th.append(t); se.append(torch.sqrt(((t * (1 - t) + (ni - 1) * (q1 - t ** 2) + (no - 1) * (q2 - t ** 2)) / (ni * no)).clamp(min=0)))
    return torch.cat(th), torch.cat(se)


def lira_aggregate(S, M, max_pairs=4_000_000, chunk=20000, seed=0):
    """Aggregate LiRA (online, leave-one-model-out Gaussians) over all (model, record) pairs -> ROC / AUC / TPR@FPR."""
    g = torch.Generator(device="cpu").manual_seed(seed); N, R = S.shape; keep = min(1.0, max_pairs / (N * R)); sc, lb = [], []
    for i in range(0, R, chunk):
        s, m = S[:, i:i + chunk], M[:, i:i + chunk].float(); o = 1 - m
        ni0, no0 = m.sum(0), o.sum(0)
        s1i, s2i, s1o, s2o = (s * m).sum(0), (s * s * m).sum(0), (s * o).sum(0), (s * s * o).sum(0)
        ni, no = ni0 - m, no0 - o                                                # leave the pair's own model out
        a1, a2, b1, b2 = s1i - s * m, s2i - s * s * m, s1o - s * o, s2o - s * s * o
        mi, mo = a1 / ni, b1 / no
        vi, vo = ((a2 - ni * mi ** 2) / (ni - 1)).clamp(min=1e-4), ((b2 - no * mo ** 2) / (no - 1)).clamp(min=1e-4)
        llr = -0.5 * torch.log(vi) - 0.5 * (s - mi) ** 2 / vi + 0.5 * torch.log(vo) + 0.5 * (s - mo) ** 2 / vo
        sel = torch.rand(llr.shape, generator=g) < keep
        sc.append(llr.cpu()[sel].numpy()); lb.append(m.cpu()[sel].numpy())
    sc, lb = np.concatenate(sc), np.concatenate(lb).astype(int)
    fpr, tpr, _ = roc_curve(lb, sc)
    out = dict(auc=float(roc_auc_score(lb, sc)), n_pairs=int(len(sc)))
    for f in (1e-3, 1e-2, 1e-1): out[f"tpr@fpr{f:g}"] = float(np.interp(f, fpr, tpr))
    return out, (fpr, tpr)


def balanced_mask(Pa, N, seed):
    """(N,Pa) bool; every patient is a member of exactly N/2 models."""
    assert N % 2 == 0
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(Pa, N, generator=g).argsort(1).argsort(1) < N // 2).T.contiguous()


def patient_max(th, pidx, Pa):
    return torch.zeros(Pa, device=th.device).scatter_reduce_(0, pidx, th, "amax", include_self=False)


def check_closed_form(S, M, th, n=300, seed=0):
    """Validate the closed-form record-level AUC against the empirical Mann-Whitney AUC of the in/out scores of random records."""
    idx = np.random.default_rng(seed).choice(S.shape[1], min(n, S.shape[1]), replace=False)
    Sn, Mn, t = S[:, idx].cpu().numpy(), M[:, idx].cpu().numpy(), th[idx].cpu().numpy()
    e = np.array([roc_auc_score(Mn[:, j], Sn[:, j]) for j in range(len(idx))])
    return dict(mean_abs_diff=float(np.abs(e - t).mean()), corr=float(np.corrcoef(e, t)[0, 1]))


# ----------------------------------------------------------------------------- encoding (out-of-fold linear probes on patient-mean embeddings)
def fit_linear(Z, y, C, ep=30, bs=2048, lr=3e-3, wd=1e-4):
    m = nn.Linear(Z.shape[1], C).to(Z.device); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
    for _ in range(ep):
        perm = torch.randperm(len(Z), device=Z.device)
        for i in range(0, len(Z), bs):
            b = perm[i:i + bs]; loss = F.cross_entropy(m(Z[b]), y[b]); opt.zero_grad(); loss.backward(); opt.step()
    return m


def encoding_scores(Zp, pdf, cfg):
    """Per-patient encoding score of attribute a = out-of-fold *balanced* posterior probability of the patient's true class,
    i.e. softmax(logits - log prior): chance level is 1/C for EVERY group, so the score is not a proxy for group size/prevalence.
    (Naive log-odds / PMI is mechanically larger for rare groups - do not use it for cross-group comparison.)"""
    rows = []
    for a in cfg.attrs:
        ok = ~pdf[a].astype(str).str.upper().isin(EXCL_PROBE).to_numpy(); idx = np.where(ok)[0]
        cl, yc = np.unique(pdf[a].to_numpy()[idx].astype(str), return_inverse=True); C = len(cl)
        Z, yt = Zp[idx], torch.tensor(yc, device=Zp.device); post = np.zeros((len(idx), C), np.float32)
        for tr, te in KFold(5, shuffle=True, random_state=cfg.seed).split(idx):
            m = fit_linear(Z[tr], yt[tr], C)
            with torch.no_grad(): o = m(Z[te]).cpu()
            prior = torch.tensor(np.bincount(yc[tr], minlength=C) / len(tr) + 1e-9, dtype=torch.float32)
            post[te] = torch.softmax(o - prior.log()[None], 1).numpy()
        enc = np.full(len(pdf), np.nan, np.float32); enc[idx] = post[np.arange(len(idx)), yc]; pdf[f"enc_{a}"] = enc
        auc = roc_auc_score(yc, post[:, 1]) if C == 2 else roc_auc_score(yc, post, multi_class="ovr", average="macro")
        rows.append(dict(attr=a, classes=str([str(c) for c in cl]), chance=1 / C, macro_auroc=auc, balanced_acc=balanced_accuracy_score(yc, post.argmax(1)),
                         mean_enc_score=float(np.nanmean(enc))))
    t = pd.DataFrame(rows); L.info("ENCODING (OOF linear probes, balanced posterior)\n" + t.to_string(index=False)); return t


@torch.no_grad()
def sparsity(Z, k=10, chunk=4096):
    Zn = F.normalize(Z, dim=1); out = []
    for i in range(0, len(Zn), chunk):
        s = Zn[i:i + chunk] @ Zn.T; s[torch.arange(len(s)), torch.arange(i, i + len(s))] = -2
        out.append(1 - s.topk(min(k, len(Zn) - 1), 1).values.mean(1))
    return torch.cat(out).cpu().numpy()


# ----------------------------------------------------------------------------- fairness on held-out test patients (reference head)
def youden(lg, Y, V):
    thr = np.zeros(lg.shape[1])
    for j in range(lg.shape[1]):
        m = V[:, j] > 0
        if m.sum() > 10 and 0 < Y[m, j].sum() < m.sum():
            f, t, th = roc_curve(Y[m, j], lg[m, j]); thr[j] = th[np.argmax(t - f)]
    return thr


def fairness_table(lgT, YT, VT, pT, thr, groups, labs, cfg, rng):
    """groups: dict attr -> array(test patients) of group names. Patient-clustered bootstrap for FNR gap & underdiagnosis gap."""
    Pt = len(groups[cfg.attrs[0]]); pred = lgT >= thr[None]; pos, neg = (YT == 1) & (VT > 0), (YT == 0) & (VT > 0)
    cnt = {n: np.zeros((Pt, len(labs)), np.float32) for n in ("TP", "FN", "FP", "TN")}
    for n, v in dict(TP=pred & pos, FN=~pred & pos, FP=pred & neg, TN=~pred & neg).items(): np.add.at(cnt[n], pT, v.astype(np.float32))
    nfi = labs.index("No Finding") if "No Finding" in labs else None; dis = [j for j, l in enumerate(labs) if l != "No Finding"]
    W = np.vstack([np.ones((1, Pt)), rng.multinomial(Pt, np.ones(Pt) / Pt, size=cfg.n_boot)]).astype(np.float32)   # row 0 = point estimate

    def stat(idx):
        fn, tp, fp, tn = (W[:, idx] @ cnt[n][idx] for n in ("FN", "TP", "FP", "TN"))
        keep = [j for j in dis if (cnt["FN"][idx, j] + cnt["TP"][idx, j]).sum() >= cfg.min_pos]
        with np.errstate(all="ignore"):
            fnr = np.nanmean((fn / np.maximum(fn + tp, 1))[:, keep], 1) if keep else np.full(len(W), np.nan)
            ud = fp[:, nfi] / np.maximum(fp[:, nfi] + tn[:, nfi], 1) if nfi is not None else np.full(len(W), np.nan)
        return fnr, ud
    rows = []
    for a in cfg.attrs:
        g = groups[a]; names = [n for n in np.unique(g) if n.upper() not in EXCL and (g == n).sum() >= cfg.min_group // 2]
        if len(names) < 2: continue
        ref = max(names, key=lambda n: (g == n).sum()); fr, ur = stat(np.where(g == ref)[0])
        for n in names:
            idx = np.where(g == n)[0]; f, u = stat(idx); au = []; rm = np.isin(pT, idx)
            for j in dis:
                m = VT[rm, j] > 0; y = YT[rm, j]; l = lgT[rm, j]
                if m.sum() > 10 and 0 < y[m].sum() < m.sum(): au.append(roc_auc_score(y[m], l[m]))
            q = lambda v: np.nanpercentile(v[1:], [2.5, 97.5])
            rows.append(dict(attr=a, group=n, n_test=len(idx), ref=ref, macro_fnr=f[0], fnr_gap=f[0] - fr[0],
                             fnr_gap_lo=q(f - fr)[0], fnr_gap_hi=q(f - fr)[1], underdiag_fpr=u[0], ud_gap=u[0] - ur[0],
                             ud_gap_lo=q(u - ur)[0], ud_gap_hi=q(u - ur)[1], macro_auroc=np.mean(au) if au else np.nan))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- patient-level association (the three-way analysis)
def _stats(C, x, y):
    r = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    ry, rx = r(y), r(x); r2c = 1 - ry.var() / y.var()
    C2 = np.c_[C, x]; r2f = 1 - (y - C2 @ np.linalg.lstsq(C2, y, rcond=None)[0]).var() / y.var()
    return np.corrcoef(x, y)[0, 1], np.corrcoef(rx, ry)[0, 1], r2c, r2f - r2c


def assoc(df, cfg, rng):
    """Does encoding strength predict patient-level MIA AUC beyond group size / rarity / disease burden / #records?
    Rank-transformed variables, patient-level pairs bootstrap CIs."""
    rows = []
    for a in cfg.attrs:
        c = ["log_nrec", "sparsity", "n_findings", f"lsize_{a}"]; d = df.dropna(subset=[f"enc_{a}", "mia", *c])
        if len(d) < 200: continue
        rk = lambda v: stats.rankdata(v) / len(v)
        y, x = rk(d.mia.to_numpy()), rk(d[f"enc_{a}"].to_numpy()); C = np.column_stack(
            [np.ones(len(d))] + [(lambda v: (v - v.mean()) / (v.std() + 1e-9))(rk(d[k].to_numpy())) for k in c])
        pt = np.array(_stats(C, x, y)); bs = np.array([_stats(C[i], x[i], y[i]) for i in (rng.integers(0, len(d), len(d)) for _ in range(cfg.n_boot))])
        lo, hi = np.percentile(bs, [2.5, 97.5], 0); n = len(d); rmin = math.tanh((1.96 + 0.84) / math.sqrt(n - 3))
        rows.append(dict(attr=a, n=n, spearman_raw=pt[0], raw_lo=lo[0], raw_hi=hi[0], partial=pt[1], part_lo=lo[1], part_hi=hi[1],
                         R2_confounds=pt[2], dR2_enc=pt[3], dR2_lo=lo[3], dR2_hi=hi[3], r_min_detectable_80pct_power=rmin))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- group tables, figures
def group_tables(dfa, fair, cfg):
    thr = np.percentile(dfa.mia, cfg.tail_q); dfa["in_tail"] = dfa.mia >= thr; rows = []
    for a in cfg.attrs:
        g = dfa[a].astype(str); names = [n for n in sorted(g.unique()) if n.upper() not in EXCL and (g == n).sum() >= cfg.min_group]
        if not names: continue
        ng = np.array([(g == n).sum() for n in names]); O = np.array([dfa.in_tail[g == n].sum() for n in names]); E = O.sum() * ng / ng.sum()
        p = stats.chisquare(O, E).pvalue if len(names) > 1 and O.sum() > 0 else np.nan
        for i, n in enumerate(names):
            m = g == n; mi = dfa.mia[m]
            rows.append(dict(attr=a, group=n, n_audit=int(ng[i]), enc_mean=dfa[f"enc_{a}"][m].mean(), mia_mean=mi.mean(),
                             mia_sem=mi.std() / math.sqrt(m.sum()), frac_auc_ge_0_7=(mi >= .7).mean(), tail_obs=int(O[i]), tail_exp=E[i],
                             tail_resid=(O[i] - E[i]) / math.sqrt(max(E[i], 1e-9)), tail_rel_change_pct=100 * (O[i] - E[i]) / max(E[i], 1e-9),
                             chi2_p_bonf=min(1.0, p * len(cfg.attrs)) if p == p else np.nan))
    gt = pd.DataFrame(rows)
    if len(fair): gt = gt.merge(fair.drop(columns=["ref"]).rename(columns={"n_test": "n_test"}), on=["attr", "group"], how="left")
    return gt, float(thr)


def make_figures(d, dfa, th_rec, th0_rec, pa0, gt, S_agg, cfg):
    x = np.linspace(.5, 1, 400); ecdf = lambda v: np.array([(v >= t).mean() for t in x])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    for a_, v, v0, t in ((ax[0], dfa.mia.to_numpy(), pa0, "Patient-level"), (ax[1], th_rec, th0_rec, "Record-level")):
        a_.semilogy(x, np.maximum(ecdf(v), 1e-6), label="observed"); a_.semilogy(x, np.maximum(ecdf(v0), 1e-6), "--", c="gray", label="null (shuffled membership)")
        a_.set_xlabel("MIA AUC"); a_.set_ylabel("fraction with AUC $\\geq$ x"); a_.set_title(t); a_.set_ylim(bottom=1e-5); a_.legend(fontsize=7)
    savefig(d, "fig1_mia_tails")
    fpr, tpr = S_agg; plt.figure(figsize=(3.8, 3.6)); plt.loglog(fpr, tpr); plt.loglog([1e-5, 1], [1e-5, 1], "--", c="gray")
    plt.xlim(1e-5, 1); plt.ylim(1e-5, 1); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("Aggregate LiRA"); savefig(d, "fig2_aggregate_roc")
    if len(gt):
        at = list(gt.attr.unique()); fig, ax = plt.subplots(1, len(at), figsize=(3 * len(at), 3), squeeze=False)
        for a_, a in zip(ax[0], at):
            s = gt[gt.attr == a]; a_.bar(s.group, s.tail_resid, color=["C3" if v > 0 else "C0" for v in s.tail_resid])
            a_.set_title(f"{a}  (p={s.chi2_p_bonf.iloc[0]:.1e})", fontsize=8); a_.tick_params(axis="x", rotation=60, labelsize=7); a_.axhline(0, c="k", lw=.5)
        ax[0][0].set_ylabel("Pearson residual\n(top-tail enrichment)"); savefig(d, "fig3_group_tail_enrichment")
        if "fnr_gap" in gt:
            g = gt.dropna(subset=["enc_mean", "mia_mean", "fnr_gap"]); pairs = [("enc_mean", "mia_mean"), ("enc_mean", "fnr_gap"), ("fnr_gap", "mia_mean")]
            fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
            for a_, (u, v) in zip(ax, pairs):
                for i, a in enumerate(g.attr.unique()):
                    s = g[g.attr == a]; a_.scatter(s[u], s[v], label=a, c=f"C{i}")
                    for _, r in s.iterrows(): a_.annotate(r.group, (r[u], r[v]), fontsize=6)
                rho = stats.spearmanr(g[u], g[v])[0] if len(g) > 3 else np.nan; a_.set_xlabel(u); a_.set_ylabel(v); a_.set_title(f"group-level Spearman={rho:.2f} (n={len(g)})", fontsize=8)
            ax[0].legend(fontsize=6); savefig(d, "fig4_three_way_groups")
    fig, ax = plt.subplots(1, len(cfg.attrs), figsize=(3 * len(cfg.attrs), 3), squeeze=False)
    for a_, a in zip(ax[0], cfg.attrs):
        s = dfa.dropna(subset=[f"enc_{a}"]); b = pd.qcut(s[f"enc_{a}"], 10, labels=False, duplicates="drop"); g = s.groupby(b).mia.agg(["mean", "sem"])
        a_.errorbar(g.index, g["mean"], 1.96 * g["sem"], marker="o"); a_.set_title(a, fontsize=8); a_.set_xlabel("encoding-strength decile")
    ax[0][0].set_ylabel("mean patient MIA AUC"); savefig(d, "fig5_encoding_vs_leakage")


# ----------------------------------------------------------------------------- main pipeline
def run(cfg):
    d = setup(cfg); dev = cfg.device; T0 = time.time(); res = {}
    with Timer("load"):
        Xn, Y, labs, pc, pdf = load_data(cfg); P = len(pdf)
        aud, val, test, ra, rv, rt = make_splits(pc, P, Y, cfg)
        mu, sd = Xn[ra].mean(0), Xn[ra].std(0) + 1e-6; Xn = (Xn - mu) / sd                       # standardise with audit statistics
        Yz = np.nan_to_num(Y, nan=0.0); Vm = (~np.isnan(Y)).astype(np.float32)
        T = lambda a: torch.tensor(a, device=dev)
        X, Yt, Vt = T(Xn[ra]), T(Yz[ra]), T(Vm[ra]); Pa = len(aud); pos_a = -np.ones(P, int); pos_a[aud] = np.arange(Pa)
        pidx = torch.tensor(pos_a[pc[ra]], device=dev); assert (pidx >= 0).all()
    # 1) target models ------------------------------------------------------------------------------------------------
    N = cfg.n_models; Mp = balanced_mask(Pa, N, cfg.seed).to(dev); M = Mp[:, pidx]
    assert (Mp.sum(0) == N // 2).all(), "membership not balanced"
    with Timer(f"train {N} target heads ({cfg.head})"): Ph = train_heads(X, Yt, Vt, M, cfg, "targets")
    with Timer("record statistics"): S = margins(Ph, X, Yt, Vt, cfg.head); assert torch.isfinite(S).all()
    # 2) MIA ------------------------------------------------------------------------------------------------------------
    with Timer("MIA (record + patient level, aggregate, null)"):
        th, se = lira_record_auc(S, M); pa = patient_max(th, pidx, Pa)
        Mp0 = balanced_mask(Pa, N, cfg.seed + 12345).to(dev); th0, _ = lira_record_auc(S, Mp0[:, pidx]); pa0 = patient_max(th0, pidx, Pa)   # null: membership unrelated to training
        agg, roc = lira_aggregate(S, M, seed=cfg.seed)
    pa_np, th_np = pa.cpu().numpy(), th.cpu().numpy(); cf = check_closed_form(S, M, th); L.info(f"CLOSED-FORM vs EMPIRICAL record AUC: {cf}")
    res["attack"] = dict(closed_form_check=cf, aggregate=agg, record_auc_median=float(np.median(th_np)), record_auc_p99=float(np.percentile(th_np, 99)),
                         patient_auc_median=float(np.median(pa_np)), patient_auc_p99=float(np.percentile(pa_np, 99)),
                         patient_frac_ge=dict((f"{t}", float((pa_np >= t).mean())) for t in (.6, .7, .8, .9, .95)),
                         record_se_median=float(se.median()), record_se_p99_tail_median=float(se[th >= torch.quantile(th, .99)].median()),
                         null=dict(record_median=float(th0.median()), record_p99=float(torch.quantile(th0, .99)), patient_median=float(pa0.median()),
                                   patient_p99=float(np.percentile(pa0.cpu().numpy(), 99))))
    L.info("ATTACK " + json.dumps(res["attack"], default=float))
    assert abs(res["attack"]["null"]["record_median"] - 0.5) < 0.03, "null check failed: record-level AUC not centred at 0.5"
    # 3) utility (reference head on all audit patients + a sample of target models) ------------------------------------
    with Timer("reference head + utility"):
        Mref = torch.ones(1, len(ra), dtype=torch.bool, device=dev); Pr = train_heads(X, Yt, Vt, Mref, cfg, "reference")
        XV, XT = T(Xn[rv]), T(Xn[rt]); lgV = logits_of(Pr, XV, cfg.head)[0].cpu().numpy(); lgT = logits_of(Pr, XT, cfg.head)[0].cpu().numpy()
        thr = youden(lgV, Yz[rv], Vm[rv]); ut = [macro_auc(lgT, Yz[rt], Vm[rt])]
        sub = [p[:cfg.n_util_models] for p in Ph]; lgs = logits_of(sub, XT, cfg.head).cpu().numpy(); ut += [macro_auc(lgs[i], Yz[rt], Vm[rt]) for i in range(len(lgs))]
    res["utility"] = dict(reference_test_macro_auroc=ut[0], target_test_macro_auroc_mean=float(np.mean(ut[1:])), target_test_macro_auroc_sd=float(np.std(ut[1:])))
    L.info("UTILITY " + json.dumps(res["utility"]))
    # 4) encoding + confounds -------------------------------------------------------------------------------------------
    with Timer("encoding probes + confounds"):
        Zp = torch.zeros(P, Xn.shape[1]).index_add_(0, torch.tensor(pc), torch.tensor(Xn)); Zp = T((Zp / torch.bincount(torch.tensor(pc), minlength=P)[:, None]).numpy())
        enc_t = encoding_scores(Zp, pdf, cfg); sp = sparsity(Zp[torch.tensor(aud, device=dev)])
    dfa = pdf.iloc[aud].reset_index(drop=True).copy(); dfa["mia"] = pa_np; dfa["mia_null"] = pa0.cpu().numpy(); dfa["sparsity"] = sp
    dfa["log_nrec"] = np.log(dfa.n_rec)
    for a in cfg.attrs: dfa[f"lsize_{a}"] = np.log(dfa.groupby(a)[a].transform("size"))
    # 5) fairness -------------------------------------------------------------------------------------------------------
    with Timer("fairness (test patients, patient-clustered bootstrap)"):
        pT = pd.factorize(pc[rt])[0]; tp_ids = pd.factorize(pc[rt])[1]; groups = {a: pdf[a].to_numpy()[tp_ids].astype(str) for a in cfg.attrs}
        fair = fairness_table(lgT, Yz[rt], Vm[rt], pT, thr, groups, labs, cfg, np.random.default_rng(cfg.seed))
    # 6) three-way analysis, tables, figures -------------------------------------------------------------------------
    with Timer("three-way analysis"):
        gt, tail_thr = group_tables(dfa, fair, cfg); ass = assoc(dfa, cfg, np.random.default_rng(cfg.seed + 1))
    res.update(tail_threshold_auc=tail_thr, n=dict(audit_patients=Pa, val_patients=len(val), test_patients=len(test), audit_records=len(ra), models=N),
               assoc=ass.to_dict("records"))
    L.info("GROUP TABLE\n" + gt.round(3).to_string(index=False)); L.info("ASSOCIATION\n" + ass.round(3).to_string(index=False))
    enc_t.to_csv(f"{d}/encoding_overall.csv", index=False); gt.to_csv(f"{d}/group_table.csv", index=False); ass.to_csv(f"{d}/patient_assoc.csv", index=False)
    fair.to_csv(f"{d}/fairness_table.csv", index=False); dfa.to_csv(f"{d}/patient_level.csv.gz", index=False)
    make_figures(d, dfa, th_np, th0.cpu().numpy(), dfa.mia_null.to_numpy(), gt, roc, cfg); jdump(res, f"{d}/paper_numbers.json")
    with open(f"{d}/summary.md", "w") as f:
        f.write(f"# {cfg.name}\nhead={cfg.head}, N={N} target models, audit patients={Pa}, runtime={time.time() - T0:.0f}s\n\n## Utility\n```\n{json.dumps(res['utility'], indent=1)}\n```\n"
                f"## Attack\n```\n{json.dumps(res['attack'], indent=1, default=float)}\n```\n## Encoding probes\n{md(enc_t)}\n\n## Group table\n{md(gt)}\n\n## Patient-level association\n{md(ass)}\n")
    L.info(f"DONE in {time.time() - T0:.0f}s -> {d}"); return res


def compare_runs(out="outputs"):
    """Collect paper_numbers.json of all runs into one table (capacity / model axis)."""
    rows = []
    for r in sorted(os.listdir(out)):
        p = f"{out}/{r}/paper_numbers.json"
        if os.path.exists(p):
            j = json.load(open(p)); a = j["attack"]
            rows.append(dict(run=r, test_auroc=j["utility"]["reference_test_macro_auroc"], agg_mia_auc=a["aggregate"]["auc"], tpr_at_1e3=a["aggregate"]["tpr@fpr0.001"],
                             patient_p99=a["patient_auc_p99"], frac_ge_0_9=a["patient_frac_ge"]["0.9"], null_p99=a["null"]["patient_p99"]))
    t = pd.DataFrame(rows); t.to_csv(f"{out}/compare_runs.csv", index=False); return t


def unit_tests():
    """Ground-truth checks of the statistics (no training involved)."""
    rng = np.random.default_rng(0); n = 4000; cfg = Cfg(attrs=("sex",), n_boot=150)
    ls = rng.normal(size=n); base = dict(log_nrec=rng.normal(size=n), sparsity=rng.normal(size=n), n_findings=rng.normal(size=n), lsize_sex=ls)
    e = rng.normal(size=n); df = pd.DataFrame(base); df["enc_sex"] = e; df["mia"] = 0.5 * e + rng.normal(size=n)          # A: leakage driven by encoding
    r = assoc(df, cfg, rng).iloc[0]; assert r.part_lo > 0.2 and r.dR2_lo > 0.05, f"assoc misses a true effect: {r.to_dict()}"
    df["enc_sex"] = ls + 0.5 * rng.normal(size=n); df["mia"] = ls + rng.normal(size=n)                                  # B: encoding is only a proxy of group size
    r = assoc(df, cfg, rng).iloc[0]; assert r.spearman_raw > 0.3 and abs(r.partial) < 0.08 and r.dR2_enc < 0.01, f"assoc not controlling confounds: {r.to_dict()}"
    M = balanced_mask(500, 40, 0); assert (M.sum(0) == 20).all()                                                         # C: balanced membership
    S = torch.randn(40, 3000); Mr = balanced_mask(3000, 40, 1); th, _ = lira_record_auc(S, Mr)                          # D: null -> centred at 0.5
    assert abs(th.median().item() - 0.5) < 0.02, "null record AUC not centred"
    S2 = torch.randn(40, 3000) + 3.0 * Mr.float(); th2, _ = lira_record_auc(S2, Mr); assert th2.median().item() > 0.95                # E: planted membership signal detected
    ag, _ = lira_aggregate(S2, Mr); assert ag["auc"] > 0.9, "aggregate attack misses planted signal"
    print("unit tests OK")


# ----------------------------------------------------------------------------- synthetic data (selftest)
def synthetic(path, n_pat=4000, d=64, seed=0, plant=True):
    """plant=True: per-patient demographic strength varies (selftest checks that the encoding score tracks it)."""
    rng = np.random.default_rng(seed); os.makedirs(path, exist_ok=True)
    sex = rng.choice(["M", "F"], n_pat); race = rng.choice(["WHITE", "BLACK", "ASIAN", "HISPANIC"], n_pat, p=[.65, .2, .05, .1])
    ins = rng.choice(["MEDICARE", "MEDICAID", "OTHER"], n_pat, p=[.5, .15, .35]); age = rng.normal(60, 16, n_pat).clip(18, 95)
    nrec = rng.integers(1, 5, n_pat); k = 14; Wl = rng.normal(size=(d, k)) / 4
    dirs = {n: rng.normal(size=d) for n in ("sex", "black", "asian", "age")}
    st = rng.uniform(.2, 2, n_pat) if plant else np.ones(n_pat); rows, E = [], []
    for i in range(n_pat):
        base = rng.normal(size=d) * 0.5; demo = (sex[i] == "F") * dirs["sex"] + (race[i] == "BLACK") * dirs["black"] + (race[i] == "ASIAN") * dirs["asian"] + (age[i] - 60) / 16 * dirs["age"] * .5
        demo = demo * st[i]
        for _ in range(nrec[i]):
            e = base + demo + rng.normal(size=d) * (1.5 if rng.random() < .03 else .6)        # a few noisy/atypical images
            y = (1 / (1 + np.exp(-(e @ Wl + rng.normal(size=k) * (1 + (1.5 * st[i] if plant else 0)) - 1.5))) > rng.random(k)).astype(float); y[0] = float(y[1:].sum() == 0)
            E.append(e); rows.append(dict(patient_id=f"p{i}", strength=st[i], sex=sex[i], race=race[i], age=age[i], insurance=ins[i], **dict(zip(LABELS, y))))
    pd.DataFrame(rows).to_csv(f"{path}/meta.csv", index=False); np.save(f"{path}/emb.npy", np.array(E, np.float32))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    for f, v in asdict(Cfg()).items():
        if f not in ("attrs", "max_audit_patients", "device"): ap.add_argument(f"--{f}", type=type(v), default=v)
    ap.add_argument("--max_audit_patients", type=int, default=None); a = ap.parse_args()
    if a.selftest:
        unit_tests(); tmp = tempfile.mkdtemp(); synthetic(tmp)
        for head in ("linear", "mlp"):
            run(Cfg(name=f"selftest_{head}", out=f"{tmp}/out", meta=f"{tmp}/meta.csv", emb=f"{tmp}/emb.npy", head=head, hidden=64, n_models=40,
                    epochs=40 if head == "mlp" else 20, bs=512, lr=3e-3 if head == "mlp" else 1e-2, wd=1e-3, min_group=40, n_boot=100, n_util_models=5))
        pl = pd.read_csv(f"{tmp}/out/selftest_mlp/patient_level.csv.gz"); stg = pd.read_csv(f"{tmp}/meta.csv").groupby("patient_id").strength.first()
        assert stats.spearmanr(pl.enc_sex, stg.reindex(pl.patient_id).to_numpy())[0] > 0.3, "encoding score does not track planted demographic strength"
        r = json.load(open(f"{tmp}/out/selftest_mlp/paper_numbers.json")); assert r["attack"]["aggregate"]["auc"] > 0.55 and r["attack"]["closed_form_check"]["corr"] > 0.9, "SELFTEST FAILED"
        print(compare_runs(f"{tmp}/out").to_string(index=False)); print("SELFTEST OK; outputs in", tmp)
    else:
        kw = {k: v for k, v in vars(a).items() if k != "selftest"}; run(Cfg(**kw))