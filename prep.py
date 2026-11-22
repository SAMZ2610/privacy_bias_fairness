"""prep.py - (1) build meta.csv from MIMIC-CXR-JPG + MIMIC-IV; (2) extract frozen foundation-model embeddings (resumable).
NOTE: MIMIC-CXR-JPG is several hundred GB - do NOT try to hold it in Colab. Use max_patients (e.g. 15000) and download only the
      listed files (build_meta_mimic writes download_list.txt), or read them from your GCS/PhysioNet access.
"""
import os, numpy as np, pandas as pd

LABELS = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
          'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']


def _mode(s):
    m = s.dropna().mode(); return m.iat[0] if len(m) else np.nan


def _race(r):
    r = str(r).upper()
    for k in ("WHITE", "BLACK", "ASIAN", "HISPANIC"):
        if r.startswith(k): return k
    return "UNKNOWN" if any(k in r for k in ("UNKNOWN", "UNABLE", "DECLINED")) else "OTHER"


def build_meta_mimic(cxr_dir, iv_dir, out="meta.csv", max_patients=None, seed=0):
    """cxr_dir: MIMIC-CXR-JPG root (has mimic-cxr-2.0.0-chexpert.csv.gz, -metadata.csv.gz, files/). iv_dir: MIMIC-IV root (hosp/)."""
    lab = pd.read_csv(f"{cxr_dir}/mimic-cxr-2.0.0-chexpert.csv.gz")
    md = pd.read_csv(f"{cxr_dir}/mimic-cxr-2.0.0-metadata.csv.gz", usecols=["dicom_id", "subject_id", "study_id", "ViewPosition"])
    md = md[md.ViewPosition.isin(["AP", "PA"])]                                                    # frontal views only
    pt = pd.read_csv(f"{iv_dir}/hosp/patients.csv.gz", usecols=["subject_id", "gender", "anchor_age"])
    ad = pd.read_csv(f"{iv_dir}/hosp/admissions.csv.gz", usecols=["subject_id", "race", "insurance"])
    adm = ad.groupby("subject_id").agg(race=("race", _mode), insurance=("insurance", _mode)).reset_index()
    df = md.merge(lab, on=["subject_id", "study_id"]).merge(pt, on="subject_id").merge(adm, on="subject_id")
    if max_patients:
        ids = np.random.default_rng(seed).permutation(df.subject_id.unique())[:max_patients]; df = df[df.subject_id.isin(ids)]
    df["rel"] = [f"files/p{str(s)[:2]}/p{s}/s{st}/{d}.jpg" for s, st, d in zip(df.subject_id, df.study_id, df.dicom_id)]
    m = pd.DataFrame(dict(patient_id=df.subject_id.astype(str), path=cxr_dir.rstrip("/") + "/" + df.rel, rel=df.rel, sex=df.gender,
                          age=df.anchor_age, race=df.race.map(_race), insurance=df.insurance.astype(str).str.upper()))
    for l in LABELS: m[l] = df[l].fillna(0.0).to_numpy() if l in df else np.nan                      # blank = not mentioned = negative; -1 (uncertain) kept, ignored later
    m = m.reset_index(drop=True); m.to_csv(out, index=False); m.rel.to_csv(os.path.splitext(out)[0] + "_download_list.txt", index=False, header=False)
    print(f"{out}: {len(m)} images, {m.patient_id.nunique()} patients; race counts: {m.drop_duplicates('patient_id').race.value_counts().to_dict()}")
    return m


def extract_embeddings(meta_csv, model="rad-dino", out="emb.npy", bs=64, workers=2, img_col="path"):
    """model: 'rad-dino' (microsoft/rad-dino, CLS) | 'biomedclip' (open_clip hf-hub BiomedCLIP, image embedding) | 'dummy' (test).
    Resumable: writes into a memmap and a .progress file; re-run the same call to continue."""
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, Subset
    paths = pd.read_csv(meta_csv)[img_col].tolist(); n = len(paths); dev = "cuda" if torch.cuda.is_available() else "cpu"
    if model == "rad-dino":
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained("microsoft/rad-dino"); net = AutoModel.from_pretrained("microsoft/rad-dino").to(dev).eval()
        prep = lambda im: proc(images=im, return_tensors="pt")["pixel_values"][0]; enc = lambda x: net(pixel_values=x).pooler_output
    elif model == "biomedclip":
        import open_clip
        net, _, pre = open_clip.create_model_and_transforms("hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit-base-patch16-224"); net = net.to(dev).eval()
        prep = pre; enc = net.encode_image
    elif model == "dummy":
        prep = lambda im: torch.from_numpy(np.asarray(im.resize((8, 8)), np.float32).reshape(-1) / 255); enc = lambda x: x
    else: raise ValueError(model)

    class DS(Dataset):
        def __len__(s): return n
        def __getitem__(s, i): return prep(Image.open(paths[i]).convert("RGB"))
    tmp, prog = out + ".partial.npy", out + ".progress"; done = int(open(prog).read()) if os.path.exists(prog) and os.path.exists(tmp) else 0
    mm = None; dl = DataLoader(Subset(DS(), range(done, n)), batch_size=bs, num_workers=workers)
    with torch.no_grad():
        for b, x in enumerate(dl):
            with torch.autocast(dev, dtype=torch.float16, enabled=(dev == "cuda")): e = enc(x.to(dev)).float().cpu().numpy()
            if mm is None:
                mm = np.lib.format.open_memmap(tmp, mode="r+" if done else "w+", dtype=np.float32, shape=(n, e.shape[1]))
            lo = done + b * bs; mm[lo:lo + len(e)] = e
            if b % 50 == 0: mm.flush(); open(prog, "w").write(str(lo + len(e))); print(f"\r{lo + len(e)}/{n}", end="")
    mm.flush(); os.replace(tmp, out); os.path.exists(prog) and os.remove(prog); print(f"\nsaved {out}")
    return out


# ---------------------------------------------------------------------------- other datasets (lighter than MIMIC-CXR)
NIH_LABELS = ['No Finding', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
              'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


def build_meta_nih(nih_dir, out="meta_nih.csv", max_patients=None, seed=0):
    """nih_dir: folder with Data_Entry_2017.csv + the PNGs (any layout: images_001/images/*.png or images/*.png). Attributes: sex, age only."""
    de = pd.read_csv(f"{nih_dir}/Data_Entry_2017.csv")
    files = {f: os.path.join(r, f) for r, _, fs in os.walk(nih_dir) for f in fs if f.lower().endswith(".png")}
    de = de[de["Image Index"].isin(files)].reset_index(drop=True)
    if max_patients:
        ids = np.random.default_rng(seed).permutation(de["Patient ID"].unique())[:max_patients]; de = de[de["Patient ID"].isin(ids)].reset_index(drop=True)
    age = pd.to_numeric(de["Patient Age"].astype(str).str.extract(r"(\d+)")[0]); age = age.where(age <= 110)       # a few ages are corrupt (>110)
    m = pd.DataFrame(dict(patient_id=de["Patient ID"].astype(str), path=de["Image Index"].map(files), sex=de["Patient Gender"].astype(str).str.upper().str[0], age=age))
    fl = de["Finding Labels"].str.split("|")
    for l in NIH_LABELS: m[l] = fl.map(lambda x: float(l in x)).to_numpy()
    m.to_csv(out, index=False); print(f"{out}: {len(m)} images, {m.patient_id.nunique()} patients; use attrs=('sex','age_bin')"); return m


def _race_chx(x):
    x = str(x).upper()
    if x in ("NAN", "", "NONE") or any(k in x for k in ("UNKNOWN", "DECLINED", "UNAVAILABLE")): return "UNKNOWN"
    for k in ("WHITE", "BLACK", "ASIAN"):
        if k in x: return k
    return "OTHER"


def build_meta_chexpert(root, demo_file, out="meta_chexpert.csv", max_patients=None, seed=0):
    """root: folder CONTAINING 'CheXpert-v1.0-small'. demo_file: path of 'CHEXPERT DEMO.xlsx' (or .csv) from Stanford AIMI. Attributes: sex, race, age."""
    d = f"{root}/CheXpert-v1.0-small"
    df = pd.concat([pd.read_csv(f"{d}/{s}.csv") for s in ("train", "valid") if os.path.exists(f"{d}/{s}.csv")], ignore_index=True)
    df = df[df["Frontal/Lateral"] == "Frontal"].copy(); df["pid"] = df.Path.str.extract(r"patient(\d+)")[0].astype(int)
    demo = pd.read_excel(demo_file) if str(demo_file).lower().endswith(("xls", "xlsx")) else pd.read_csv(demo_file)
    cp = next(c for c in demo.columns if "PATIENT" in str(c).upper()); cr = next(c for c in demo.columns if "RACE" in str(c).upper())
    demo = demo.assign(pid=demo[cp].astype(str).str.extract(r"(\d+)")[0]).dropna(subset=["pid"]); demo["pid"] = demo.pid.astype(int)
    df["race"] = df.pid.map(demo.drop_duplicates("pid").set_index("pid")[cr]).map(_race_chx)
    if max_patients:
        ids = np.random.default_rng(seed).permutation(df.pid.unique())[:max_patients]; df = df[df.pid.isin(ids)]
    m = pd.DataFrame(dict(patient_id=df.pid.astype(str), path=[os.path.join(root, p) for p in df.Path], sex=df.Sex.astype(str).str.upper().str[0],
                          age=df.Age.where(df.Age > 0), race=df.race))
    for l in LABELS: m[l] = df[l].fillna(0.0).to_numpy() if l in df else np.nan        # blank = not mentioned = negative; -1 (uncertain) is kept and ignored later
    m = m.reset_index(drop=True); m.to_csv(out, index=False)
    print(f"{out}: {len(m)} images, {m.patient_id.nunique()} patients; race: {m.drop_duplicates('patient_id').race.value_counts().to_dict()}; use attrs=('sex','race','age_bin')"); return m