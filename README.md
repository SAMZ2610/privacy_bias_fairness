# Encoding Is Not Exposure

Code for the paper **"Encoding Is Not Exposure: Demographic Encoding, Diagnostic Fairness and Membership-Inference Risk in Chest X-ray Foundation Models"** (Sahar Askari, Shiraz University).

We measure three things on the same models and test whether they are aligned:

1. **Demographic encoding**: how well sex, race and age can be decoded from frozen embeddings (out-of-fold linear probes).
2. **Diagnostic fairness**: FNR and underdiagnosis gaps across demographic groups.
3. **Membership-inference (MIA) vulnerability**: a patient-level, closed-form likelihood-ratio audit (following Knolle et al., Nature 2026).

Setup: 200 small heads (linear or MLP) on frozen BiomedCLIP, DINOv2 and RAD-DINO embeddings. Each head is trained on a random half of the audit patients, so every patient is a member of exactly 100 models. Datasets: CheXpert and NIH ChestX-ray14.

## Files

| File | Purpose |
|---|---|
| `prep.py` | Builds `meta.csv` for each dataset (`build_meta_chexpert`, `build_meta_nih`) and extracts resumable frozen embeddings (`extract_embeddings`: RAD-DINO, BiomedCLIP). A MIMIC-CXR builder is included but not used in the paper. |
| `fmpriv.py` | Main pipeline: batched head training, LiRA audit, encoding probes, fairness tables, association analysis, figures. Also has `compare_runs` and a `--selftest`. |
| `bias-privacy-fairness.ipynb` | Kaggle notebook that runs all experiments: seeds, capacity and training-length sweep, LEACE erasure, equivalence tests, DINOv2. |

## Quick start

```bash
pip install torch numpy pandas scipy scikit-learn matplotlib transformers open_clip_torch pillow openpyxl

python fmpriv.py --selftest          # unit tests + synthetic end-to-end check

# 1) metadata and embeddings
python -c "import prep; prep.build_meta_chexpert('<root>', '<CHEXPERT DEMO.xlsx>', 'meta.csv', max_patients=15000); \
           prep.extract_embeddings('meta.csv', model='rad-dino', out='emb_raddino.npy')"

# 2) one audit run (writes outputs/<name>/)
python fmpriv.py --meta meta.csv --emb emb_raddino.npy --name raddino_mlp --head mlp --hidden 512 --lr 1e-3 --epochs 30 --n_models 200
```

Default settings used in the paper: linear head `lr=3e-3`, MLP `lr=1e-3` (512 hidden units, GELU), `wd=1e-2`, 30 epochs, three seeds. Each run saves `paper_numbers.json`, `patient_assoc.csv`, `fairness_table.csv`, `group_table.csv`, `encoding_overall.csv`, `patient_level.csv.gz` and figures. `fmpriv.compare_runs("outputs")` collects them into one table.

## Data

The datasets are not included. Both were obtained from Kaggle mirrors of the original releases:

- CheXpert-v1.0-small: https://www.kaggle.com/datasets/ashery/chexpert (original: Irvin et al., AAAI 2019). Race labels come from the Stanford AIMI `CHEXPERT DEMO.xlsx` file.
- NIH Chest X-rays: https://www.kaggle.com/datasets/nih-chest-xrays/data (original: Wang et al., CVPR 2017).

Please follow the original dataset licences and terms.

## Notes

- The notebook uses Kaggle paths (`/kaggle/input`, `/kaggle/working`); change them for your environment.
- The LEACE erasure and the equivalence-test (TOST) analyses live in the notebook, not in `fmpriv.py`.
- The DINOv2 embeddings reuse the `rad-dino` code path with the model name swapped to `facebook/dinov2-base` (see the last notebook cell).
- RAD-DINO pretraining includes CheXpert, so its CheXpert results are an upper bound, not a clean generalisation test.

## Citation

```bibtex
@article{askari2026encoding,
  author  = {Askari, Sahar},
  title   = {Encoding Is Not Exposure: Demographic Encoding, Diagnostic Fairness and Membership-Inference Risk in Chest {X-ray} Foundation Models},
  year    = {2026},
  note    = {[journal / preprint link to be added]}
}
```

## License

[Add a license, e.g. MIT]
