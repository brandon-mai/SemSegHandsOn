# SemSeg Hands-on

**Sem**antic **Seg**mentation hands-on exercise with BKAI-IGH NeoPolyp
competition dataset.

Model checkpoint is available at
[Google Drive](https://drive.google.com/file/d/14BpXrgZhgr5dPmE6Xt6lNL33N0McNy59/view?usp=sharing).
Please download and place it in the root directory of this repository (same level as
`infer.py`).

Clone repository
```bash
git clone https://github.com/brandon-mai/SemSegHandsOn.git
```

Change directory
```bash
cd SemSegHandsOn
```

Install dependencies (optional)
```bash
pip install -r requirements.txt
```

Infer
```bash
python3 infer.py --image_path image.jpeg
```