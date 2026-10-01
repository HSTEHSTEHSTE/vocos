Created and reconciled the Vocos environment:

- Prefix: `/home/jhu/xli257/scratch_nandrew9/xli257/miniconda3/envs/vocos`
- Python: `3.10.21`
- Torch: `2.6.0+cu124`
- Torchaudio: `2.6.0+cu124`
- CUDA build: `12.4`
- `torch.version.cuda` asserted non-null.
- `torch.cuda.is_available()` is `False`, expected on this CPU-only node.
- `pip check`: passed.
- Required imports: all passed (`torch`, `torchaudio`, `pytorch_lightning`, `transformers`, `yaml`, `encodec`, `vocos`, `vocos.experiment`, `vocos.pretrained`).

Activate with:

```bash
conda activate "/home/jhu/xli257/scratch_nandrew9/xli257/miniconda3/envs/vocos"
```

No Slurm jobs were submitted.