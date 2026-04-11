# Contributing — Turb-DETR

## Team Workflow

### Branch Strategy
- `main` — stable, publishable code only
- `dev` — integration branch
- `feature/<name>` — individual work

### Before Every Commit
```bash
# Run tests
pytest tests/ -v

# Check data integrity
python src/utils/data_leak_check.py \
    --train data/splits/train.txt \
    --val data/splits/val.txt \
    --test data/splits/test.txt
```

### Experiment Tracking
- Log ALL training runs to Weights & Biases
- Never overwrite result CSVs — append run ID
- Record: seed, epochs, batch size, GPU type, training time

### Critical Rules
1. **Never regenerate data splits** after experiments begin
2. **Never train on test data** — even accidentally
3. **Never report 50-epoch numbers as final** — publication needs 150 epochs
4. **Always run 3 seeds** for final results (42, 123, 456)
5. **Commit configs before training** — if config isn't in git, results are unreproducible

### Task Ownership (assign per team member)
- [ ] Data pipeline + validation
- [ ] Baseline training + evaluation
- [ ] Jaffe-McGlamery implementation + calibration
- [ ] SimAM injection + Turb-DETR training
- [ ] Paper writing + figures
