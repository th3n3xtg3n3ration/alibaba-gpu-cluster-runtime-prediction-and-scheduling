---
description: "Audit a thesis chapter section for academic rigor, claim accuracy, methodological soundness, and writing quality. Apply Stanford PhD / IEEE/ACM publication standards."
agent: "agent"
argument-hint: "Chapter number (1-7) or section name to review"
tools: [read, search, web]
---

Review the thesis chapter or section provided by the user.

Read [docs/thesis_outline.md](docs/thesis_outline.md) first to understand the full
chapter structure and their interdependencies.

## Review Criteria

### Claim Accuracy (Critical)

Every quantitative claim must be cross-verified against checkpoints:

```
Claim: "XGBoost achieves R²=0.53"
Verify: grep results/checkpoints/exp_a_xgb.json for "r2": 0.53
Status: ✅ Verified / ❌ Discrepancy: actual=0.XX
```

### Methodological Soundness

- Is the train/test split described correctly (80/20, chronological)?
- Are evaluation metrics appropriate for heavy-tailed distributions? (MAE preferred over MSE)
- Are comparisons made on the same test set?
- Are there appropriate baselines (not just FIFO, but also naive heuristics)?
- Is the claim about ~85.8% improvement correctly contextualized (sample size, assumptions)?

### Statistical Rigor

- Are confidence intervals reported where applicable?
- Is the sample size (1,000-job scheduler test) acknowledged as a limitation?
- Is the heavy-tail nature of the distribution addressed in metric selection?
- Are ablation studies present (what happens without utilization features)?

### Related Work Coverage

Check if these seminal works are cited where relevant:
- Borg (Verma et al., 2015) — cluster scheduler
- Tiresias (Gu et al., 2019) — GPU scheduling with LAS
- Gandiva (Xiao et al., 2018) — GPU cluster management
- Ernest (Venkataraman et al., 2016) — performance prediction
- Alibaba trace papers (2017, 2019, 2022)

### Writing Quality

- Are contributions stated clearly and without ambiguity?
- Is the narrative progression logical (problem → method → results → implications)?
- Are all acronyms defined on first use?
- Are figures referenced correctly (Figure N — not just "the figure above")?

## Report Format

```markdown
## Chapter/Section Audit: {Chapter N — Section Title}

### Claim Verification
| Claim | Source | Status |
|-------|--------|--------|
| ... | results/checkpoints/... | ✅ / ❌ |

### Methodological Issues
1. ...

### Missing Citations
1. ...

### Writing Issues
1. ...

### Verdict: PUBLICATION-READY / REVISION NEEDED / MAJOR REVISION
Justification: ...
```
