---
description: "Compare ML model performance across experiments. Loads all available checkpoints and produces a ranked comparison table with statistical commentary."
agent: "agent"
tools: [read, execute]
---

Compare all available ML experiment results and produce a definitive performance report.

## What to Do

1. Load all checkpoint files from `results/checkpoints/*.json`
2. Build a comparison table with these columns:
   - Experiment tag
   - Model name
   - Feature mode
   - MAE (seconds)
   - RMSE (seconds)
   - R² score
   - MAPE (if available)
   - MdAE (if available)

3. Rank models by R² (descending), then by MAE (ascending) as tiebreaker

4. For each completed experiment, provide commentary:
   - Which model wins and by what margin?
   - Is the improvement statistically meaningful (if sample size known)?
   - What does the feature mode contribute?
   - How do tree models compare to DL models?

5. Cross-reference with scheduling results:
   - Higher R² → more accurate SJF-Pred → closer to oracle wait time
   - Quantify the MAE-to-scheduling-improvement relationship

## Output Format

```markdown
## Model Performance Comparison

### Regression Metrics Table
| Experiment | Model | Feature Mode | MAE (s) | RMSE (s) | R² |
| ...

### Key Findings
1. ...

### Tree vs Deep Learning
...

### Implications for Scheduling
...

### Next Experiments to Prioritize
...
```

## Context

See [.github/context/results-summary.md](.github/context/results-summary.md) for
current results snapshot.  
See [.github/context/experiments-map.md](.github/context/experiments-map.md) for
the full experiment matrix.
