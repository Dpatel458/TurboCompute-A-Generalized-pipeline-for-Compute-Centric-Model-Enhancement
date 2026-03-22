# Optional Retraining

Retraining is available after pruning.

## Why Retraining?

Pruning may reduce accuracy depending on the KEEP ratio selected. Retraining helps the model recover lost performance and stabilize learning.

## User-Controlled Retraining Settings

When selecting retraining, users can configure:

- Learning Rate
- Number of Epochs

These settings allow better control over stability and recovery.

## Recommended Settings

For most models:

- Suggested Learning Rate: 1e-4
- Adjust epochs depending on model size and convergence behavior.

If retraining becomes unstable:

- Reduce learning rate by 10x
- Increase number of epochs
- Use gradient clipping for stability

## Important Note

Retraining consumes computational resources. It is recommended primarily for:

- Small to medium-sized models
- Moderate KEEP ratios
- When accuracy recovery is required

## After Retraining

Users can:

- Download retrained pruned model
- Compare:
  - Base model
  - Initially pruned model
  - Retrained pruned model
- Analyze performance improvements