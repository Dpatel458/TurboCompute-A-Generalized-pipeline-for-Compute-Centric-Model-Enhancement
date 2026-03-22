# Proteus Optimization Tips

These tips help you achieve the best balance between compression and accuracy.

---

## 1. Start Conservative

Begin with a high KEEP ratio:

- approx. 90%

Gradually decrease based on validation accuracy and performance metrics.

Avoid aggressive pruning in the first attempt.

---

## 2. Monitor Validation Closely

Track:

- Validation loss
- Accuracy trends
- Generalization gap

Do not rely only on training accuracy.

Look for stability before reducing KEEP ratio further.

---

## 3. Stability First

If retraining becomes unstable:

- Reduce learning rate by 10x
- Increase number of epochs
- Apply gradient clipping

Recommended learning rate for most models:

- 1e-4

Always prioritize stable convergence over aggressive compression.

---

## 4. Compare After Every Run

After each pruning cycle:

- Compare accuracy drop
- Compare compression gain
- Measure inference speed

Find the sweet spot between:

- Performance
- Model size
- Speed

Do not optimize only for sparsity.

---

## 5. Layer-wise Sensitivity Analysis

Different layers have different sensitivity to pruning.

Use the Architecture Viewer to:

- Identify robust layers
- Identify sensitive layers
- Inspect attention layers in transformers
- Analyze convolutional or dense layer behavior

Not all layers should be pruned equally.

Layer-wise understanding improves pruning decisions.