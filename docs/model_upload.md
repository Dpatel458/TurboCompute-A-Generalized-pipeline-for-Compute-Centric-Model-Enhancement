# Model Upload

## Supported Formats

Users can upload models in:
- H5 format
- Keras format

## Supported Model Types

- FNN (Feedforward Neural Networks)
- CNN (Convolutional Neural Networks)
- Transfer Learning Models:
  - ResNet
  - MobileNet
  - EfficientNet
- Transformer Models

## KEEP Ratio Selection

Users can select a KEEP ratio between 1% and 100%.

KEEP ratio defines how much of the model is retained after pruning.

Example:
- 100% → No pruning
- 50% → Half of parameters retained
- 10% → Highly sparse model

## Special Condition for FNN

If FNN is selected:
- Dataset upload is required.

Other model types do not require dataset upload during pruning.