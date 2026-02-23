import os
import time
import numpy as np
import tensorflow as tf
from matplotlib.patches import Circle, Wedge, FancyBboxPatch, Ellipse
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches

# ---- Server safe backend ----
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from matplotlib.colors import LinearSegmentedColormap

# ---- NeuralPrune dark theme colors ----
THEME_COLORS = {
    "base": "#8b5cf6",      # violet
    "pruned": "#22d3ee",    # cyan
    "retrained": "#10b981", # emerald
    "accent": "#f59e0b",    # amber
    "bg": "#0f172a",        # slate-900
    "grid": "#334155",      # slate-700
    "text": "#e2e8f0",      # slate-200
    "danger": "#ef4444",    # red
    "surface": "#1e293b",   # slate-800
}

THEME_PALETTE = [
    THEME_COLORS["base"],
    THEME_COLORS["pruned"],
    THEME_COLORS["retrained"],
    THEME_COLORS["accent"],
]

# Create custom gradients
def create_gradient_cmap(color1, color2, name="custom"):
    return LinearSegmentedColormap.from_list(name, [color1, color2])

GRADIENT_CMAP = create_gradient_cmap(THEME_COLORS["base"], THEME_COLORS["pruned"], "base_to_pruned")

# ---- Seaborn global config for dark website contrast ----
sns.set_theme(
    style="darkgrid",
    context="notebook",
    palette=THEME_PALETTE,
    font_scale=1.05,
    rc={
        "figure.facecolor": THEME_COLORS["bg"],
        "axes.facecolor": THEME_COLORS["bg"],
        "axes.edgecolor": THEME_COLORS["grid"],
        "axes.labelcolor": THEME_COLORS["text"],
        "axes.titlecolor": THEME_COLORS["text"],
        "xtick.color": THEME_COLORS["text"],
        "ytick.color": THEME_COLORS["text"],
        "grid.color": THEME_COLORS["grid"],
        "text.color": THEME_COLORS["text"],
    },
)

tf.get_logger().setLevel("ERROR")


class _CompatRandomFlip(layers.RandomFlip):
    def __init__(self, *args, data_format=None, **kwargs):
        super().__init__(*args, **kwargs)


class _CompatRandomRotation(layers.RandomRotation):
    def __init__(self, *args, data_format=None, **kwargs):
        super().__init__(*args, **kwargs)


class _CompatRandomZoom(layers.RandomZoom):
    def __init__(self, *args, data_format=None, **kwargs):
        super().__init__(*args, **kwargs)


_LOAD_CUSTOM_OBJECTS = {
    "RandomFlip": _CompatRandomFlip,
    "RandomRotation": _CompatRandomRotation,
    "RandomZoom": _CompatRandomZoom,
}


def _load_model_safe(path):
    return load_model(path, compile=False, custom_objects=_LOAD_CUSTOM_OBJECTS)

# ===================== BASIC METRICS (UNCHANGED LOGIC) =====================

def _count_params(model):
    return model.count_params()


def _model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def _get_input_shape_from_model(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    return tuple(1 if s is None else s for s in shape)


def _inference_time_ms(model, runs=10):
    input_shape = _get_input_shape_from_model(model)
    dummy = tf.random.normal(input_shape)

    start = time.time()
    for _ in range(runs):
        model(dummy, training=False)
    return (time.time() - start) / runs * 1000


def _nonzero_ratio(model):
    total, nonzero = 0, 0
    for w in model.get_weights():
        total += w.size
        nonzero += np.count_nonzero(w)
    return nonzero / total if total else 0


def _graph_density(model):
    densities = []
    for layer in model.layers:
        w = layer.get_weights()
        if w:
            densities.append(np.count_nonzero(w[0]) / w[0].size)
    return float(np.mean(densities)) if densities else 0


def _layer_sparsity(model):
    sparsity = {}
    for layer in model.layers:
        w = layer.get_weights()
        if w:
            sparsity[layer.name] = 1 - np.count_nonzero(w[0]) / w[0].size
    return sparsity


def _per_layer_param_reduction(base, pruned):
    reduction = {}
    pruned_by_name = {l.name: l for l in pruned.layers}

    for l1 in base.layers:
        l2 = pruned_by_name.get(l1.name)
        if not l2:
            continue

        w1 = l1.get_weights()
        w2 = l2.get_weights()
        if w1 and w2:
            reduction[l1.name] = w1[0].size - np.count_nonzero(w2[0])

    return reduction


def _weight_magnitude_retention(base, pruned, steps=200):
    base_w = np.abs(np.concatenate([w.flatten() for w in base.get_weights()]))
    pruned_w = np.abs(np.concatenate([w.flatten() for w in pruned.get_weights()]))

    base_sorted = np.sort(base_w)[::-1]
    pruned_sorted = np.sort(pruned_w)[::-1]

    base_cum = np.cumsum(base_sorted)
    pruned_cum = np.cumsum(pruned_sorted)

    base_cum /= base_cum[-1]
    pruned_cum /= pruned_cum[-1]

    x = np.linspace(0, 100, steps)
    base_idx = (x / 100 * (len(base_cum) - 1)).astype(int)
    pruned_idx = (x / 100 * (len(pruned_cum) - 1)).astype(int)

    return x, base_cum[base_idx] * 100, pruned_cum[pruned_idx] * 100


def _get_layer_weights_flat(model):
    """Extract all weights flattened for violin plots"""
    weights = []
    for w in model.get_weights():
        weights.extend(w.flatten())
    return np.array(weights)


def _get_layer_magnitudes_by_layer(model):
    """Get weight magnitudes organized by layer for heatmap"""
    layer_data = {}
    for i, layer in enumerate(model.layers):
        w = layer.get_weights()
        if w:
            # Sample weights for visualization (max 1000 per layer)
            flat = np.abs(w[0].flatten())
            if len(flat) > 1000:
                indices = np.linspace(0, len(flat)-1, 1000, dtype=int)
                flat = flat[indices]
            layer_data[layer.name[:15]] = flat  # Truncate long names
    return layer_data

# ===================== CREATIVE VISUALIZATIONS =====================

def _create_radar_chart(path, title, categories, values_base, values_pruned, values_retrained=None):
    """1. RADAR CHART - Multi-metric comparison with glowing effect"""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    fig.patch.set_facecolor(THEME_COLORS["bg"])
    ax.set_facecolor(THEME_COLORS["surface"])
   
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
   
    # Normalize values to 0-1 scale for radar
    max_vals = np.maximum(values_base, values_pruned)
    if values_retrained is not None:
        max_vals = np.maximum(max_vals, values_retrained)
    max_vals = np.maximum(max_vals, 1e-8)  # Avoid division by zero
   
    values_base_norm = np.concatenate([values_base/max_vals, [values_base[0]/max_vals[0]]])
    values_pruned_norm = np.concatenate([values_pruned/max_vals, [values_pruned[0]/max_vals[0]]])
   
    # Plot with glow effect
    ax.plot(angles, values_base_norm, 'o-', linewidth=3, color=THEME_COLORS["base"], label='Base', markersize=8)
    ax.fill(angles, values_base_norm, alpha=0.15, color=THEME_COLORS["base"])
   
    ax.plot(angles, values_pruned_norm, 'o-', linewidth=3, color=THEME_COLORS["pruned"], label='Pruned', markersize=8)
    ax.fill(angles, values_pruned_norm, alpha=0.15, color=THEME_COLORS["pruned"])
   
    if values_retrained is not None:
        values_retrained_norm = np.concatenate([values_retrained/max_vals, [values_retrained[0]/max_vals[0]]])
        ax.plot(angles, values_retrained_norm, 'o-', linewidth=3, color=THEME_COLORS["retrained"], label='Retrained', markersize=8)
        ax.fill(angles, values_retrained_norm, alpha=0.15, color=THEME_COLORS["retrained"])
   
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color=THEME_COLORS["text"], size=11)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], color=THEME_COLORS["grid"], size=9)
    ax.grid(True, color=THEME_COLORS["grid"], alpha=0.3)
   
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), facecolor=THEME_COLORS["surface"])
    plt.title(title, size=16, color=THEME_COLORS["text"], pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=THEME_COLORS["bg"])
    plt.close()


def _create_donut_comparison(path, title, base_val, pruned_val, label_base="Base", label_pruned="Pruned",
                             unit="MB", center_text=""):
    """2. DONUT CHARTS - Modern circular progress indicators"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor(THEME_COLORS["bg"])
   
    # Calculate percentages for visualization
    max_val = max(base_val, pruned_val) * 1.2
    base_pct = base_val / max_val
    pruned_pct = pruned_val / max_val
   
    for ax, val, pct, color, label in [(ax1, base_val, base_pct, THEME_COLORS["base"], label_base),
                                        (ax2, pruned_val, pruned_pct, THEME_COLORS["pruned"], label_pruned)]:
        ax.set_facecolor(THEME_COLORS["bg"])
       
        # Outer ring (background)
        theta = np.linspace(0, 2*np.pi, 100)
        r_outer = 1.0
        x_outer = r_outer * np.cos(theta)
        y_outer = r_outer * np.sin(theta)
        ax.fill(x_outer, y_outer, color=THEME_COLORS["surface"], alpha=0.5)
       
        # Progress arc
        theta_fill = np.linspace(-np.pi/2, -np.pi/2 + 2*np.pi*pct, 100)
        x_fill = 0.85 * np.cos(theta_fill)
        y_fill = 0.85 * np.sin(theta_fill)
        ax.fill_between(x_fill, y_fill, alpha=0.8, color=color)
       
        # Inner circle
        circle = plt.Circle((0, 0), 0.6, color=THEME_COLORS["bg"])
        ax.add_patch(circle)
       
        # Text
        ax.text(0, 0.1, f"{val:.2f}", ha='center', va='center', fontsize=24,
                color=color, fontweight='bold')
        ax.text(0, -0.2, unit, ha='center', va='center', fontsize=12, color=THEME_COLORS["text"])
        ax.text(0, -0.5, label, ha='center', va='center', fontsize=14, color=THEME_COLORS["text"])
       
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
   
    # Center text between donuts
    if center_text:
        fig.text(0.5, 0.85, center_text, ha='center', fontsize=14, color=THEME_COLORS["accent"], fontweight='bold')
   
    plt.suptitle(title, fontsize=16, color=THEME_COLORS["text"], y=0.98, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=THEME_COLORS["bg"])
    plt.close()


def _create_violin_distribution(path, title, base_weights, pruned_weights, retrained_weights=None):
    """3. VIOLIN PLOT - Sophisticated distribution comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(THEME_COLORS["bg"])
    ax.set_facecolor(THEME_COLORS["surface"])
   
    # Prepare data
    data = [base_weights, pruned_weights]
    labels = ['Base Model', 'Pruned Model']
    colors = [THEME_COLORS["base"], THEME_COLORS["pruned"]]
   
    if retrained_weights is not None:
        data.append(retrained_weights)
        labels.append('Retrained Model')
        colors.append(THEME_COLORS["retrained"])
   
    # Create violin plot
    parts = ax.violinplot(data, positions=range(len(data)), showmeans=True, showmedians=True)
   
    # Style violins
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
        pc.set_edgecolor(color)
        pc.set_linewidth(2)
   
    for partname in ('cbars','cmins','cmaxes','cmeans','cmedians'):
        if partname in parts:
            parts[partname].set_color(THEME_COLORS["text"])
            parts[partname].set_linewidth(1.5)
   
    # Add scatter points for outliers (sample)
    for i, (d, color) in enumerate(zip(data, colors)):
        # Sample for visualization
        sample = np.random.choice(d, min(500, len(d)), replace=False)
        jitter = np.random.normal(i, 0.04, size=len(sample))
        ax.scatter(jitter, sample, alpha=0.3, s=10, color=color)
   
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, color=THEME_COLORS["text"], fontsize=12)
    ax.set_ylabel('Weight Values', color=THEME_COLORS["text"], fontsize=12)
    ax.set_title(title, fontsize=16, color=THEME_COLORS["text"], pad=20, fontweight='bold')
    ax.grid(True, alpha=0.2, color=THEME_COLORS["grid"])
    ax.tick_params(colors=THEME_COLORS["text"])
   
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=THEME_COLORS["bg"])
    plt.close()


def _create_heatmap_sparsity(path, title, model, is_comparison=False, model2=None):
    """4. HEATMAP - Layer-wise weight magnitude visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(THEME_COLORS["bg"])
    ax.set_facecolor(THEME_COLORS["surface"])
   
    # Extract layer data
    layer_names = []
    weight_matrix = []
   
    for layer in model.layers:
        w = layer.get_weights()
        if w:
            layer_names.append(layer.name[:20])
            # Create histogram of weight magnitudes for this layer
            magnitudes = np.abs(w[0].flatten())
            hist, _ = np.histogram(magnitudes, bins=50, range=(0, np.percentile(magnitudes, 99)))
            weight_matrix.append(hist)
   
    if not weight_matrix:
        weight_matrix = [[0]*50]
        layer_names = ["No Data"]
   
    weight_matrix = np.array(weight_matrix)
   
    # Normalize for color mapping
    weight_matrix = weight_matrix / (weight_matrix.max(axis=1, keepdims=True) + 1e-8)
   
    # Create heatmap
    im = ax.imshow(weight_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
   
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Frequency', color=THEME_COLORS["text"])
    cbar.ax.yaxis.set_tick_params(color=THEME_COLORS["text"])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=THEME_COLORS["text"])
   
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names, color=THEME_COLORS["text"], fontsize=9)
    ax.set_xlabel('Weight Magnitude Bins (Low → High)', color=THEME_COLORS["text"], fontsize=12)
    ax.set_title(title, fontsize=16, color=THEME_COLORS["text"], pad=20, fontweight='bold')
    ax.tick_params(colors=THEME_COLORS["text"])
   
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=THEME_COLORS["bg"])
    plt.close()


def _create_bubble_chart(path, title, base_model, pruned_model, retrained_model=None):
    """5. BUBBLE CHART - Layer analysis with 3D info (x=layer, y=sparsity, size=params)"""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(THEME_COLORS["bg"])
    ax.set_facecolor(THEME_COLORS["surface"])
   
    def extract_bubble_data(model, color, label, alpha=0.6):
        x, y, sizes = [], [], []
        for i, layer in enumerate(model.layers):
            w = layer.get_weights()
            if w:
                x.append(i)
                sparsity = 1 - np.count_nonzero(w[0]) / w[0].size
                y.append(sparsity * 100)
                # Size based on parameter count (scaled for visibility)
                size = np.log10(w[0].size + 1) * 300
                sizes.append(size)
       
        if x:
            scatter = ax.scatter(x, y, s=sizes, c=color, alpha=alpha, edgecolors='white',
                               linewidth=1.5, label=label)
        return x, y
   
    x_base, y_base = extract_bubble_data(base_model, THEME_COLORS["base"], 'Base Model', 0.4)
    x_pruned, y_pruned = extract_bubble_data(pruned_model, THEME_COLORS["pruned"], 'Pruned Model', 0.7)
   
    if retrained_model:
        x_retrained, y_retrained = extract_bubble_data(retrained_model, THEME_COLORS["retrained"], 'Retrained Model', 0.7)
   
    ax.set_xlabel('Layer Index', color=THEME_COLORS["text"], fontsize=12)
    ax.set_ylabel('Sparsity (%)', color=THEME_COLORS["text"], fontsize=12)
    ax.set_title(title + '\n(Bubble size = Parameter count)', fontsize=14, color=THEME_COLORS["text"], pad=20, fontweight='bold')
    ax.grid(True, alpha=0.2, color=THEME_COLORS["grid"])
    ax.tick_params(colors=THEME_COLORS["text"])
    ax.legend(loc='upper left', facecolor=THEME_COLORS["surface"], edgecolor=THEME_COLORS["grid"])
   
    # Add annotation
    ax.text(0.02, 0.98, 'Bubble size ∝ log(Parameters)', transform=ax.transAxes,
            fontsize=10, color=THEME_COLORS["text"], alpha=0.7, va='top')
   
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=THEME_COLORS["bg"])
    plt.close()


def _create_gauge_efficiency(path, title, base_params, pruned_params, base_time, pruned_time):
    """6. GAUGE CHARTS - Efficiency scores with speedup metrics"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(THEME_COLORS["bg"])
   
    # Calculate metrics
    compression_ratio = base_params / pruned_params if pruned_params > 0 else 1
    speedup = base_time / pruned_time if pruned_time > 0 else 1
    efficiency_score = (compression_ratio + speedup) / 2
   
    metrics = [
        ("Compression", compression_ratio, "x", 10),
        ("Speedup", speedup, "x", 5),
        ("Efficiency", efficiency_score, "score", 10)
    ]
   
    colors = [THEME_COLORS["pruned"], THEME_COLORS["accent"], THEME_COLORS["retrained"]]
   
    for ax, (name, value, unit, max_val), color in zip(axes, metrics, colors):
        ax.set_facecolor(THEME_COLORS["bg"])
       
        # Create semi-circle gauge
        theta = np.linspace(0, np.pi, 100)
        r = 1.0
       
        # Background arc
        ax.fill_between(np.cos(theta), np.sin(theta), 0, alpha=0.2, color=THEME_COLORS["grid"])
       
        # Value arc
        value_ratio = min(value / max_val, 1.0)
        theta_value = np.linspace(0, np.pi * value_ratio, 100)
        ax.fill_between(np.cos(theta_value), np.sin(theta_value), 0, alpha=0.8, color=color)
       
        # Needle
        needle_angle = np.pi * value_ratio
        ax.plot([0, 0.8*np.cos(needle_angle)], [0, 0.8*np.sin(needle_angle)],
                color='white', linewidth=4)
        ax.plot([0, 0.8*np.cos(needle_angle)], [0, 0.8*np.sin(needle_angle)],
                color=color, linewidth=2)
       
        # Center dot
        circle = plt.Circle((0, 0), 0.05, color='white')
        ax.add_patch(circle)
       
        # Text
        ax.text(0, -0.3, f"{value:.2f}{unit}", ha='center', va='center',
                fontsize=24, color=color, fontweight='bold')
        ax.text(0, -0.6, name, ha='center', va='center', fontsize=14, color=THEME_COLORS["text"])
       
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.8, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
   
    plt.suptitle(title, fontsize=18, color=THEME_COLORS["text"], y=0.95, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=THEME_COLORS["bg"])
    plt.close()


def _create_sankey_flow(path, title, base_params, pruned_params):
    """7. SANKEY-STYLE DIAGRAM - Parameter flow visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(THEME_COLORS["bg"])
    ax.set_facecolor(THEME_COLORS["bg"])
   
    removed_params = base_params - pruned_params
   
    # Create flow visualization using bars and connections
    y_positions = [0.7, 0.3]
    labels = ['Base Model\nParameters', 'Pruned Model\nParameters', 'Removed\nParameters']
    values = [base_params, pruned_params, removed_params]
    colors = [THEME_COLORS["base"], THEME_COLORS["pruned"], THEME_COLORS["danger"]]
   
    # Normalize for display
    max_val = max(values)
    widths = [v/max_val * 0.3 for v in values]
   
    # Draw nodes
    for i, (label, val, color, width) in enumerate(zip(labels, values, colors, widths)):
        if i < 2:
            x = 0.2 if i == 0 else 0.8
            y = 0.5
           
            # Node rectangle
            rect = FancyBboxPatch((x-width/2, y-width/2), width, width,
                                  boxstyle="round,pad=0.01",
                                  facecolor=color, edgecolor='white', linewidth=2, alpha=0.8)
            ax.add_patch(rect)
           
            # Label
            ax.text(x, y+width/2+0.08, label, ha='center', va='bottom', fontsize=11,
                   color=THEME_COLORS["text"], fontweight='bold')
            ax.text(x, y, f"{val/1e6:.2f}M", ha='center', va='center', fontsize=13,
                   color='white', fontweight='bold')
        else:
            # Removed params (side node)
            x, y = 0.5, 0.85
            rect = FancyBboxPatch((x-width/2, y-width/2), width, width,
                                  boxstyle="round,pad=0.01",
                                  facecolor=color, edgecolor='white', linewidth=2, alpha=0.8)
            ax.add_patch(rect)
            ax.text(x, y+width/2+0.05, label, ha='center', va='bottom', fontsize=10,
                   color=THEME_COLORS["text"])
            ax.text(x, y, f"{val/1e6:.2f}M", ha='center', va='center', fontsize=11,
                   color='white', fontweight='bold')
   
    # Draw flow arrows using polygons
    # Base to Pruned
    ax.annotate('', xy=(0.8-0.15, 0.5), xytext=(0.2+0.15, 0.5),
                arrowprops=dict(arrowstyle='->', color=THEME_COLORS["pruned"], lw=3))
   
    # Base to Removed (curved)
    ax.annotate('', xy=(0.5, 0.85-0.1), xytext=(0.2+0.1, 0.5+0.1),
                arrowprops=dict(arrowstyle='->', color=THEME_COLORS["danger"], lw=2,
                               connectionstyle="arc3,rad=0.3"))
   
    # Percentage labels
    retention_pct = (pruned_params / base_params) * 100
    removal_pct = (removed_params / base_params) * 100
   
    ax.text(0.5, 0.5, f"{retention_pct:.1f}%\nretained", ha='center', va='center',
            fontsize=12, color=THEME_COLORS["pruned"], fontweight='bold',
            bbox=dict(boxstyle='round', facecolor=THEME_COLORS["bg"], alpha=0.8))
   
    ax.text(0.35, 0.7, f"{removal_pct:.1f}%\nremoved", ha='center', va='center',
            fontsize=10, color=THEME_COLORS["danger"],
            bbox=dict(boxstyle='round', facecolor=THEME_COLORS["bg"], alpha=0.8))
   
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
   
    plt.title(title, fontsize=16, color=THEME_COLORS["text"], pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=THEME_COLORS["bg"])
    plt.close()


def _create_network_graph(path, title, model):
    """8. NETWORK GRAPH - Visual representation of layer connectivity"""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(THEME_COLORS["bg"])
    ax.set_facecolor(THEME_COLORS["bg"])
   
    layers_list = [l for l in model.layers if l.get_weights()]
    n_layers = len(layers_list)
   
    if n_layers == 0:
        ax.text(0.5, 0.5, "No layers to visualize", ha='center', va='center',
               color=THEME_COLORS["text"], fontsize=14)
        plt.savefig(path, dpi=150, facecolor=THEME_COLORS["bg"])
        plt.close()
        return
   
    # Position layers horizontally
    x_positions = np.linspace(0.1, 0.9, n_layers)
    y_center = 0.5
   
    node_positions = []
   
    for i, (x, layer) in enumerate(zip(x_positions, layers_list)):
        w = layer.get_weights()[0]
        sparsity = 1 - np.count_nonzero(w) / w.size
        density = 1 - sparsity
       
        # Node size based on parameter count
        size = np.log10(w.size + 1) * 0.03
       
        # Color based on density
        color_intensity = density
        color = plt.cm.viridis(color_intensity)
       
        # Draw node
        circle = plt.Circle((x, y_center), size, color=color, alpha=0.8, ec='white', linewidth=2)
        ax.add_patch(circle)
       
        # Layer name
        ax.text(x, y_center-size-0.05, layer.name[:10], ha='center', va='top',
               fontsize=8, color=THEME_COLORS["text"])
       
        # Density label
        ax.text(x, y_center, f"{density*100:.0f}%", ha='center', va='center',
               fontsize=9, color='white', fontweight='bold')
       
        node_positions.append((x, y_center, size))
   
    # Draw connections
    for i in range(len(node_positions)-1):
        x1, y1, s1 = node_positions[i]
        x2, y2, s2 = node_positions[i+1]
       
        # Draw curved connection
        ax.annotate('', xy=(x2-s2, y2), xytext=(x1+s1, y1),
                   arrowprops=dict(arrowstyle='-', color=THEME_COLORS["grid"],
                                  lw=1.5, alpha=0.5))
   
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Density %', color=THEME_COLORS["text"])
    cbar.ax.yaxis.set_tick_params(color=THEME_COLORS["text"])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=THEME_COLORS["text"])
   
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
   
    plt.title(title, fontsize=16, color=THEME_COLORS["text"], pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=THEME_COLORS["bg"])
    plt.close()


def _create_stacked_area(path, title, base_model, pruned_model):
    """9. STACKED AREA - Cumulative parameter distribution"""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(THEME_COLORS["bg"])
    ax.set_facecolor(THEME_COLORS["surface"])
   
    # Get cumulative parameters per layer
    def get_cumulative_params(model):
        params = []
        cumsum = 0
        for layer in model.layers:
            w = layer.get_weights()
            if w:
                cumsum += w[0].size
                params.append(cumsum)
        return np.array(params)
   
    base_cum = get_cumulative_params(base_model)
    pruned_cum = get_cumulative_params(pruned_model)
   
    # Normalize to percentages
    base_total = base_cum[-1] if len(base_cum) > 0 else 1
    pruned_total = pruned_cum[-1] if len(pruned_cum) > 0 else 1
   
    base_pct = base_cum / base_total * 100
    pruned_pct = pruned_cum / pruned_total * 100
   
    x_base = np.linspace(0, 100, len(base_pct))
    x_pruned = np.linspace(0, 100, len(pruned_pct))
   
    # Create filled areas
    ax.fill_between(x_base, 0, base_pct, alpha=0.4, color=THEME_COLORS["base"], label='Base Model')
    ax.fill_between(x_pruned, 0, pruned_pct, alpha=0.6, color=THEME_COLORS["pruned"], label='Pruned Model')
   
    # Add lines on top
    ax.plot(x_base, base_pct, color=THEME_COLORS["base"], linewidth=2)
    ax.plot(x_pruned, pruned_pct, color=THEME_COLORS["pruned"], linewidth=2)
   
    ax.set_xlabel('Layer Progression (%)', color=THEME_COLORS["text"], fontsize=12)
    ax.set_ylabel('Cumulative Parameters (%)', color=THEME_COLORS["text"], fontsize=12)
    ax.set_title(title, fontsize=16, color=THEME_COLORS["text"], pad=20, fontweight='bold')
    ax.legend(loc='lower right', facecolor=THEME_COLORS["surface"], edgecolor=THEME_COLORS["grid"])
    ax.grid(True, alpha=0.2, color=THEME_COLORS["grid"])
    ax.tick_params(colors=THEME_COLORS["text"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
   
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=THEME_COLORS["bg"])
    plt.close()


def _create_retention_gradient(path, title, base, pruned):
    """10. GRADIENT RETENTION CURVE - Enhanced line plot with gradient fill"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(THEME_COLORS["bg"])
    ax.set_facecolor(THEME_COLORS["surface"])
   
    x, base_cum, pruned_cum = _weight_magnitude_retention(base, pruned)
   
    # Plot lines with glow effect
    ax.plot(x, base_cum, linewidth=3, color=THEME_COLORS["base"], label='Base Model')
    ax.plot(x, pruned_cum, linewidth=3, color=THEME_COLORS["pruned"], label='Pruned Model')
   
    # Add gradient fill
    ax.fill_between(x, base_cum, alpha=0.3, color=THEME_COLORS["base"])
    ax.fill_between(x, pruned_cum, alpha=0.3, color=THEME_COLORS["pruned"])
   
    # Add difference area highlighting
    diff = np.array(base_cum) - np.array(pruned_cum)
    ax.fill_between(x, pruned_cum, base_cum, where=(diff > 0), alpha=0.2,
                    color=THEME_COLORS["accent"], label='Difference')
   
    ax.set_xlabel('Percentage of Weights Kept', color=THEME_COLORS["text"], fontsize=12)
    ax.set_ylabel('Cumulative Weight Magnitude (%)', color=THEME_COLORS["text"], fontsize=12)
    ax.set_title(title, fontsize=16, color=THEME_COLORS["text"], pad=20, fontweight='bold')
    ax.legend(loc='lower right', facecolor=THEME_COLORS["surface"], edgecolor=THEME_COLORS["grid"])
    ax.grid(True, alpha=0.2, color=THEME_COLORS["grid"])
    ax.tick_params(colors=THEME_COLORS["text"])
   
    # Add annotations
    ax.axhline(y=80, color=THEME_COLORS["accent"], linestyle='--', alpha=0.5)
    ax.text(5, 82, '80% threshold', color=THEME_COLORS["accent"], fontsize=10)
   
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=THEME_COLORS["bg"])
    plt.close()


# ===================== GRAPH CONTROLLER =====================

def graph_count(flops_pair=None):
    return 10 if flops_pair and None not in flops_pair else 9


def graph_count_retrain():
    return 9


_GRAPH_DESCRIPTIONS = {
    0: (
        "This view shows how lean the model becomes after pruning.\n"
        "A big drop in parameters means lighter memory use and simpler deployment.\n"
        "The gap between bars reflects true structural compression.\n"
        "If performance stays strong with fewer parameters, the pruning strategy is high quality.\n"
        "This is your first signal that efficiency gains are real, not cosmetic."
    ),
    1: (
        "This chart tracks model file size before and after pruning.\n"
        "Smaller size means faster sharing, faster loading, and easier edge rollout.\n"
        "It converts architecture changes into practical storage savings.\n"
        "When size shrinks without accuracy loss, compression is doing its job.\n"
        "Think of this as deployment readiness in MB."
    ),
    2: (
        "This graph compares prediction speed in milliseconds.\n"
        "Lower latency means snappier responses and stronger real-time behavior.\n"
        "Pruning should cut heavy computation paths and speed up inference.\n"
        "If speed does not improve, runtime overhead or hardware bottlenecks may be limiting gains.\n"
        "This metric speaks directly to user experience in production."
    ),
    3: (
        "This histogram reveals how pruning reshapes weight values.\n"
        "A healthy shift toward zero suggests cleaner sparsity and reduced complexity.\n"
        "It confirms whether low-impact weights were removed as intended.\n"
        "Extreme distortion can warn that pruning was too aggressive.\n"
        "A smooth shift usually indicates disciplined, stable compression."
    ),
    4: (
        "This metric shows how much active capacity the model still uses.\n"
        "Lower non-zero ratio means stronger sparsity and deeper compression.\n"
        "The target is efficient capacity, not blind capacity loss.\n"
        "Pair this with accuracy to see if pruning stayed balanced.\n"
        "Great pruning keeps the model sharp while trimming excess."
    ),
    5: (
        "This chart compares overall network density across layers.\n"
        "Lower density means fewer active connections and lighter compute cost.\n"
        "It captures structural simplification at a whole-model level.\n"
        "A controlled reduction points to intentional pruning, not random degradation.\n"
        "Use it as a structural health check after compression."
    ),
    6: (
        "This graph maps sparsity layer by layer after pruning.\n"
        "It highlights where compression was aggressive and where it stayed conservative.\n"
        "Uneven patterns can expose fragile blocks or optimization bottlenecks.\n"
        "Strong sparsity in resilient layers is usually a good sign.\n"
        "It guides precise fine-tuning instead of guesswork."
    ),
    7: (
        "This chart shows exactly where parameters were removed versus the base model.\n"
        "It identifies the layers carrying most of the compression load.\n"
        "Large drops in key layers can explain both gains and regressions.\n"
        "If sensitive blocks are over-pruned, accuracy usually suffers first.\n"
        "Use this to tune pruning policy with layer-level precision."
    ),
    8: (
        "This curve tracks how much important weight mass is retained.\n"
        "When curves stay close, pruning preserved high-value signal.\n"
        "Wider separation suggests information was sacrificed for compression.\n"
        "It gives a richer view than sparsity alone.\n"
        "This is a strong quality indicator for pruning fidelity."
    ),
    9: (
        "This graph compares estimated compute cost (FLOPs) per forward pass.\n"
        "Lower FLOPs usually translate into faster and cheaper inference.\n"
        "The reduction quantifies the real compute savings from pruning.\n"
        "It is a core KPI for edge, cloud, and high-throughput serving.\n"
        "This is where optimization meets operational impact."
    ),
}


_GRAPH_DESCRIPTIONS_RETRAIN = {
    0: (
        "This graph compares parameter count across base, pruned, and retrained stages.\n"
        "Pruned and retrained counts should remain close if structure is preserved.\n"
        "Stable counts mean retraining improved weights, not model size.\n"
        "That is the ideal path: better quality without added complexity.\n"
        "Use it to confirm compression gains survived retraining."
    ),
    1: (
        "This chart tracks file size through base, pruned, and retrained models.\n"
        "Pruning should drive size down sharply from the baseline.\n"
        "Retraining should keep size close, with only minor shifts.\n"
        "If size stays low, fine-tuning preserved deployment efficiency.\n"
        "It validates that quality recovery did not reintroduce bloat."
    ),
    2: (
        "This graph compares inference speed across all three stages.\n"
        "Pruned and retrained models should remain faster than the base model.\n"
        "Best-case outcome: retraining recovers accuracy while keeping speed gains.\n"
        "If retrained latency regresses, pruning benefits may be partially diluted.\n"
        "It captures the real-world tradeoff after fine-tuning."
    ),
    3: (
        "This histogram shows how weight distributions evolve across stages.\n"
        "Pruning pushes weights toward sparsity; retraining should refine, not destabilize.\n"
        "A moderate shift after retraining is usually healthy adaptation.\n"
        "Large shifts can hint at instability or overfitting.\n"
        "It explains how fine-tuning reshapes compressed representations."
    ),
    4: (
        "This metric compares effective active capacity through the pipeline.\n"
        "Pruning should reduce capacity from the baseline.\n"
        "Retraining may recover a little capacity while staying compact.\n"
        "That pattern usually signals better accuracy without full complexity return.\n"
        "It summarizes the sparsity-performance balance after retraining."
    ),
    5: (
        "This graph compares average graph density across all stages.\n"
        "Pruning should lower density and simplify structure.\n"
        "Retraining can slightly increase density if it improves generalization.\n"
        "Large density rebound may weaken compression benefits.\n"
        "It tells you whether structural efficiency held after fine-tuning."
    ),
    6: (
        "This graph compares layer-wise sparsity between pruned and retrained models.\n"
        "It reveals where retraining preserved sparsity and where it relaxed it.\n"
        "Stable patterns usually indicate controlled, robust fine-tuning.\n"
        "Large layer shifts can mark sensitive architecture zones.\n"
        "Use it to target selective pruning and retrain policies."
    ),
    7: (
        "This chart compares per-layer reduction versus base for pruned and retrained models.\n"
        "Similar bars mean retraining respected pruning structure.\n"
        "Noticeable deviations highlight layers that changed behavior after fine-tuning.\n"
        "Those shifts often explain final accuracy gains or drops.\n"
        "It is ideal for block-wise policy tuning."
    ),
    8: (
        "This curve compares retained weight magnitude across base, pruned, and retrained models.\n"
        "If retrained moves closer to base, quality signal is being recovered.\n"
        "If it stays separated, compression effect remains strong.\n"
        "Together, this shows restoration versus efficiency in one view.\n"
        "It is a compact summary of post-retrain optimization quality."
    ),
}


def get_graph_description(index, retrain=False):
    if retrain:
        return _GRAPH_DESCRIPTIONS_RETRAIN.get(index, "")
    return _GRAPH_DESCRIPTIONS.get(index, "")


def generate_graph(base_model_path, pruned_model_path, out_dir, run_id, index, flops_pair=None):
    os.makedirs(out_dir, exist_ok=True)
    run_dir = os.path.join(out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    total = graph_count(flops_pair)
    if index < 0 or index >= total:
        raise IndexError("Graph index out of range")

    base = _load_model_safe(base_model_path)
    pruned = _load_model_safe(pruned_model_path)

    # 0) RADAR CHART - Multi-metric comparison
    if index == 0:
        path = os.path.join(run_dir, "radar_metrics.png")
        # Normalize metrics for radar (0-1 scale)
        params_b = _count_params(base)
        params_p = _count_params(pruned)
        size_b = _model_size_mb(base_model_path)
        size_p = _model_size_mb(pruned_model_path)
        time_b = _inference_time_ms(base)
        time_p = _inference_time_ms(pruned)
        density_b = _graph_density(base)
        density_p = _graph_density(pruned)
        capacity_b = _nonzero_ratio(base)
        capacity_p = _nonzero_ratio(pruned)
       
        categories = ['Parameters', 'Size', 'Inference Speed', 'Density', 'Capacity']
        values_base = np.array([params_b, size_b, 1/time_b, density_b, capacity_b])
        values_pruned = np.array([params_p, size_p, 1/time_p, density_p, capacity_p])
       
        _create_radar_chart(path, "Model Metrics Comparison", categories, values_base, values_pruned)
        return path, total

    # 1) DONUT CHART - Model Size
    if index == 1:
        path = os.path.join(run_dir, "size_donut.png")
        size_b = _model_size_mb(base_model_path)
        size_p = _model_size_mb(pruned_model_path)
        reduction = ((size_b - size_p) / size_b) * 100
        _create_donut_comparison(path, "Model Size Comparison", size_b, size_p,
                                unit="MB", center_text=f"{reduction:.1f}% Reduction")
        return path, total

    # 2) GAUGE CHARTS - Efficiency Metrics
    if index == 2:
        path = os.path.join(run_dir, "efficiency_gauges.png")
        params_b = _count_params(base)
        params_p = _count_params(pruned)
        time_b = _inference_time_ms(base)
        time_p = _inference_time_ms(pruned)
        _create_gauge_efficiency(path, "Efficiency Metrics", params_b, params_p, time_b, time_p)
        return path, total

    # 3) VIOLIN PLOT - Weight Distribution
    if index == 3:
        path = os.path.join(run_dir, "weight_violin.png")
        base_w = _get_layer_weights_flat(base)
        pruned_w = _get_layer_weights_flat(pruned)
        _create_violin_distribution(path, "Weight Distribution Comparison", base_w, pruned_w)
        return path, total

    # 4) HEATMAP - Layer-wise Weight Magnitudes
    if index == 4:
        path = os.path.join(run_dir, "weight_heatmap.png")
        _create_heatmap_sparsity(path, "Layer-wise Weight Magnitude Distribution", pruned)
        return path, total

    # 5) BUBBLE CHART - Layer Analysis
    if index == 5:
        path = os.path.join(run_dir, "layer_bubble.png")
        _create_bubble_chart(path, "Layer Sparsity vs Parameter Count", base, pruned)
        return path, total

    # 6) SANKEY DIAGRAM - Parameter Flow
    if index == 6:
        path = os.path.join(run_dir, "param_flow.png")
        params_b = _count_params(base)
        params_p = _count_params(pruned)
        _create_sankey_flow(path, "Parameter Flow: Base → Pruned", params_b, params_p)
        return path, total

    # 7) NETWORK GRAPH - Architecture Visualization
    if index == 7:
        path = os.path.join(run_dir, "network_graph.png")
        _create_network_graph(path, "Neural Network Connectivity Graph", pruned)
        return path, total

    # 8) STACKED AREA - Cumulative Distribution
    if index == 8:
        path = os.path.join(run_dir, "cumulative_params.png")
        _create_stacked_area(path, "Cumulative Parameter Distribution", base, pruned)
        return path, total

    # 9) RETENTION CURVE (original but enhanced)
    if index == 9:
        path = os.path.join(run_dir, "weight_retention.png")
        _create_retention_gradient(path, "Weight Magnitude Retention Curve", base, pruned)
        return path, total

    # 10) FLOPs Comparison (if provided)
    path = os.path.join(run_dir, "flops_comparison.png")
    _create_donut_comparison(path, "FLOPs Comparison", flops_pair[0], flops_pair[1],
                            unit="FLOPs", center_text="Computational Cost")
    return path, total


def generate_graph_retrain(base_model_path, pruned_model_path, retrained_model_path, out_dir, run_id, index):
    os.makedirs(out_dir, exist_ok=True)
    run_dir = os.path.join(out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    total = graph_count_retrain()
    if index < 0 or index >= total:
        raise IndexError("Graph index out of range")

    base = _load_model_safe(base_model_path)
    pruned = _load_model_safe(pruned_model_path)
    retrained = _load_model_safe(retrained_model_path)

    # 0) RADAR CHART - Three-way comparison
    if index == 0:
        path = os.path.join(run_dir, "radar_metrics_retrain.png")
        categories = ['Parameters', 'Size', 'Inference Speed', 'Density', 'Capacity']
       
        def get_metrics(model, path):
            return np.array([
                _count_params(model),
                _model_size_mb(path),
                1/_inference_time_ms(model),
                _graph_density(model),
                _nonzero_ratio(model)
            ])
       
        values_base = get_metrics(base, base_model_path)
        values_pruned = get_metrics(pruned, pruned_model_path)
        values_retrained = get_metrics(retrained, retrained_model_path)
       
        _create_radar_chart(path, "Three-Way Model Comparison", categories,
                           values_base, values_pruned, values_retrained)
        return path, total

    # 1) DONUT CHART - Size comparison (3 models)
    if index == 1:
        path = os.path.join(run_dir, "size_comparison_retrain.png")
        size_b = _model_size_mb(base_model_path)
        size_p = _model_size_mb(pruned_model_path)
        size_r = _model_size_mb(retrained_model_path)
       
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.patch.set_facecolor(THEME_COLORS["bg"])
       
        for ax, size, color, label in zip(axes, [size_b, size_p, size_r],
                                          [THEME_COLORS["base"], THEME_COLORS["pruned"], THEME_COLORS["retrained"]],
                                          ["Base", "Pruned", "Retrained"]):
            ax.set_facecolor(THEME_COLORS["bg"])
            max_size = max(size_b, size_p, size_r) * 1.2
            pct = size / max_size
           
            theta = np.linspace(0, 2*np.pi*pct, 100)
            x_circle = 0.8 * np.cos(theta)
            y_circle = 0.8 * np.sin(theta)
           
            ax.fill(x_circle, y_circle, color=color, alpha=0.8)
            circle_bg = plt.Circle((0, 0), 0.8, fill=False, color=THEME_COLORS["grid"], linewidth=3)
            ax.add_patch(circle_bg)
           
            ax.text(0, 0, f"{size:.2f}\nMB", ha='center', va='center', fontsize=14,
                   color=color, fontweight='bold')
            ax.text(0, -1.1, label, ha='center', va='center', fontsize=12, color=THEME_COLORS["text"])
           
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.3, 1.2)
            ax.set_aspect('equal')
            ax.axis('off')
       
        plt.suptitle("Model Size Comparison", fontsize=16, color=THEME_COLORS["text"], y=0.95)
        plt.tight_layout()
        plt.savefig(path, dpi=150, facecolor=THEME_COLORS["bg"])
        plt.close()
        return path, total

    # 2) GAUGE CHARTS - Efficiency with retrained
    if index == 2:
        path = os.path.join(run_dir, "efficiency_retrain.png")
        params_b = _count_params(base)
        params_r = _count_params(retrained)
        time_b = _inference_time_ms(base)
        time_r = _inference_time_ms(retrained)
        _create_gauge_efficiency(path, "Retrained Model Efficiency", params_b, params_r, time_b, time_r)
        return path, total

    # 3) VIOLIN PLOT - Three distributions
    if index == 3:
        path = os.path.join(run_dir, "weight_violin_retrain.png")
        base_w = _get_layer_weights_flat(base)
        pruned_w = _get_layer_weights_flat(pruned)
        retrained_w = _get_layer_weights_flat(retrained)
        _create_violin_distribution(path, "Weight Distribution: Three Models",
                                   base_w, pruned_w, retrained_w)
        return path, total

    # 4) BUBBLE CHART - Three-way comparison
    if index == 4:
        path = os.path.join(run_dir, "bubble_retrain.png")
        _create_bubble_chart(path, "Layer Analysis: Pruned vs Retrained", base, pruned, retrained)
        return path, total

    # 5) HEATMAP COMPARISON
    if index == 5:
        path = os.path.join(run_dir, "heatmap_comparison.png")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor(THEME_COLORS["bg"])
       
        for ax, model, title_suffix, color in [(ax1, pruned, "Pruned", THEME_COLORS["pruned"]),
                                                (ax2, retrained, "Retrained", THEME_COLORS["retrained"])]:
            ax.set_facecolor(THEME_COLORS["surface"])
           
            layer_names = []
            weight_matrix = []
           
            for layer in model.layers:
                w = layer.get_weights()
                if w:
                    layer_names.append(layer.name[:15])
                    magnitudes = np.abs(w[0].flatten())
                    hist, _ = np.histogram(magnitudes, bins=50, range=(0, np.percentile(magnitudes, 99) if len(magnitudes) > 0 else 1))
                    weight_matrix.append(hist)
           
            if weight_matrix:
                weight_matrix = np.array(weight_matrix)
                weight_matrix = weight_matrix / (weight_matrix.max(axis=1, keepdims=True) + 1e-8)
                im = ax.imshow(weight_matrix, aspect='auto', cmap='plasma', interpolation='nearest')
                ax.set_yticks(range(len(layer_names)))
                ax.set_yticklabels(layer_names, color=THEME_COLORS["text"], fontsize=8)
                ax.set_title(f"{title_suffix} Model", color=color, fontsize=14, fontweight='bold')
                ax.tick_params(colors=THEME_COLORS["text"])
       
        plt.suptitle("Layer-wise Weight Distribution Comparison", fontsize=16, color=THEME_COLORS["text"], y=0.98)
        plt.tight_layout()
        plt.savefig(path, dpi=150, facecolor=THEME_COLORS["bg"])
        plt.close()
        return path, total

    # 6) NETWORK GRAPH - Retrained
    if index == 6:
        path = os.path.join(run_dir, "network_retrained.png")
        _create_network_graph(path, "Retrained Model Architecture", retrained)
        return path, total

    # 7) STACKED AREA - Three models
    if index == 7:
        path = os.path.join(run_dir, "cumulative_retrain.png")
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor(THEME_COLORS["bg"])
        ax.set_facecolor(THEME_COLORS["surface"])
       
        def get_cumulative(model):
            params = []
            cumsum = 0
            for layer in model.layers:
                w = layer.get_weights()
                if w:
                    cumsum += w[0].size
                    params.append(cumsum)
            return np.array(params)
       
        base_cum = get_cumulative(base)
        pruned_cum = get_cumulative(pruned)
        retrained_cum = get_cumulative(retrained)
       
        if len(base_cum) > 0:
            x = np.linspace(0, 100, len(base_cum))
            ax.fill_between(x, 0, base_cum/base_cum[-1]*100, alpha=0.3, color=THEME_COLORS["base"], label='Base')
        if len(pruned_cum) > 0:
            x_p = np.linspace(0, 100, len(pruned_cum))
            ax.fill_between(x_p, 0, pruned_cum/pruned_cum[-1]*100, alpha=0.4, color=THEME_COLORS["pruned"], label='Pruned')
        if len(retrained_cum) > 0:
            x_r = np.linspace(0, 100, len(retrained_cum))
            ax.fill_between(x_r, 0, retrained_cum/retrained_cum[-1]*100, alpha=0.5, color=THEME_COLORS["retrained"], label='Retrained')
       
        ax.set_xlabel('Layer Progression (%)', color=THEME_COLORS["text"])
        ax.set_ylabel('Cumulative Parameters (%)', color=THEME_COLORS["text"])
        ax.set_title("Cumulative Parameter Distribution Comparison", fontsize=16, color=THEME_COLORS["text"], pad=20)
        ax.legend(facecolor=THEME_COLORS["surface"])
        ax.grid(True, alpha=0.2, color=THEME_COLORS["grid"])
        ax.tick_params(colors=THEME_COLORS["text"])
       
        sns.despine(ax=ax)
        plt.tight_layout()
        plt.savefig(path, dpi=150, facecolor=THEME_COLORS["bg"])
        plt.close()
        return path, total

    # 8) RETENTION CURVE - Three-way
    path = os.path.join(run_dir, "retention_retrain.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(THEME_COLORS["bg"])
    ax.set_facecolor(THEME_COLORS["surface"])
   
    x, b, p = _weight_magnitude_retention(base, pruned)
    _, _, r = _weight_magnitude_retention(base, retrained)
   
    ax.plot(x, b, linewidth=3, color=THEME_COLORS["base"], label='Base')
    ax.plot(x, p, linewidth=3, color=THEME_COLORS["pruned"], label='Pruned')
    ax.plot(x, r, linewidth=3, color=THEME_COLORS["retrained"], label='Retrained')
   
    ax.fill_between(x, b, alpha=0.2, color=THEME_COLORS["base"])
    ax.fill_between(x, p, alpha=0.2, color=THEME_COLORS["pruned"])
    ax.fill_between(x, r, alpha=0.2, color=THEME_COLORS["retrained"])
   
    ax.set_xlabel('Percentage of Weights Kept', color=THEME_COLORS["text"])
    ax.set_ylabel('Cumulative Weight Magnitude (%)', color=THEME_COLORS["text"])
    ax.set_title("Weight Magnitude Retention: Three Models", fontsize=16, color=THEME_COLORS["text"], pad=20)
    ax.legend(facecolor=THEME_COLORS["surface"])
    ax.grid(True, alpha=0.2, color=THEME_COLORS["grid"])
    ax.tick_params(colors=THEME_COLORS["text"])
   
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=THEME_COLORS["bg"])
    plt.close()
    return path, total
