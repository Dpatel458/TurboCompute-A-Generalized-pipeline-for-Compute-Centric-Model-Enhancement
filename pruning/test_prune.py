from pruning.cnn_l1_pruning import prune_cnn

# ===========================
# CONFIG
# ===========================
MODEL_PATH = r"D:\college\sem-8\models\CNN\garbage_cnn_model.h5"
OUTPUT_DIR = "output_cnn_test"
KEEP_RATIO = 0.7

print("\n=== STARTING CNN PRUNING TEST ===\n")

result = prune_cnn(
    model_path=MODEL_PATH,
    keep_ratio=KEEP_RATIO,
    output_dir=OUTPUT_DIR
)

print("\n=== PRUNING COMPLETE ===")
print(f"Pruned model saved at: {result['pruned_model_path']}")

print("\n=== REPORT ===")
print(f"Base GFLOPs   : {result['base_gflops']}")
print(f"Pruned GFLOPs : {result['pruned_gflops']}")
print(f"Reduction (%) : {result['reduction']}")
