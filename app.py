from flask import Flask, render_template, request, send_from_directory
import os
import zipfile
import tempfile
import shutil
import time

from pruning.insights import (
    generate_graph,
    graph_count,
    generate_graph_retrain,
    graph_count_retrain,
    get_graph_description,
)

# =====================================================
# IMPORT PRUNING LOGIC
# =====================================================
from pruning.fnn_prune_no_ft import prune_fnn_without_finetune
from pruning.cnn_l1_pruning import prune_cnn
from pruning.resnet_mask_pruning import prune_resnet_mask
from pruning.mobilenet_mask_pruning import prune_mobilenet_mask
from pruning.efficientnet_mask_pruning import prune_efficientnet_mask
from pruning.transformer_head_pruning import prune_transformer_heads
from pruning.retrain import retrain_model


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# ============================
# RAG SETUP
# ============================
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)
# Allow large model uploads (e.g. 500 MB)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB
RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "1.2"))



# Load embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# Load vectorstore (already persisted)
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

vectorstore = Chroma(
    persist_directory=os.path.join(BASE_DIR, "persist_directory5"),
    embedding_function=embedding_model
)

print("Collection count:", vectorstore._collection.count())

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
all_docs = vectorstore.get()["documents"]
documents = [Document(page_content=text) for text in all_docs]

bm25 = BM25Retriever.from_documents(documents)
bm25.k = 3

def hybrid_retrieve(query, k_dense=3, k_bm25=3, max_chunks=6):
    
    # Dense
    dense_docs = dense_retriever.invoke(query)
    
    # Sparse
    bm25_docs = bm25.invoke(query)
    
    # Combine
    combined = dense_docs + bm25_docs
    
    # Deduplicate by content
    seen = set()
    unique_docs = []
    
    for doc in combined:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)
    
    return unique_docs[:max_chunks]

rag_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a retrieval-based assistant for TurboCompute.

            You are given extracted context from the TurboCompute knowledge base.

            Rules:
            - Use ONLY the provided context.
            - Chunks may have more information than needed, but you should only use what is relevant to the question.
            - You may combine information across multiple context sections.
            - Greet user if he is greeting (e.g. "hello", "hi", etc.).
            - You may rephrase and summarize.
            - Do NOT introduce external knowledge.
            - If answer is not present in context, reply exactly:
            The requested information is not available in the TurboCompute knowledge base.
            """
                    ),
                    (
                        "human",
                        """Context:
            {context}

            Question:
            {question}

            Answer:
            """
        ),
    ]
)

from google import genai
client = genai.Client(api_key="KEY")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    query = data.get("question", "")

    if not query:
        return {"answer": "No question provided."}

    results = hybrid_retrieve(query)
    context = "\n\n".join([doc.page_content for doc in results])

    final = rag_prompt.format(context=context, question=query)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents = final
    ).text


    return {"answer": response}


#----------------------------------------------------------------------------------------------------------------

def model_gflops(model, input_shape):
    total_flops = 0
    shape = input_shape  # (H, W, C) or (features,)

    for layer in model.layers:
        # ---------------- Conv2D ----------------
        if isinstance(layer, layers.Conv2D):
            kh, kw, cin, cout = layer.get_weights()[0].shape
            h, w, _ = shape

            # FLOPs = H * W * Cin * Cout * Kh * Kw * 2
            total_flops += h * w * cin * cout * kh * kw * 2
            shape = (h, w, cout)

        # ---------------- Dense ----------------
        elif isinstance(layer, layers.Dense):
            in_features = shape[0]
            out_features = layer.units
            total_flops += in_features * out_features * 2
            shape = (out_features,)

        # ---------------- Flatten ----------------
        elif isinstance(layer, layers.Flatten):
            shape = (np.prod(shape),)

    return total_flops / 1e9  # GFLOPs




# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_MODELS = os.path.join(BASE_DIR, "uploads", "models")
UPLOAD_DATASETS = os.path.join(BASE_DIR, "uploads", "datasets")
OUTPUTS = os.path.join(BASE_DIR, "uploads", "outputs")

os.makedirs(UPLOAD_MODELS, exist_ok=True)
os.makedirs(UPLOAD_DATASETS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

# =====================================================
# HELPERS
# =====================================================
def extract_zip_to_dir(zip_path, base_dir):
    extract_dir = tempfile.mkdtemp(prefix="efficientnet_", dir=base_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # If zip contains a single top-level folder, use it
    entries = [e for e in os.listdir(extract_dir) if not e.startswith("__MACOSX")]
    if len(entries) == 1:
        single = os.path.join(extract_dir, entries[0])
        if os.path.isdir(single):
            return single
    return extract_dir

# =====================================================
# ROUTES
# =====================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":

        # ---------- MODEL TYPE ----------
        model_type = request.form.get("model_type")

        # ---------- KEEP RATIO ----------
        keep_ratio = request.form.get("keep_ratio", type=float, default=70.0)
        keep_ratio = keep_ratio / 100.0  # convert to 0–1

        # ---------- MODEL FILE ----------
        model_file = request.files.get("model_file")
        if not model_file:
            return "Model file missing", 400

        model_path = os.path.join(UPLOAD_MODELS, model_file.filename)
        model_file.save(model_path)

        # =================================================
        # FNN PRUNING
        # =================================================
        if model_type == "fnn":
            dataset_file = request.files.get("dataset_file")
            if not dataset_file:
                return "Dataset required for FNN pruning", 400

            dataset_filename = dataset_file.filename
            dataset_path = os.path.join(UPLOAD_DATASETS, dataset_filename)
            dataset_file.save(dataset_path)

            pruned_path, report = prune_fnn_without_finetune(
                model_path=model_path,
                dataset_path=dataset_path,
                output_dir=OUTPUTS,
                keep_ratio=keep_ratio
            )

            return render_template(
                "output.html",
                model_type="fnn",
                metrics=report,
                pruned_model=os.path.basename(pruned_path),
                base_model=os.path.basename(model_path),
                keep_ratio_percent=int(keep_ratio * 100),
                dataset_dir=dataset_filename   # ✅ ADD THIS
            )


        # =================================================
        # CNN PRUNING
        # =================================================
        if model_type == "cnn":
            report = prune_cnn(
                model_path=model_path,
                keep_ratio=keep_ratio,
                output_dir=OUTPUTS
            )

            return render_template(
                "output.html",
                model_type="cnn",
                metrics=report,
                pruned_model=os.path.basename(report["pruned_model_path"]),
                base_model=os.path.basename(model_path),
                keep_ratio_percent=int(keep_ratio * 100)
            )

        # =================================================
        # TRANSFER LEARNING (RESNET - NO DATASET)
        # =================================================
        if model_type == "transfer":
            transfer_subtype = request.form.get("transfer_subtype")

            if transfer_subtype == "resnet":
                pruned_path, report = prune_resnet_mask(
                    model_path=model_path,
                    keep_ratio=keep_ratio,
                    output_dir=OUTPUTS
                )

                return render_template(
                    "output.html",
                    model_type="resnet",
                    metrics=report,
                    pruned_model=os.path.basename(pruned_path),
                    base_model=os.path.basename(model_path),
                    keep_ratio_percent=int(keep_ratio * 100)
                )

            if transfer_subtype == "mobilenet":
                pruned_path, report = prune_mobilenet_mask(
                    model_path=model_path,
                    keep_ratio=keep_ratio,
                    output_dir=OUTPUTS
                )

                return render_template(
                    "output.html",
                    model_type="mobilenet",
                    metrics=report,
                    pruned_model=os.path.basename(pruned_path),
                    base_model=os.path.basename(model_path),
                    keep_ratio_percent=int(keep_ratio * 100)
                )

            if transfer_subtype == "efficientnet":
                dataset_zip = request.files.get("efficientnet_dataset")
                if not dataset_zip:
                    return "Dataset ZIP required for EfficientNet pruning", 400

                zip_path = os.path.join(UPLOAD_DATASETS, dataset_zip.filename)
                dataset_zip.save(zip_path)
                dataset_dir = extract_zip_to_dir(zip_path, UPLOAD_DATASETS)

                pruned_path, report = prune_efficientnet_mask(
                    model_path=model_path,
                    keep_ratio=keep_ratio,
                    output_dir=OUTPUTS,
                    dataset_dir=dataset_dir
                )

                # Cleanup uploaded zip and extracted dataset
                try:
                    os.remove(zip_path)
                except OSError:
                    pass

                return render_template(
                    "output.html",
                    model_type="efficientnet",
                    metrics=report,
                    pruned_model=os.path.basename(pruned_path),
                    base_model=os.path.basename(model_path),
                    keep_ratio_percent=int(keep_ratio * 100),
                    dataset_dir=os.path.basename(dataset_dir)
                )

            return "Selected transfer model not implemented yet", 400


        # =================================================
        # TRANSFORMER PRUNING
        # =================================================
        if model_type == "transformer":
            pruned_path, report = prune_transformer_heads(
                model_path=model_path,
                keep_ratio=keep_ratio,
                output_dir=OUTPUTS
            )

            return render_template(
                "output.html",
                model_type="transformer",
                metrics=report,
                pruned_model=os.path.basename(pruned_path),
                base_model=os.path.basename(model_path),
                keep_ratio_percent=int(keep_ratio * 100)
            )

        # =================================================
        # FUTURE ARCHITECTURES
        # =================================================
        return "Selected architecture not implemented yet", 400

    return render_template("upload.html")


@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(OUTPUTS, filename, as_attachment=True)

@app.route("/insight")
def insight():
    base_name = os.path.basename(request.args.get("base", ""))
    pruned_name = os.path.basename(request.args.get("pruned", ""))
    model_type = request.args.get("model_type", "unknown")
    keep_ratio = request.args.get("keep_ratio", "")
    if not base_name or not pruned_name:
        return "Missing model filenames", 400

    base_flops = request.args.get("base_flops", type=float)
    pruned_flops = request.args.get("pruned_flops", type=float)
    flops_pair = None
    if base_flops is not None and pruned_flops is not None:
        flops_pair = (base_flops, pruned_flops)

    total_graphs = graph_count(flops_pair)
    run_id = f"{model_type}_{int(time.time())}"

    return render_template(
        "insight.html",
        plot_urls=[],
        model_type=model_type,
        keep_ratio=keep_ratio,
        base_model=base_name,
        pruned_model=pruned_name,
        base_flops=request.args.get("base_flops", ""),
        pruned_flops=request.args.get("pruned_flops", ""),
        total_graphs=total_graphs,
        run_id=run_id
    )


@app.route("/insight_data")
def insight_data():
    base_name = os.path.basename(request.args.get("base", ""))
    pruned_name = os.path.basename(request.args.get("pruned", ""))
    model_type = request.args.get("model_type", "unknown")
    keep_ratio = request.args.get("keep_ratio", "")
    base_flops = request.args.get("base_flops", type=float)
    pruned_flops = request.args.get("pruned_flops", type=float)
    index = request.args.get("index", type=int)
    run_id = request.args.get("run_id", "")

    if not base_name or not pruned_name:
        return {"error": "Missing model filenames"}, 400
    if index is None:
        return {"error": "Missing index"}, 400
    if not run_id:
        return {"error": "Missing run_id"}, 400

    base_path = os.path.join(UPLOAD_MODELS, base_name)
    if not os.path.exists(base_path):
        alt = os.path.join(OUTPUTS, base_name)
        if os.path.exists(alt):
            base_path = alt
        else:
            return {"error": "Base model not found"}, 404

    pruned_path = os.path.join(OUTPUTS, pruned_name)
    if not os.path.exists(pruned_path):
        alt = os.path.join(UPLOAD_MODELS, pruned_name)
        if os.path.exists(alt):
            pruned_path = alt
        else:
            return {"error": "Pruned model not found"}, 404

    out_dir = os.path.join(BASE_DIR, "static", "insights")
    flops_pair = None
    if base_flops is not None and pruned_flops is not None:
        flops_pair = (base_flops, pruned_flops)

    try:
        plot_path, total = generate_graph(
            base_path,
            pruned_path,
            out_dir,
            run_id,
            index,
            flops_pair=flops_pair
        )
    except IndexError:
        return {"error": "Index out of range"}, 400

    plot_url = "/static/insights/" + run_id + "/" + os.path.basename(plot_path)

    return {
        "plot_url": plot_url,
        "index": index,
        "total": total,
        "model_type": model_type,
        "keep_ratio": keep_ratio,
        "description": get_graph_description(index, retrain=False),
    }


@app.route("/insight_retrain")
def insight_retrain():
    base_name = os.path.basename(request.args.get("base", ""))
    pruned_name = os.path.basename(request.args.get("pruned", ""))
    retrained_name = os.path.basename(request.args.get("retrained", ""))
    model_type = request.args.get("model_type", "unknown")

    if not base_name or not pruned_name or not retrained_name:
        return "Missing model filenames", 400

    run_id = f"{model_type}_retrain_{int(time.time())}"

    return render_template(
        "insight.html",
        plot_urls=[],
        model_type=model_type,
        keep_ratio="retrain",
        base_model=base_name,
        pruned_model=pruned_name,
        retrained_model=retrained_name,
        base_flops="",
        pruned_flops="",
        total_graphs=graph_count_retrain(),
        run_id=run_id,
        insight_data_endpoint="/insight_retrain_data"
    )


@app.route("/insight_retrain_data")
def insight_retrain_data():
    base_name = os.path.basename(request.args.get("base", ""))
    pruned_name = os.path.basename(request.args.get("pruned", ""))
    retrained_name = os.path.basename(request.args.get("retrained", ""))
    model_type = request.args.get("model_type", "unknown")
    keep_ratio = request.args.get("keep_ratio", "retrain")
    index = request.args.get("index", type=int)
    run_id = request.args.get("run_id", "")

    if not base_name or not pruned_name or not retrained_name:
        return {"error": "Missing model filenames"}, 400
    if index is None:
        return {"error": "Missing index"}, 400
    if not run_id:
        return {"error": "Missing run_id"}, 400

    base_path = os.path.join(UPLOAD_MODELS, base_name)
    if not os.path.exists(base_path):
        alt = os.path.join(OUTPUTS, base_name)
        if os.path.exists(alt):
            base_path = alt
        else:
            return {"error": "Base model not found"}, 404

    pruned_path = os.path.join(OUTPUTS, pruned_name)
    if not os.path.exists(pruned_path):
        alt = os.path.join(UPLOAD_MODELS, pruned_name)
        if os.path.exists(alt):
            pruned_path = alt
        else:
            return {"error": "Pruned model not found"}, 404

    retrained_path = os.path.join(OUTPUTS, retrained_name)
    if not os.path.exists(retrained_path):
        alt = os.path.join(UPLOAD_MODELS, retrained_name)
        if os.path.exists(alt):
            retrained_path = alt
        else:
            return {"error": "Retrained model not found"}, 404

    out_dir = os.path.join(BASE_DIR, "static", "insights")

    try:
        plot_path, total = generate_graph_retrain(
            base_path,
            pruned_path,
            retrained_path,
            out_dir,
            run_id,
            index
        )
    except IndexError:
        return {"error": "Index out of range"}, 400

    plot_url = "/static/insights/" + run_id + "/" + os.path.basename(plot_path)

    return {
        "plot_url": plot_url,
        "index": index,
        "total": total,
        "model_type": model_type,
        "keep_ratio": keep_ratio,
        "description": get_graph_description(index, retrain=True),
    }

@app.route("/retrain", methods=["GET", "POST"])
def retrain():
    model_name = request.args.get("model")
    model_type = request.args.get("model_type")
    base_model_name = request.args.get("base_model")
    dataset_dir = request.args.get("dataset_dir") or None  # may be None

    if not model_name or not model_type:
        return "Missing retrain context", 400

    model_path = os.path.join(OUTPUTS, model_name)
    if not os.path.exists(model_path):
        return "Model not found", 404

    if request.method == "POST":
        epochs = int(request.form.get("epochs"))
        lr = float(request.form.get("learning_rate"))

        dataset_path = None

        # 1️⃣ If dataset already exists, reuse it
        if dataset_dir:
            dataset_path = os.path.join(UPLOAD_DATASETS, dataset_dir)
            if not os.path.exists(dataset_path):
                return "Existing dataset not found. Please re-upload.", 400

        # 2️⃣ Else expect new upload
        else:
            dataset_file = request.files.get("dataset_file")
            if not dataset_file:
                return "Dataset required for retraining", 400

            dataset_path = os.path.join(UPLOAD_DATASETS, dataset_file.filename)
            dataset_file.save(dataset_path)

            # If a zip is uploaded (common for image datasets), extract it
            if dataset_path.lower().endswith(".zip"):
                zip_path = dataset_path
                dataset_path = extract_zip_to_dir(zip_path, UPLOAD_DATASETS)
                try:
                    os.remove(zip_path)
                except OSError:
                    pass

        # 🔥 Call retraining logic
        try:
            retrained_path, history = retrain_model(
                model_path=model_path,
                dataset_path=dataset_path,
                model_type=model_type,
                epochs=epochs,
                learning_rate=lr,
                output_dir=OUTPUTS
            )
        except ValueError as exc:
            return str(exc), 400

        return render_template(
            "retrain_result.html",
            retrained_model=os.path.basename(retrained_path),
            history=history,
            model_type=model_type,
            base_model=base_model_name,
            pruned_model=model_name
        )

    return render_template(
        "retrain.html",
        model_name=model_name,
        model_type=model_type,
        base_model=base_model_name,
        dataset_dir=dataset_dir
    )


@app.route("/architecture")
def architecture():
    return render_template("architecture.html")

@app.route("/test", methods=["GET", "POST"])
def test():
    results = None

    if request.method == "POST":
        
        model_file = request.files.get("model_file")
        sample_size = int(request.form.get("sample_size"))
        epochs = int(request.form.get("epochs"))
        batch_size = int(request.form.get("batch_size"))
        optimizer = request.form.get("optimizer")

        if not model_file:
            return "Model file missing", 400

        model_path = os.path.join(UPLOAD_MODELS, model_file.filename)
        model_file.save(model_path)

        model = tf.keras.models.load_model(model_path, compile=False)

        # ==========================
        # TRUE FORWARD GFLOPs
        # ==========================
        input_shape = model.input_shape[1:]
        forward_gflops = model_gflops(model, input_shape)

        # ==========================
        # TRAINING GFLOPs ESTIMATION
        # ==========================
        # Rule of thumb:
        # backward + optimizer ≈ 2.5 × forward
        training_gflops = forward_gflops * sample_size * epochs * 2.5

        # ==========================
        # MEMORY ESTIMATION
        # ==========================
        param_mb = model.count_params() * 4 / (1024 ** 2)
        activation_mb = param_mb * batch_size * 3

        optimizer_multiplier = {
            "sgd": 1,
            "momentum": 2,
            "rmsprop": 2,
            "adam": 2,
            "adamw": 2,
            "adagrad": 2,
            "adadelta": 2,
        }.get(optimizer, 2)

        optimizer_mb = param_mb * optimizer_multiplier
        total_memory = param_mb + activation_mb + optimizer_mb + 600

        results = {
            "forward_gflops": round(forward_gflops, 4),
            "training_gflops": round(training_gflops, 2),
            "params_mb": round(param_mb, 2),
            "activation_mb": round(activation_mb, 2),
            "optimizer_mb": round(optimizer_mb, 2),
            "total_memory": round(total_memory, 2),
            "optimizer": optimizer.upper(),
        }

    return render_template("test.html", results=results)


@app.route("/credit")
def credit():
    return render_template("credit.html")

@app.route("/grokking")
def grokking():
    return render_template("grokking.html")

@app.route("/help")
def help():
    return render_template("help.html")

@app.route("/rpaper")
def rpaper():
    return render_template("rpaper.html")


# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    app.run(debug=True)


