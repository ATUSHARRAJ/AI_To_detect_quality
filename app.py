# from flask import Flask, request, render_template, redirect, url_for
# from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
# from sentence_transformers.util import cos_sim
# from PIL import Image
# import torch
# import os
# import json
# import shutil
# import numpy as np
# from datetime import datetime

# app = Flask(__name__)
# os.makedirs("static", exist_ok=True)
# os.makedirs("feedback", exist_ok=True)

# # Load CLIP model and tokenizer
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# # Load label.json
# with open("label.json", "r") as f:
#     label_data = json.load(f)

# # Truncate long descriptions to 77 tokens
# MAX_TOKENS = 77
# processed_descriptions = []

# for entry in label_data:
#     desc = entry["description"]
#     tokens = clip_tokenizer(desc, return_tensors="pt", truncation=False)["input_ids"][0]
#     if len(tokens) > MAX_TOKENS:
#         tokens = tokens[:MAX_TOKENS]
#         desc = clip_tokenizer.decode(tokens, skip_special_tokens=True).strip()
#     processed_descriptions.append(desc)

# # Generate label embeddings
# print("🔄 Generating label embeddings...")
# text_inputs = clip_processor(text=processed_descriptions, return_tensors="pt", padding=True)
# with torch.no_grad():
#     label_embeddings = clip_model.get_text_features(**text_inputs)
# print("✅ Label embeddings ready.")

# @app.route("/", methods=["GET", "POST"])
# def index():
#     result = {}

#     if request.method == "POST":
#         if "image" not in request.files:
#             result["error"] = "No image uploaded."
#             return render_template("index.html", result=result)

#         file = request.files["image"]
#         image = Image.open(file).convert("RGB")
#         image_path = os.path.join("static", "uploaded_image.jpg")
#         image.save(image_path)

#         # Generate image embedding
#         inputs = clip_processor(images=image, return_tensors="pt")
#         with torch.no_grad():
#             image_embedding = clip_model.get_image_features(**inputs)

#         # Compare to label embeddings (text)
#         similarity_text = cos_sim(image_embedding, label_embeddings)[0]
#         top_text_score, top_text_idx = similarity_text.max(0)
#         top3_text = torch.topk(similarity_text, 3)

#         # Compare to feedback image embeddings
#         feedback_similarities = []
#         for file in os.listdir("feedback"):
#             if file.endswith(".npy"):
#                 emb_path = os.path.join("feedback", file)
#                 with open(emb_path, "rb") as f:
#                     saved_emb = torch.tensor(np.load(f)).cpu()
#                 sim = cos_sim(image_embedding, saved_emb)[0][0].item()
#                 json_path = emb_path.replace(".npy", ".json")
#                 if os.path.exists(json_path):
#                     with open(json_path) as jf:
#                         meta = json.load(jf)
#                         feedback_similarities.append((sim, meta))

#         top_feedback = sorted(feedback_similarities, key=lambda x: x[0], reverse=True)[:1]

#         # Compare best of both
#         if top_feedback and top_feedback[0][0] > top_text_score.item():
#             result["top_label"] = f"User Feedback: {top_feedback[0][1]['user_feedback']}"
#             result["top_score"] = float(top_feedback[0][0])
#         else:
#             result["top_label"] = label_data[top_text_idx.item()]["label"]
#             result["top_score"] = float(top_text_score)

#         result["top3"] = [(label_data[i.item()]["label"], float(s)) for i, s in zip(top3_text.indices, top3_text.values)]

#     return render_template("index.html", result=result)

# @app.route("/feedback", methods=["POST"])
# def feedback():
#     user_feedback = request.form["user_feedback"]
#     prediction = request.form["prediction"]

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     feedback_id = f"feedback_{timestamp}"
#     saved_image_path = os.path.join("feedback", f"{feedback_id}.jpg")
#     saved_json_path = os.path.join("feedback", f"{feedback_id}.json")
#     saved_embedding_path = os.path.join("feedback", f"{feedback_id}.npy")

#     shutil.copy("static/uploaded_image.jpg", saved_image_path)

#     image = Image.open(saved_image_path).convert("RGB")
#     inputs = clip_processor(images=image, return_tensors="pt")
#     with torch.no_grad():
#         image_embedding = clip_model.get_image_features(**inputs)
#     image_embedding = image_embedding.detach().cpu().numpy()
#     with open(saved_embedding_path, "wb") as f:
#         np.save(f, image_embedding)

#     feedback_data = {
#         "timestamp": timestamp,
#         "prediction": prediction,
#         "user_feedback": user_feedback,
#         "image": saved_image_path,
#         "embedding": saved_embedding_path
#     }
#     with open(saved_json_path, "w") as f:
#         json.dump(feedback_data, f, indent=2)

#     return redirect(url_for("index"))

# @app.route("/search", methods=["POST"])
# def search_from_image():
#     if "image" not in request.files:
#         return "No image uploaded.", 400

#     file = request.files["image"]
#     image = Image.open(file).convert("RGB")
#     inputs = clip_processor(images=image, return_tensors="pt")
#     with torch.no_grad():
#         query_embedding = clip_model.get_image_features(**inputs)
#     query_embedding = query_embedding.cpu()

#     similarities = []
#     for file in os.listdir("feedback"):
#         if file.endswith(".npy"):
#             emb_path = os.path.join("feedback", file)
#             with open(emb_path, "rb") as f:
#                 saved_emb = torch.tensor(np.load(f)).cpu()
#             sim = cos_sim(query_embedding, saved_emb)[0][0].item()

#             json_path = emb_path.replace(".npy", ".json")
#             if os.path.exists(json_path):
#                 with open(json_path) as jf:
#                     meta = json.load(jf)
#                     similarities.append((sim, meta))

#     top_matches = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]
#     return render_template("search_results.html", results=top_matches)

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 5000))
#     app.run(debug=True, port=port)

from flask import Flask, request, render_template, redirect, url_for
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers.util import cos_sim
from PIL import Image
import torch
import os
import json
import shutil
import numpy as np
from datetime import datetime

app = Flask(__name__)
os.makedirs("static", exist_ok=True)
os.makedirs("feedback", exist_ok=True)

# Load model on CPU and in eval mode to reduce memory
device = torch.device("cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_model.eval()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load label.json
with open("label.json", "r") as f:
    label_data = json.load(f)

# Load precomputed label embeddings (assumed saved as .npy file)
label_embeddings_path = "label_embeddings.npy"
if os.path.exists(label_embeddings_path):
    label_embeddings = torch.tensor(np.load(label_embeddings_path)).to(device)
else:
    # Generate and save label embeddings if not found
    print("Generating label embeddings from scratch...")
    descriptions = [entry["description"] for entry in label_data]
    text_inputs = clip_processor(text=descriptions, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        label_embeddings = clip_model.get_text_features(**text_inputs)
    np.save(label_embeddings_path, label_embeddings.cpu().numpy())

@app.route("/", methods=["GET", "POST"])
def index():
    result = {}

    if request.method == "POST":
        if "image" not in request.files:
            result["error"] = "No image uploaded."
            return render_template("index.html", result=result)

        file = request.files["image"]
        image = Image.open(file).convert("RGB")
        image_path = os.path.join("static", "uploaded_image.jpg")
        image.save(image_path)

        # Generate image embedding
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            image_embedding = clip_model.get_image_features(**inputs)

        # Compare to label embeddings (text)
        similarity_text = cos_sim(image_embedding, label_embeddings)[0]
        top_text_score, top_text_idx = similarity_text.max(0)
        top3_text = torch.topk(similarity_text, 3)

        # Compare to feedback image embeddings
        feedback_similarities = []
        for file in os.listdir("feedback"):
            if file.endswith(".npy"):
                emb_path = os.path.join("feedback", file)
                with open(emb_path, "rb") as f:
                    saved_emb = torch.tensor(np.load(f)).to(device)
                sim = cos_sim(image_embedding, saved_emb)[0][0].item()
                json_path = emb_path.replace(".npy", ".json")
                if os.path.exists(json_path):
                    with open(json_path) as jf:
                        meta = json.load(jf)
                        feedback_similarities.append((sim, meta))

        top_feedback = sorted(feedback_similarities, key=lambda x: x[0], reverse=True)[:1]

        # Compare best of both
        if top_feedback and top_feedback[0][0] > top_text_score.item():
            result["top_label"] = f"User Feedback: {top_feedback[0][1]['user_feedback']}"
            result["top_score"] = float(top_feedback[0][0])
        else:
            result["top_label"] = label_data[top_text_idx.item()]["label"]
            result["top_score"] = float(top_text_score)

        result["top3"] = [(label_data[i.item()]["label"], float(s)) for i, s in zip(top3_text.indices, top3_text.values)]

    return render_template("index.html", result=result)

@app.route("/feedback", methods=["POST"])
def feedback():
    user_feedback = request.form["user_feedback"]
    prediction = request.form["prediction"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feedback_id = f"feedback_{timestamp}"
    saved_image_path = os.path.join("feedback", f"{feedback_id}.jpg")
    saved_json_path = os.path.join("feedback", f"{feedback_id}.json")
    saved_embedding_path = os.path.join("feedback", f"{feedback_id}.npy")

    shutil.copy("static/uploaded_image.jpg", saved_image_path)

    image = Image.open(saved_image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        image_embedding = clip_model.get_image_features(**inputs)
    image_embedding = image_embedding.detach().cpu().numpy()
    with open(saved_embedding_path, "wb") as f:
        np.save(f, image_embedding)

    feedback_data = {
        "timestamp": timestamp,
        "prediction": prediction,
        "user_feedback": user_feedback,
        "image": saved_image_path,
        "embedding": saved_embedding_path
    }
    with open(saved_json_path, "w") as f:
        json.dump(feedback_data, f, indent=2)

    return redirect(url_for("index"))

@app.route("/search", methods=["POST"])
def search_from_image():
    if "image" not in request.files:
        return "No image uploaded.", 400

    file = request.files["image"]
    image = Image.open(file).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        query_embedding = clip_model.get_image_features(**inputs)
    query_embedding = query_embedding.cpu()

    similarities = []
    for file in os.listdir("feedback"):
        if file.endswith(".npy"):
            emb_path = os.path.join("feedback", file)
            with open(emb_path, "rb") as f:
                saved_emb = torch.tensor(np.load(f)).cpu()
            sim = cos_sim(query_embedding, saved_emb)[0][0].item()

            json_path = emb_path.replace(".npy", ".json")
            if os.path.exists(json_path):
                with open(json_path) as jf:
                    meta = json.load(jf)
                    similarities.append((sim, meta))

    top_matches = sorted(similarities, key=lambda x: x[0], reverse=True)[:3]
    return render_template("search_results.html", results=top_matches)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)