from flask import Flask, request, render_template
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from sentence_transformers.util import cos_sim
from PIL import Image
import torch
import os
import json

app = Flask(__name__)
os.makedirs("static", exist_ok=True)

# Load CLIP model and tokenizer
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

# ✅ Load label.json
with open("label.json", "r") as f:
    label_data = json.load(f)

# ✅ Truncate long descriptions to 77 tokens using tokenizer
MAX_TOKENS = 77
processed_descriptions = []

for entry in label_data:
    desc = entry["description"]
    tokens = clip_tokenizer(desc, return_tensors="pt", truncation=False)["input_ids"][0]
    if len(tokens) > MAX_TOKENS:
        # Truncate from the end only
        tokens = tokens[:MAX_TOKENS]
        desc = clip_tokenizer.decode(tokens, skip_special_tokens=True).strip()
    processed_descriptions.append(desc)

# ✅ Generate label embeddings
print("🔄 Generating label embeddings...")
text_inputs = clip_processor(text=processed_descriptions, return_tensors="pt", padding=True)
with torch.no_grad():
    label_embeddings = clip_model.get_text_features(**text_inputs)
print("✅ Label embeddings ready.")

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
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_embedding = clip_model.get_image_features(**inputs)

        # Compare image embedding with label embeddings
        similarity = cos_sim(image_embedding, label_embeddings)[0]
        top_score, top_idx = similarity.max(0)
        top3 = torch.topk(similarity, 3)

        # Get best matching labels
        result["top_label"] = label_data[top_idx.item()]["label"]
        result["top_score"] = float(top_score)
        result["top3"] = [(label_data[i.item()]["label"], float(s)) for i, s in zip(top3.indices, top3.values)]

    return render_template("index.html", result=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"✅ Flask app running at http://127.0.0.1:{port}")
    app.run(debug=True, port=port)
