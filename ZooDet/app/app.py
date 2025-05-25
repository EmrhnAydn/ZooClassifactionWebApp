import torch, json, os, io
from transformers import AutoConfig, AutoModelForImageClassification
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template, jsonify, url_for, Response, stream_with_context
from werkzeug.utils import secure_filename
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
import time
import tempfile
from threading import Thread
import copy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open("model/classes.txt") as f:
    CLASS_NAMES = [ln.strip() for ln in f]
NUM_CLASSES = len(CLASS_NAMES)


BASE_CHECKPOINT = "facebook/deit-base-patch16-224"

cfg = AutoConfig.from_pretrained(BASE_CHECKPOINT)
cfg.num_labels   = NUM_CLASSES
cfg.id2label     = {i: c for i, c in enumerate(CLASS_NAMES)}
cfg.label2id     = {c: i for i, c in enumerate(CLASS_NAMES)}

model = AutoModelForImageClassification.from_pretrained(
            BASE_CHECKPOINT,
            config=cfg,
            ignore_mismatched_sizes=True  
        ).to(DEVICE)


state = torch.load("model/deit_base_patch16_224_best.pth", map_location=DEVICE)

model.load_state_dict(state)
model.eval()

IMG_SIZE = 224
TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]),
])


test_results = {}

def allowed(fname):
    return fname.lower().rsplit('.', 1)[-1] in {"jpg","jpeg","png"}

def test_data_allowed(fname):
    return fname.lower().rsplit('.', 1)[-1] in {"zip","json","csv"}

def predict(img: Image.Image, top_k: int = 5):
    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)  
    with torch.inference_mode():
        logits = model(x).logits
        probs  = logits.softmax(dim=1)
        topk   = probs.topk(top_k)
    return [
        {"label": CLASS_NAMES[i], "prob": float(p)}
        for p,i in zip(topk.values[0], topk.indices[0])
    ]

def predict_class(img: Image.Image):
    """Predict the most likely class for a single image."""
    x = TRANSFORM(img).unsqueeze(0).to(DEVICE)  
    with torch.inference_mode():
        logits = model(x).logits
        probs = logits.softmax(dim=1)
        max_prob, max_idx = probs.max(dim=1)
    return CLASS_NAMES[max_idx], float(max_prob)

def predict_batch(image_batch, class_name, test_id):
    """Process a batch of images and update test results."""
    true_labels = []
    predicted_labels = []
    progress = 0
    
    for img_data in image_batch:
        try:
            # Process the image directly from memory
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            
            # Get prediction
            pred_class, _ = predict_class(img)
            
            # Store results
            true_labels.append(class_name)
            predicted_labels.append(pred_class)
            
            # Update progress
            progress += 1
            if test_id in test_results:
                test_results[test_id]["progress"] += 1
                
        except Exception as e:
            print(f"Error processing image: {str(e)}")
    
    # Return results from this batch
    return true_labels, predicted_labels

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # simple upload form

@app.route("/predict", methods=["POST"])
def classify():
    if 'file' not in request.files:
        return jsonify({"error":"No file part"}), 400
    file = request.files['file']
    if file.filename == '' or not allowed(file.filename):
        return jsonify({"error":"Invalid file"}), 400

    img = Image.open(file.stream).convert("RGB")
    result = predict(img, top_k=3)
    return jsonify(result)                # [{label,prob}, …]

@app.route("/start_test", methods=["POST"])
def start_test():
    """Initialize a new test and return a test ID"""
    test_id = str(int(time.time() * 1000))  # Timestamp as ID
    
    # Get classes from the request
    classes = json.loads(request.form.get('classes', '[]'))
    if not classes:
        return jsonify({"error": "No classes provided"}), 400
    
    total_files = sum(int(request.form.get(f"{cls}_count", 0)) for cls in classes)
    
    test_results[test_id] = {
        "status": "initializing",
        "classes": classes,
        "total_files": total_files,
        "progress": 0,
        "true_labels": [],
        "predicted_labels": [],
        "results": None
    }
    
    return jsonify({"test_id": test_id, "status": "initialized"})

@app.route("/upload_test_batch", methods=["POST"])
def upload_test_batch():
    """Handle one batch of test data"""
    test_id = request.form.get('test_id')
    class_name = request.form.get('class_name')
    
    if not test_id or not class_name or test_id not in test_results:
        return jsonify({"error": "Invalid test ID or class name"}), 400
    
    # Get image files and read them directly into memory
    image_batch = []
    for key in request.files:
        file = request.files[key]
        if allowed(file.filename):
            # Read the entire file into memory
            image_batch.append(file.read())
    
    if not image_batch:
        return jsonify({"error": "No valid image files in batch"}), 400
    
    # Process this batch in a separate thread to avoid timeouts
    def process_batch():
        true_labels, predicted_labels = predict_batch(image_batch, class_name, test_id)
        if test_id in test_results:
            test_results[test_id]["true_labels"].extend(true_labels)
            test_results[test_id]["predicted_labels"].extend(predicted_labels)
    
    # Start processing thread
    thread = Thread(target=process_batch)
    thread.daemon = True  # Make thread daemon so it dies when main thread dies
    thread.start()
    
    return jsonify({
        "status": "processing", 
        "received": len(image_batch),
        "progress": test_results[test_id]["progress"],
        "total": test_results[test_id]["total_files"]
    })

@app.route("/test_status/<test_id>", methods=["GET"])
def test_status(test_id):
    """Get the current status of a test"""
    if test_id not in test_results:
        return jsonify({"error": "Invalid test ID"}), 404
    
    result = test_results[test_id]
    
    # If processing is complete and results not calculated yet, calculate them
    if (result["progress"] >= result["total_files"] and 
        result["status"] != "completed" and
        len(result["true_labels"]) > 0):
        
        # Calculate metrics
        try:
            # Convert string labels to indices for sklearn
            labels = set(result["true_labels"] + result["predicted_labels"])
            label_to_idx = {label: idx for idx, label in enumerate(labels)}
            y_true = [label_to_idx[label] for label in result["true_labels"]]
            y_pred = [label_to_idx[label] for label in result["predicted_labels"]]
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # Store results
            result["results"] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "total_images": len(y_true)
            }
            result["status"] = "completed"
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
    
    response_data = {
        "status": result["status"],
        "progress": result["progress"],
        "total": result["total_files"],
        "percentage": int(100 * result["progress"] / max(1, result["total_files"]))
    }
    
    if result["status"] == "completed" and result["results"]:
        response_data["results"] = result["results"]
    
    if result["status"] == "error":
        response_data["error"] = result.get("error", "Unknown error")
    
    return jsonify(response_data)

@app.route("/upload_test_data", methods=["POST"])
def upload_test_data():
    """Legacy endpoint - redirects to new batch processing approach"""
    # Get classes from the request
    classes = json.loads(request.form.get('classes', '[]'))
    if not classes:
        return jsonify({"error": "No classes provided"}), 400
    
    # For small datasets, still use the direct approach
    true_labels = []
    predicted_labels = []
    
    for class_name in classes:
        # Find all files for this class
        class_files = [key for key in request.files.keys() if key.startswith(f"{class_name}_")]
        
        for file_key in class_files:
            file = request.files[file_key]
            if file and allowed(file.filename):
                try:
                    # Read file data to memory first
                    file_data = file.read()
                    img = Image.open(io.BytesIO(file_data)).convert("RGB")
                    
                    # Get prediction
                    pred_class, _ = predict_class(img)
                    
                    # Store true and predicted labels
                    true_labels.append(class_name)
                    predicted_labels.append(pred_class)
                except Exception as e:
                    app.logger.error(f"Error processing {file.filename}: {str(e)}")
    
    # Calculate metrics
    if len(true_labels) > 0:
        # Convert string labels to indices for sklearn
        label_to_idx = {label: idx for idx, label in enumerate(set(true_labels + predicted_labels))}
        y_true = [label_to_idx[label] for label in true_labels]
        y_pred = [label_to_idx[label] for label in predicted_labels]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Handle case where some classes might not be in the predictions
        try:
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        except:
            precision = recall = f1 = 0
        
        return jsonify({
            "success": True,
            "message": f"Processed {len(true_labels)} images from {len(classes)} classes",
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "total_images": len(true_labels)
        })
    else:
        return jsonify({"error": "No valid images were processed"}), 400

if __name__ == '__main__':
    app.run(debug=True, threaded=True)

