from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64, io, pickle, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

app = Flask(__name__)
CORS(app)

# Model Definition

class EmbeddingNet(nn.Module):
    def __init__(self, backbone='resnet50', embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone_name = backbone
        if backbone == 'resnet50':
            base = models.resnet50(weights=None)
            in_features = base.fc.in_features          # 2048
            self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x):
        feat = self.backbone(x)
        emb  = self.head(feat)
        return F.normalize(emb, p=2, dim=1)


class TripletNet(nn.Module):
    def __init__(self, backbone='resnet50', embedding_dim=128):
        super().__init__()
        self.emb_net = EmbeddingNet(backbone=backbone, embedding_dim=embedding_dim)

    def get_embedding(self, x):
        return self.emb_net(x)


# Loading Model & Database

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pth')
DB_PATH    = os.path.join(os.path.dirname(__file__), 'embeddings_db.pkl')
THRESHOLD  = 0.9

model = TripletNet(backbone='resnet50', embedding_dim=128).to(DEVICE)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"✅ Model loaded — epoch {ckpt['epoch']}, val_loss={ckpt.get('val_loss', '?')}")

with open(DB_PATH, 'rb') as f:
    embedding_db = pickle.load(f)
print(f"✅ Embedding DB loaded — {len(embedding_db)} dogs registered")

if 'threshold' in ckpt:
    THRESHOLD = ckpt['threshold']
    print(f"✅ Threshold from checkpoint: {THRESHOLD}")


# Image Transform 

infer_transform = T.Compose([
    T.Resize((160, 160)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


#  Helper Functions 

@torch.no_grad()
def extract_embedding(pil_img):
    t   = infer_transform(pil_img.convert('RGB')).unsqueeze(0).to(DEVICE)
    emb = model.get_embedding(t).squeeze(0).cpu().numpy()
    return emb.astype(np.float32)


def get_mean_embedding(dog_name):
    embs = np.stack(embedding_db[dog_name], axis=0)
    mean = embs.mean(axis=0)
    return (mean / (np.linalg.norm(mean) + 1e-8)).astype(np.float32)


def decode_image(b64_string):
    # Strip data URI prefix if present (e.g. "data:image/jpeg;base64,...")
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_bytes))


# Routes 

@app.route('/')
def index():
    return send_from_directory('templates','index.html')


@app.route('/health')
def health():
    return jsonify({
        'status':      'ok',
        'dogs_in_db':  len(embedding_db),
        'device':      str(DEVICE),
        'threshold':   THRESHOLD,
    })


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image_b64' not in data:
        return jsonify({'error': 'Missing image_b64 field'}), 400

    try:
        pil_img = decode_image(data['image_b64'])
    except Exception as e:
        return jsonify({'error': f'Invalid image: {str(e)}'}), 400

    query_emb = extract_embedding(pil_img)

    # Compare against every registered dog
    results = []
    for dog_name in embedding_db:
        avg_emb = get_mean_embedding(dog_name)
        dist    = float(np.linalg.norm(query_emb - avg_emb))
        results.append({'name': dog_name, 'distance': round(dist, 4)})

    results.sort(key=lambda x: x['distance'])

    if not results:
        return jsonify({
            'is_known':     False,
            'matched_name': None,
            'confidence':   0.0,
            'distance':     None,
            'threshold':    THRESHOLD,
            'top_k':        [],
        })

    best      = results[0]
    is_known  = best['distance'] < THRESHOLD
    confidence = round(max(0.0, 1.0 - best['distance'] / THRESHOLD), 4)

    return jsonify({
        'is_known':     bool(is_known),
        'matched_name': best['name'] if is_known else None,
        'confidence':   confidence,
        'distance':     best['distance'],
        'threshold':    THRESHOLD,
        'top_k':        results[:3],
    })


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or 'image_b64' not in data or 'dog_name' not in data:
        return jsonify({'error': 'Missing image_b64 or dog_name'}), 400

    try:
        pil_img = decode_image(data['image_b64'])
    except Exception as e:
        return jsonify({'error': f'Invalid image: {str(e)}'}), 400

    name = data['dog_name'].strip()
    emb  = extract_embedding(pil_img)

    if name not in embedding_db:
        embedding_db[name] = []
    embedding_db[name].append(emb)

    # Persist updated DB to disk immediately
    with open(DB_PATH, 'wb') as f:
        pickle.dump(embedding_db, f, protocol=pickle.HIGHEST_PROTOCOL)

    return jsonify({
        'success':          True,
        'dog_name':         name,
        'total_embeddings': len(embedding_db[name]),
        'total_dogs':       len(embedding_db),
    })


@app.route('/dogs')
def list_dogs():
    return jsonify({
        'dogs':  list(embedding_db.keys()),
        'count': len(embedding_db),
    })


@app.route('/threshold', methods=['POST'])
def update_threshold():
    data = request.get_json()
    global THRESHOLD
    if 'threshold' not in data:
        return jsonify({'error': 'Missing threshold'}), 400
    THRESHOLD = float(data['threshold'])
    return jsonify({'threshold': THRESHOLD})


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True, port=5000)