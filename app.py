from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64, io, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image

import db  # Neon PostgreSQL helper

app = Flask(__name__)
CORS(app)


# ── Model Definition ──────────────────────────────────────────────────────────

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


# ── Load Model ────────────────────────────────────────────────────────────────

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.pth')
THRESHOLD  = 1.3

model = TripletNet(backbone='resnet50', embedding_dim=128).to(DEVICE)
ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
print(f"✅ Model loaded — epoch {ckpt['epoch']}, val_loss={ckpt.get('val_loss', '?')}")

if 'threshold' in ckpt:
    THRESHOLD = ckpt['threshold']
    print(f"✅ Threshold from checkpoint: {THRESHOLD}")

# ── Init Neon DB ──────────────────────────────────────────────────────────────

db.init_db()


# ── Image Transform ───────────────────────────────────────────────────────────

infer_transform = T.Compose([
    T.Resize((160, 160)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


# ── Helper Functions ──────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embedding(pil_img):
    t   = infer_transform(pil_img.convert('RGB')).unsqueeze(0).to(DEVICE)
    emb = model.get_embedding(t).squeeze(0).cpu().numpy()
    return emb.astype(np.float32)


def decode_image(b64_string):
    """Accept raw base64 or data-URI base64."""
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(img_bytes))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')


@app.route('/health')
def health():
    return jsonify({
        'status':     'ok',
        'dogs_in_db': db.count_dogs(),
        'device':     str(DEVICE),
        'threshold':  THRESHOLD,
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

    # Query Neon for nearest neighbours
    results = db.search_nearest(query_emb, top_k=5)

    if not results:
        return jsonify({
            'is_known':     False,
            'matched_name': None,
            'matched_id':   None,
            'confidence':   0.0,
            'distance':     None,
            'threshold':    THRESHOLD,
            'top_k':        [],
        })

    best      = results[0]
    is_known  = best['distance'] < THRESHOLD
    confidence = round(max(0.0, 1.0 - best['distance'] / THRESHOLD), 4)

    # If matched, fetch full dog profile from DB
    dog_profile = None
    if is_known:
        dog_profile = db.get_dog(best['dog_id'])

    return jsonify({
        'is_known':     bool(is_known),
        'matched_name': best['name'] if is_known else None,
        'matched_id':   best['dog_id'] if is_known else None,
        'confidence':   confidence,
        'distance':     round(best['distance'], 4),
        'threshold':    THRESHOLD,
        'top_k':        [
            {'name': r['name'], 'distance': round(r['distance'], 4)}
            for r in results[:3]
        ],
        'dog_profile':  dog_profile,
    })


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if not data or 'image_b64' not in data or 'name' not in data:
        return jsonify({'error': 'Missing image_b64 or name'}), 400

    try:
        pil_img = decode_image(data['image_b64'])
    except Exception as e:
        return jsonify({'error': f'Invalid image: {str(e)}'}), 400

    # Build profile dict from all supplied fields
    profile = {
        'name':             data.get('name', '').strip(),
        'breed':            data.get('breed', 'Unknown'),
        'dob':              data.get('dob') or None,
        'sex':              data.get('sex', 'Unknown'),
        'weight':           data.get('weight') or None,
        'color':            data.get('color') or None,
        'owner':            data.get('owner') or None,
        'area':             data.get('area') or None,
        'vaccinated':       bool(data.get('vaccinated', False)),
        'rabies_vacc_date': data.get('rabies_vacc_date') or None,
        'rabies_vacc_due':  data.get('rabies_vacc_due') or None,
        'dhpp_date':        data.get('dhpp_date') or None,
        'dhpp_due':         data.get('dhpp_due') or None,
        'last_checkup':     data.get('last_checkup') or None,
        'next_checkup':     data.get('next_checkup') or None,
        'notes':            data.get('notes') or None,
        'photo_url':        data.get('photo_url') or None,
    }

    if not profile['name']:
        return jsonify({'error': 'name cannot be empty'}), 400

    emb    = extract_embedding(pil_img)
    dog_id = db.insert_dog(profile)
    db.insert_embedding(dog_id, emb)

    return jsonify({
        'success': True,
        'dog_id':  dog_id,
        'name':    profile['name'],
    })


@app.route('/dogs')
def list_dogs():
    dogs = db.list_dogs()
    return jsonify({'dogs': dogs, 'count': len(dogs)})


@app.route('/dog/<int:dog_id>')
def get_dog(dog_id):
    dog = db.get_dog(dog_id)
    if dog is None:
        return jsonify({'error': 'Dog not found'}), 404
    return jsonify(dog)


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