# RabiesAlert

A lightweight Flask app that recognizes known dogs from photos and returns vaccination and profile information. It combines a PyTorch embedding model (triplet network) with a small PostgreSQL-backed catalogue and a nearest-neighbour search over pre-computed embeddings.

## What this is
RabiesAlert identifies known dogs from an uploaded image, shows their stored metadata (owner, vaccination dates, notes) and offers a simple admin workflow to add new dogs and photos. It's intended for small deployments (local or simple cloud host) where a pre-trained embedding model is available.

### Stack
- Language(s): Python (Flask) + HTML for the frontend
- Framework / runtime: Flask (WSGI) + PyTorch for the embedding model
- Notable libraries: Flask, torch, torchvision, PIL (Pillow), psycopg2 (Postgres driver), numpy, scikit-learn (used for nearest-neighbour/transform utilities)

## How it's organized
```
app.py                # Flask application (routes, model loading, inference handlers)
db.py                 # Database helpers (psycopg2 wrapper, migration-ish helpers)
requirements.txt      # Python deps
templates/            # UI (templates/index.html is the single-page frontend)
best_model.pth        # (binary) pre-trained PyTorch model used for embeddings
embeddings_db.pkl     # precomputed embeddings / small lookup (pickle)
migrate_pkl.py        # helper script to rebuild embeddings_db.pkl from saved images
```

How it fits together: app.py is the entry point. On startup it loads the PyTorch model (best_model.pth) and an embeddings index if present (embeddings_db.pkl). Incoming images are decoded, converted to embeddings with the model, then queried against the stored embeddings (a lightweight nearest-neighbour lookup in Python). db.py wraps the Postgres operations (inserting/reading dog profiles) and exposes helper functions app.py uses for listing/searching profiles.

## How to run it
Note: the repository currently includes a large binary model file (best_model.pth). If you plan to run the app, ensure you have enough disk space and a Python environment.

1. Clone repository

```bash
git clone https://github.com/Asta-V22/RabiesAlert.git
cd RabiesAlert
```

2. Create a virtual environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Provide configuration

- The app expects a Postgres connection to store dog profiles. Create a `.env` file in repo root with a suitable DATABASE_URL (or the environment variable name your deployment expects). Example (replace with your DB connection string):

```
DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

- If you do not have a Postgres instance readily available, many features (listing/searching/storing profiles) will not work; you can still run the app to exercise image-to-embedding and local search if you seed `embeddings_db.pkl`.

4. (Optional) If you have images and want to build embeddings locally, use migrate_pkl.py

```bash
python migrate_pkl.py
# This script helps convert image folders into the embeddings_db.pkl used by the app
```

5. Run the app

```bash
python app.py
# or if using flask CLI: FLASK_APP=app flask run --port 5000
```

6. Open the UI

- Visit http://127.0.0.1:5000/ in a browser. The single-page UI (templates/index.html) lets you upload an image, inspect search results and add new dog profiles.

## Important notes / caveats
- The repo currently ships a large model file (best_model.pth). That file makes the repository heavy — consider storing the model in object storage and loading it at runtime for production deployments.
- The included .env file (if present) may contain sensitive connection strings. Do not publish real secrets.
- The nearest-neighbour search is in-Python and designed for small datasets; for larger catalogs use a proper vector DB (Milvus, Pinecone, FAISS index persisted to disk, or Postgres + pgvector).

## Files of interest
- app.py — main Flask app and inference logic
- db.py — database connection and convenience functions (psycopg2)
- templates/index.html — frontend UI packed into one file
- best_model.pth — PyTorch model (binary) used to compute embeddings
- embeddings_db.pkl — a pickle with saved embeddings & metadata
- migrate_pkl.py — script to build embeddings_db.pkl from images
- requirements.txt — pinned dependency list

## Development / Contributing
- Run the app locally using the steps above.
- If you change the model or embeddings, re-run migrate_pkl.py and restart the server.
- Consider adding unit tests around db.py and the model transformation utilities.

## License
No license specified — add a LICENSE file if you intend to make this open-source.

---

If you want, I can (pick one):
- open a PR that adds this README.md to the repository (I already added it),
- trim the README to match your preferred content/voice, or
- add a short CONTRIBUTING.md and a .gitignore entry to avoid committing large model files in future.
