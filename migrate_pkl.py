"""
migrate_pkl.py — One-off script to migrate embeddings_db.pkl into Neon PostgreSQL.
Run this ONCE from your dogproject directory after setting up DATABASE_URL in .env

Usage:
    python migrate_pkl.py
"""

import pickle, os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Import db helper (must be run from the dogproject directory)
import db

PKL_PATH = os.path.join(os.path.dirname(__file__), 'embeddings_db.pkl')


def migrate():
    db.init_db()

    with open(PKL_PATH, 'rb') as f:
        embedding_db = pickle.load(f)

    print(f"Found {len(embedding_db)} dogs in pkl file")

    for dog_name, embeddings in embedding_db.items():
        # Create a stub dog profile
        profile = {
            'name':  dog_name,
            'breed': 'Unknown',
            'notes': 'Migrated from local embeddings_db.pkl',
        }
        dog_id = db.insert_dog(profile)
        for emb in embeddings:
            db.insert_embedding(dog_id, np.array(emb, dtype=np.float32))
        print(f"  ✅ {dog_name} → dog_id={dog_id} ({len(embeddings)} embeddings)")

    print(f"\n✅ Migration complete — {len(embedding_db)} dogs now in Neon.")


if __name__ == '__main__':
    migrate()
