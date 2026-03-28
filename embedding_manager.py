"""
Embedding Manager: save and load FAISS vector stores locally.
Allows users to reuse embeddings for the same textbook instead of
re-indexing every time.

Storage layout:
  embeddings_store/
    index.json          ← registry: {alias: {path, created_at, pdf_name}}
    <alias>/
      index.faiss
      index.pkl
"""

import os
import json
import shutil
from datetime import datetime

STORE_DIR = os.path.join(os.path.dirname(__file__), "embeddings_store")
INDEX_FILE = os.path.join(STORE_DIR, "index.json")


def _load_index() -> dict:
    os.makedirs(STORE_DIR, exist_ok=True)
    if not os.path.exists(INDEX_FILE):
        return {}
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_index(index: dict):
    os.makedirs(STORE_DIR, exist_ok=True)
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def list_saved_books() -> list[dict]:
    """Return list of saved book entries: [{alias, pdf_name, created_at}, ...]"""
    index = _load_index()
    result = []
    for alias, meta in index.items():
        result.append({
            "alias": alias,
            "pdf_name": meta.get("pdf_name", "Unknown"),
            "created_at": meta.get("created_at", ""),
        })
    return sorted(result, key=lambda x: x["created_at"], reverse=True)


def list_aliases() -> list[str]:
    """Return just the alias names for dropdown."""
    return [b["alias"] for b in list_saved_books()]


def save_vectorstore(vectorstore, alias: str, pdf_name: str):
    """Save a FAISS vectorstore under the given alias."""
    alias_clean = alias.strip().replace(" ", "_").replace("/", "_")
    save_path = os.path.join(STORE_DIR, alias_clean)
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)

    index = _load_index()
    index[alias_clean] = {
        "path": save_path,
        "pdf_name": pdf_name,
        "created_at": datetime.now().isoformat(),
        "alias_display": alias.strip(),
    }
    _save_index(index)
    return alias_clean


def load_vectorstore(alias: str, embeddings) -> object:
    """Load a FAISS vectorstore by alias. Returns vectorstore or raises."""
    from langchain_community.vectorstores import FAISS
    index = _load_index()
    alias_clean = alias.strip().replace(" ", "_").replace("/", "_")
    if alias_clean not in index:
        raise KeyError(f"No saved embedding found for alias: '{alias}'")
    path = index[alias_clean]["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embedding folder missing: {path}")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def delete_vectorstore(alias: str):
    """Delete a saved embedding."""
    index = _load_index()
    alias_clean = alias.strip().replace(" ", "_").replace("/", "_")
    if alias_clean in index:
        path = index[alias_clean]["path"]
        if os.path.exists(path):
            shutil.rmtree(path)
        del index[alias_clean]
        _save_index(index)