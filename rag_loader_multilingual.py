#!/usr/bin/env python3
"""
RAG Loader for Multilingual Embedding Models
Loads Amazon Reviews Multi dataset (DE, EN, ES, FR, JA, ZH) via kagglehub,
creates embeddings and indexes them in Qdrant with language metadata.

Supports HuggingFace models via sentence-transformers
Optimized for:
  - Mac M4 with MPS (Metal Performance Shaders)
  - RunPod/Cloud with CUDA + INT8 Quantization
"""

import os
import uuid
import time
import logging
import argparse
from typing import List, Dict, Any, Optional

import torch
import pandas as pd
import kagglehub
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from tqdm import tqdm

# ─────────────────────────── LOGGING SETUP ─────────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("multilingual_loader")

# ─────────────────────────── ENV & CONFIG ───────────────────────────── #
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_CLOUD_URL") or os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

SUPPORTED_LANGUAGES = ["de", "en", "es", "fr", "ja", "zh"]

# ─────────────────────────── DEVICE DETECTION ───────────────────────── #
def get_device() -> str:
    """Detect best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"CUDA available: {device_name}")
        return "cuda"
    elif torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon) available")
        return "mps"
    else:
        logger.info("Using CPU")
        return "cpu"


# ─────────────────────────── MODEL CONFIGS ──────────────────────────── #
MODEL_CONFIGS = {
    # ═══════════════════════════════════════════════════════════════════
    # SMALL MODELS (< 500M parameters)
    # ═══════════════════════════════════════════════════════════════════
    "e5_small": {
        "hf_model": "intfloat/multilingual-e5-small",
        "dimension": 384,
        "max_length": 512,
        "batch_size": 32,
        "gpu_batch_size": 256,
        "query_prefix": "query: ",
        "doc_prefix": "passage: "
    },
    "e5_base": {
        "hf_model": "intfloat/multilingual-e5-base",
        "dimension": 768,
        "max_length": 512,
        "batch_size": 16,
        "gpu_batch_size": 128,
        "query_prefix": "query: ",
        "doc_prefix": "passage: "
    },
    "mpnet_multilingual": {
        "hf_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "dimension": 768,
        "max_length": 512,
        "batch_size": 16,
        "gpu_batch_size": 128,
        "prefix": ""
    },
    "minilm_multilingual": {
        "hf_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384,
        "max_length": 512,
        "batch_size": 32,
        "gpu_batch_size": 256,
        "prefix": ""
    },
    "minilm_v2": {
        "hf_model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "max_length": 256,
        "batch_size": 64,
        "gpu_batch_size": 256,
        "prefix": ""
    },
    "qwen3_emb_06b": {
        "hf_model": "Qwen/Qwen3-Embedding-0.6B",
        "dimension": 1024,
        "max_length": 8192,
        "batch_size": 8,
        "gpu_batch_size": 32,
        "prefix": "",
        "query_prompt_name": "query",
    },

    # ═══════════════════════════════════════════════════════════════════
    # MEDIUM MODELS (500M - 1B parameters)
    # ═══════════════════════════════════════════════════════════════════
    "e5_large_instruct": {
        "hf_model": "intfloat/multilingual-e5-large-instruct",
        "dimension": 1024,
        "max_length": 512,
        "batch_size": 8,
        "gpu_batch_size": 64,
        "query_prefix": "query: ",
        "doc_prefix": "passage: "
    },
    "bge_m3": {
        "hf_model": "BAAI/bge-m3",
        "dimension": 1024,
        "max_length": 512,
        "batch_size": 8,
        "gpu_batch_size": 64,
        "prefix": ""
    },
    "gte_multilingual_base": {
        "hf_model": "Alibaba-NLP/gte-multilingual-base",
        "dimension": 768,
        "max_length": 512,
        "batch_size": 16,
        "gpu_batch_size": 128,
        "prefix": "",
        "trust_remote_code": True
    },
    "jina_v3": {
        "hf_model": "jinaai/jina-embeddings-v3",
        "dimension": 1024,
        "max_length": 512,
        "batch_size": 8,
        "gpu_batch_size": 64,
        "prefix": "",
        "trust_remote_code": True
    },
    "nomic_embed_v1_5": {
        "hf_model": "nomic-ai/nomic-embed-text-v1.5",
        "dimension": 768,
        "max_length": 512,
        "batch_size": 16,
        "gpu_batch_size": 128,
        "doc_prefix": "search_document: ",
        "query_prefix": "search_query: ",
        "trust_remote_code": True
    },

    # ═══════════════════════════════════════════════════════════════════
    # LARGE MODELS (7B+ parameters) - CUDA + INT8 RECOMMENDED
    # ═══════════════════════════════════════════════════════════════════
    "e5_mistral_7b": {
        "hf_model": "intfloat/e5-mistral-7b-instruct",
        "dimension": 4096,
        "max_length": 512,
        "batch_size": 2,
        "gpu_batch_size": 16,
        "query_prefix": "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ",
        "doc_prefix": "",
        "load_in_8bit": True,
        "use_fp16": True
    },
    "gte_qwen2_7b": {
        "hf_model": "Alibaba-NLP/gte-Qwen2-7B-instruct",
        "dimension": 3584,
        "max_length": 512,
        "batch_size": 1,
        "gpu_batch_size": 8,
        "query_prefix": "Instruct: Given a query, retrieve relevant passages\nQuery: ",
        "doc_prefix": "",
        "trust_remote_code": True,
        "load_in_8bit": True,
        "use_fp16": True
    },
    "llama_embed_nemotron_8b": {
        "hf_model": "nvidia/llama-embed-nemotron-8b",
        "dimension": 4096,
        "max_length": 4096,
        "batch_size": 1,
        "gpu_batch_size": 16,
        "prefix": "",
        "trust_remote_code": True,
        "load_in_8bit": False,
        "use_fp16": True,
        "use_native_encode": True,
        "model_kwargs": {
            "attn_implementation": "flash_attention_2",
            "torch_dtype": "bfloat16"
        },
        "tokenizer_kwargs": {
            "padding_side": "left"
        }
    },
}


# ──────────────────────────── EMBEDDING MODEL CLASS ─────────────────── #
class OpenSourceEmbeddings:
    """Wrapper for open-source embedding models using sentence-transformers"""

    def __init__(self, model_config: Dict[str, Any], device: str = None):
        self.config = dict(model_config)  # copy to avoid mutating global config
        self.device = device or get_device()

        # Use gpu_batch_size on CUDA if available
        if self.device == "cuda" and "gpu_batch_size" in self.config:
            old_bs = self.config["batch_size"]
            self.config["batch_size"] = self.config["gpu_batch_size"]
            logger.info(f"GPU detected: batch_size {old_bs} -> {self.config['batch_size']}")

        logger.info(f"Loading model: {model_config['hf_model']}")
        logger.info(f"Device: {self.device}, Expected Dimension: {model_config['dimension']}")

        # Build model kwargs
        model_kwargs = {}

        # Handle trust_remote_code
        if model_config.get("trust_remote_code", False):
            model_kwargs["trust_remote_code"] = True

        # Handle tokenizer kwargs
        tokenizer_kwargs = model_config.get("tokenizer_kwargs", {})
        if tokenizer_kwargs:
            model_kwargs["tokenizer_kwargs"] = tokenizer_kwargs
            logger.info(f"Tokenizer kwargs: {tokenizer_kwargs}")

        # ═══════════════════════════════════════════════════════════════
        # INT8 QUANTIZATION (CUDA ONLY)
        # ═══════════════════════════════════════════════════════════════
        if model_config.get("load_in_8bit", False) and self.device == "cuda":
            logger.info("Attempting INT8 quantization (bitsandbytes)...")
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
                if "model_kwargs" not in model_kwargs:
                    model_kwargs["model_kwargs"] = {}
                model_kwargs["model_kwargs"]["quantization_config"] = quantization_config
                model_kwargs["model_kwargs"]["device_map"] = "auto"
                logger.info("INT8 quantization configured")
            except ImportError:
                logger.warning("bitsandbytes not installed, falling back to FP16")
                if "model_kwargs" not in model_kwargs:
                    model_kwargs["model_kwargs"] = {}
                model_kwargs["model_kwargs"]["torch_dtype"] = torch.float16

        # ═══════════════════════════════════════════════════════════════
        # FP16 / BFLOAT16 HANDLING
        # ═══════════════════════════════════════════════════════════════
        elif model_config.get("use_fp16", False):
            if "model_kwargs" not in model_kwargs:
                model_kwargs["model_kwargs"] = {}

            custom_model_kwargs = model_config.get("model_kwargs", {})
            if custom_model_kwargs:
                if "torch_dtype" in custom_model_kwargs:
                    dtype_str = custom_model_kwargs["torch_dtype"]
                    if dtype_str == "bfloat16":
                        custom_model_kwargs["torch_dtype"] = torch.bfloat16
                    elif dtype_str == "float16":
                        custom_model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["model_kwargs"].update(custom_model_kwargs)
                logger.info(f"Custom model_kwargs applied: {list(custom_model_kwargs.keys())}")

            if "torch_dtype" not in model_kwargs.get("model_kwargs", {}):
                model_kwargs["model_kwargs"]["torch_dtype"] = torch.float16

            logger.info("Loading with FP16/BF16 precision")

        # Handle CUDA-only features on non-CUDA devices
        if self.device != "cuda":
            if model_config.get("load_in_8bit", False):
                logger.warning("INT8 quantization requires CUDA - skipping")
            if "model_kwargs" in model_kwargs:
                if model_kwargs["model_kwargs"].get("attn_implementation") == "flash_attention_2":
                    model_kwargs["model_kwargs"]["attn_implementation"] = "eager"
                    logger.info("Switched to eager attention (flash_attention_2 requires CUDA)")

        # ═══════════════════════════════════════════════════════════════
        # LOAD MODEL
        # ═══════════════════════════════════════════════════════════════
        try:
            self.model = SentenceTransformer(
                model_config["hf_model"],
                device=self.device,
                **model_kwargs
            )
            logger.info("Model loaded successfully")

            self.has_native_encode = (
                model_config.get("use_native_encode", False) and
                hasattr(self.model, 'encode_query') and
                hasattr(self.model, 'encode_document')
            )
            if self.has_native_encode:
                logger.info("Model has native encode_query/encode_document methods")

        except TypeError as e:
            if "unexpected keyword argument" in str(e):
                logger.warning(f"Model doesn't support some kwargs, retrying: {e}")
                model_kwargs_clean = {"trust_remote_code": model_config.get("trust_remote_code", False)}
                self.model = SentenceTransformer(
                    model_config["hf_model"],
                    device=self.device,
                    **model_kwargs_clean
                )
                logger.info("Model loaded (with minimal kwargs)")
                self.has_native_encode = False
            else:
                raise

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # ═══════════════════════════════════════════════════════════════
        # POST-LOAD HOTFIXES
        # ═══════════════════════════════════════════════════════════════
        self._apply_hotfixes()

        # Set max sequence length
        try:
            self.model.max_seq_length = self.config.get("max_length", 512)
            logger.info(f"Max seq length set to: {self.model.max_seq_length}")
        except Exception as e:
            logger.warning(f"Could not set max_seq_length: {e}")

        # Get actual embedding dimension
        if hasattr(self.model, 'get_sentence_embedding_dimension'):
            self.dimension = self.model.get_sentence_embedding_dimension()
        else:
            self.dimension = model_config["dimension"]

        logger.info(f"Embedding dimension: {self.dimension}")

    def _apply_hotfixes(self):
        """Apply model-specific hotfixes after loading"""
        hf_model = self.config["hf_model"].lower()

        try:
            if hasattr(self.model, '_first_module'):
                transformer = self.model._first_module()
                if hasattr(transformer, 'auto_model'):
                    auto_model = transformer.auto_model
                else:
                    return
            elif hasattr(self.model, '__getitem__'):
                try:
                    transformer = self.model[0]
                    if hasattr(transformer, 'auto_model'):
                        auto_model = transformer.auto_model
                    else:
                        return
                except Exception:
                    return
            else:
                return

            # Qwen2: use_cache fix
            if "qwen2" in hf_model:
                if hasattr(auto_model, 'config'):
                    auto_model.config.use_cache = False
                    logger.info("HOTFIX: use_cache=False applied (Qwen2)")

        except Exception as e:
            logger.debug(f"Hotfix not applicable: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        if not texts:
            return []

        if self.has_native_encode:
            logger.debug("Using native encode_document method")
            try:
                embeddings = self.model.encode_document(
                    texts,
                    batch_size=self.config["batch_size"],
                    show_progress_bar=False
                )
                if hasattr(embeddings, 'cpu'):
                    embeddings = embeddings.cpu()
                if hasattr(embeddings, 'numpy'):
                    embeddings = embeddings.numpy()
                if hasattr(embeddings, 'tolist'):
                    return embeddings.tolist()
                return [list(emb) for emb in embeddings]
            except Exception as e:
                logger.warning(f"Native encode_document failed, falling back: {e}")

        doc_prefix = self.config.get("doc_prefix", "")
        if doc_prefix:
            texts = [doc_prefix + text for text in texts]

        embeddings = self.model.encode(
            texts,
            batch_size=self.config["batch_size"],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query"""
        if self.has_native_encode:
            logger.debug("Using native encode_query method")
            try:
                embedding = self.model.encode_query([query])
                if hasattr(embedding, 'cpu'):
                    embedding = embedding.cpu()
                if hasattr(embedding, 'numpy'):
                    embedding = embedding.numpy()
                if len(embedding.shape) > 1:
                    embedding = embedding[0]
                if hasattr(embedding, 'tolist'):
                    return embedding.tolist()
                return list(embedding)
            except Exception as e:
                logger.warning(f"Native encode_query failed, falling back: {e}")

        query_prefix = self.config.get("query_prefix", "")
        if query_prefix:
            query = query_prefix + query

        encode_kwargs = {
            "convert_to_numpy": True,
            "normalize_embeddings": True,
        }
        prompt_name = self.config.get("query_prompt_name")
        if prompt_name:
            encode_kwargs["prompt_name"] = prompt_name

        embedding = self.model.encode(
            query,
            **encode_kwargs
        )

        return embedding.tolist()


# ─────────────────────────── DATASET LOADING ───────────────────────── #
def load_dataset_csv(split: str = "train.csv") -> pd.DataFrame:
    """Load Amazon Reviews Multi dataset via kagglehub"""
    logger.info(f"Downloading/loading dataset via kagglehub (split: {split})...")
    dataset_path = kagglehub.dataset_download("mexwell/amazon-reviews-multi")
    csv_path = os.path.join(dataset_path, split)
    logger.info(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {split}")
    return df


def sample_reviews_by_language(
    df: pd.DataFrame,
    max_per_lang: int = 5000
) -> pd.DataFrame:
    """Sample up to max_per_lang reviews per language"""
    sampled_dfs = []
    for lang in SUPPORTED_LANGUAGES:
        lang_df = df[df["language"] == lang]
        if len(lang_df) > max_per_lang:
            lang_df = lang_df.sample(n=max_per_lang, random_state=42)
        sampled_dfs.append(lang_df)
        logger.info(f"  {lang}: {len(lang_df)} reviews sampled")

    result = pd.concat(sampled_dfs, ignore_index=True)
    logger.info(f"Total sampled: {len(result)} reviews across {len(SUPPORTED_LANGUAGES)} languages")
    return result


# ─────────────────────────── DATA LOADING & INDEXING ────────────────── #
def load_and_index_data(
    model_name: str,
    max_reviews_per_lang: Optional[int] = None
):
    """
    Load multilingual reviews, create embeddings, and index them in Qdrant.
    Single collection with language metadata for filtering.
    """
    if model_name not in MODEL_CONFIGS:
        logger.error(f"Unknown model: {model_name}")
        logger.info(f"Available models: {list(MODEL_CONFIGS.keys())}")
        return

    config = MODEL_CONFIGS[model_name]

    # Initialize embedding model
    device = get_device()
    embeddings_model = OpenSourceEmbeddings(config, device=device)

    # Initialize Qdrant
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
        logger.info(f"Connected to Qdrant at {QDRANT_URL}")
    except Exception as e:
        logger.error(f"Failed to connect to Qdrant: {e}")
        return

    # Collection name
    collection_name = f"multilingual_{model_name}"
    logger.info(f"Collection name: {collection_name}")

    # Create or recreate collection
    try:
        qdrant_client.delete_collection(collection_name)
        logger.info(f"Deleted existing collection: {collection_name}")
        time.sleep(2)  # Wait for Qdrant to fully process deletion
    except Exception:
        pass

    try:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=embeddings_model.dimension,
                distance=models.Distance.COSINE
            )
        )
        # Create payload index on language field for efficient filtering
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name="language",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        logger.info(f"Created collection with dimension {embeddings_model.dimension}")
        logger.info("Created payload index on 'language' field")
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        return

    # Load dataset
    df = load_dataset_csv("train.csv")

    # Sample by language (0 = use all reviews)
    if max_reviews_per_lang and max_reviews_per_lang > 0:
        df = sample_reviews_by_language(df, max_per_lang=max_reviews_per_lang)
    else:
        logger.info("Using ALL reviews (no sampling)")
        for lang in SUPPORTED_LANGUAGES:
            lang_count = len(df[df["language"] == lang])
            logger.info(f"  {lang}: {lang_count} reviews")
        logger.info(f"Total: {len(df)} reviews")

    # Process reviews in batches
    logger.info("Processing reviews...")

    texts_to_embed = []
    payloads = []
    total_processed = 0
    total_skipped = 0

    embedding_batch_size = config["batch_size"]
    upsert_batch_size = 5000 if device == "cuda" else 200
    points_batch = []

    start_time = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding reviews", unit=" reviews"):
        try:
            review_title = str(row.get("review_title", "")) if pd.notna(row.get("review_title")) else ""
            review_body = str(row.get("review_body", "")) if pd.notna(row.get("review_body")) else ""
            language = str(row.get("language", ""))
            product_id = str(row.get("product_id", ""))
            product_category = str(row.get("product_category", "")) if pd.notna(row.get("product_category")) else ""
            review_rating = int(row.get("stars", 0)) if pd.notna(row.get("stars")) else 0

            if not review_body and not review_title:
                total_skipped += 1
                continue

            # Build enriched text
            text_parts = []
            if review_title:
                text_parts.append(f"Review Title: {review_title}")
            if review_body:
                text_parts.append(f"Review: {review_body}")

            full_text = "\n".join(text_parts)

            # Create payload
            payload = {
                "page_content": full_text,
                "product_id": product_id,
                "product_category": product_category,
                "language": language,
                "review_rating": review_rating,
                "review_title": review_title,
                "review_body": review_body,
            }

            texts_to_embed.append(full_text)
            payloads.append(payload)
            total_processed += 1

            # Process batch when full
            if len(texts_to_embed) >= embedding_batch_size:
                embeddings = embeddings_model.embed_documents(texts_to_embed)

                for emb, pl in zip(embeddings, payloads):
                    points_batch.append(
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=emb,
                            payload=pl
                        )
                    )

                texts_to_embed = []
                payloads = []

                # Upsert to Qdrant when batch is full
                if len(points_batch) >= upsert_batch_size:
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=points_batch,
                        wait=True
                    )
                    logger.info(f"Uploaded {len(points_batch)} points (Total: {total_processed})")
                    points_batch = []

        except Exception as e:
            logger.warning(f"Error processing review: {e}")
            total_skipped += 1
            continue

    # Process remaining items
    if texts_to_embed:
        logger.info("Processing final batch...")
        embeddings = embeddings_model.embed_documents(texts_to_embed)

        for emb, pl in zip(embeddings, payloads):
            points_batch.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb,
                    payload=pl
                )
            )

    # Final upsert
    if points_batch:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points_batch,
            wait=True
        )
        logger.info(f"Uploaded final batch of {len(points_batch)} points")

    duration = time.time() - start_time

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DATA LOADING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Model: {model_name}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Total processed: {total_processed}")
    logger.info(f"Total skipped: {total_skipped}")
    logger.info(f"Vector dimension: {embeddings_model.dimension}")
    logger.info(f"Reviews per language: {'ALL' if not max_reviews_per_lang or max_reviews_per_lang <= 0 else max_reviews_per_lang}")
    logger.info(f"Duration: {duration:.2f}s ({total_processed / duration:.1f} reviews/s)")
    logger.info("=" * 60)


# ─────────────────────────── MAIN CLI ───────────────────────────── #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load multilingual embedding models into Qdrant"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model name to use"
    )
    parser.add_argument(
        "--max_reviews_per_lang",
        type=int,
        default=5000,
        help="Maximum number of reviews per language (default: 5000, 0 = use all reviews)"
    )

    args = parser.parse_args()

    logger.info(f"Starting multilingual data loading for model: {args.model}")

    load_and_index_data(
        model_name=args.model,
        max_reviews_per_lang=args.max_reviews_per_lang
    )
