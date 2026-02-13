#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multilingual RAG Evaluation for Embedding Models

Three evaluation modes:
  1. Monolingual: Each language evaluated separately (Qdrant filter by language)
  2. Cross-lingual: Queries search across ALL languages (no filter)
  3. True cross-lingual: Translated queries search across ALL languages
     (e.g. Japanese question finds German review). No cosine fallback.

Metrics:
  - Per-language Top-1/3/5 accuracy, latency (avg/p95/p99), throughput
  - Cross-lingual accuracy, language gap score, retrieval language distribution matrix
  - True cross-lingual: 6x6 language pair matrix, per-pair accuracy

Usage:
python evaluate_multilingual.py \
  --model bge_m3 \
  --queries_file benchmark_queries_multilingual.json \
  --mode monolingual \
  --top_k 5 \
  --output_dir evaluation_results

# True cross-lingual (requires translated queries):
python evaluate_multilingual.py \
  --model bge_m3 \
  --queries_file benchmark_queries_crosslingual.json \
  --mode true_crosslingual \
  --top_k 5 \
  --output_dir evaluation_results
"""

import os
import json
import time
import logging
import argparse
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict

import torch
import numpy as np
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from tqdm import tqdm

# ─────────── LOGGING ─────────── #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("multilingual_evaluator")

# ─────────── ENV ─────────── #
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_CLOUD_URL") or os.getenv("QDRANT_LOCAL_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Embedding side
from rag_loader_multilingual import MODEL_CONFIGS, OpenSourceEmbeddings, SUPPORTED_LANGUAGES

# ─────────── DATA CLASSES ─────────── #
@dataclass
class QueryResult:
    query_id: str
    query_text: str
    ground_truth: str
    language: str
    retrieved_contexts: List[str]
    retrieved_scores: List[float]
    retrieved_languages: List[str]
    top_k_hit: bool
    rank_of_ground_truth: Optional[int]
    avg_retrieval_score: float
    latency_ms: float


# ─────────── UTILS ─────────── #
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    try:
        a = np.array(vec1, dtype=np.float32)
        b = np.array(vec2, dtype=np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)
    except Exception as e:
        logger.warning(f"Error calculating cosine similarity: {e}")
        return 0.0


# ─────────── EVALUATOR ─────────── #
class MultilingualRAGEvaluator:
    """
    Multilingual retrieval evaluator.
    - Monolingual mode: filters Qdrant by query language
    - Cross-lingual mode: searches across all languages
    - Ground truth matching: product_id match (primary), semantic similarity fallback
    """

    def __init__(
        self,
        model_name: str,
        collection_name: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        no_cosine_fallback: bool = False,
    ):
        self.model_name = model_name
        self.collection_name = collection_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.no_cosine_fallback = no_cosine_fallback

        # Embedding model
        config = MODEL_CONFIGS[model_name]
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.embeddings_model = OpenSourceEmbeddings(config, device=device)

        # Qdrant
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=120.0
        )

        self.vector_dimension = getattr(self.embeddings_model, "dimension", 0)

        logger.info(f"Initialized evaluator for {model_name}")
        logger.info(f"Collection: {collection_name}")
        logger.info(f"Top-K: {top_k}")

    def _search_qdrant(
        self,
        query_text: str,
        language_filter: Optional[str] = None
    ) -> Tuple[List[str], List[float], List[str], float, list]:
        """
        Search Qdrant. Returns contexts, scores, languages, latency_ms, raw results.
        If language_filter is set, restricts results to that language.
        Retries once on timeout.
        """
        start = time.perf_counter()
        q_emb = self.embeddings_model.embed_query(query_text)

        query_filter = None
        if language_filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="language",
                        match=models.MatchValue(value=language_filter)
                    )
                ]
            )

        for attempt in range(2):
            try:
                results = self.qdrant_client.search(
                    collection_name=self.collection_name,
                    query_vector=q_emb,
                    query_filter=query_filter,
                    limit=self.top_k,
                    with_payload=True
                )
                break
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"Qdrant search failed (retrying): {e}")
                    time.sleep(2)
                else:
                    logger.error(f"Qdrant search failed after retry: {e}")
                    results = []

        latency_ms = (time.perf_counter() - start) * 1000

        ctxs, scores, langs = [], [], []
        for r in results:
            ctxs.append(r.payload.get("page_content", ""))
            scores.append(float(r.score))
            langs.append(r.payload.get("language", ""))
        return ctxs, scores, langs, latency_ms, results

    def evaluate_single_query(
        self,
        query_obj: Dict[str, Any],
        mode: str = "monolingual"
    ) -> Optional[QueryResult]:
        query_text = query_obj.get("question", "")
        gt_answer = query_obj.get("answer", "")
        qid = query_obj.get("id", str(hash(query_text)))
        gt_product_id = query_obj.get("product_id")
        query_lang = query_obj.get("language", "en")

        if not query_text or not gt_answer:
            logger.warning(f"Skipping invalid query: {qid}")
            return None

        # Determine filter
        lang_filter = query_lang if mode == "monolingual" else None

        contexts, scores, langs, latency_ms, raw = self._search_qdrant(
            query_text, language_filter=lang_filter
        )

        if not raw:
            return QueryResult(
                query_id=qid,
                query_text=query_text,
                ground_truth=gt_answer,
                language=query_lang,
                retrieved_contexts=[],
                retrieved_scores=[],
                retrieved_languages=[],
                top_k_hit=False,
                rank_of_ground_truth=None,
                avg_retrieval_score=0.0,
                latency_ms=latency_ms
            )

        # 1) product_id exact match (primary)
        rank_of_gt = None
        top_k_hit = False
        if gt_product_id:
            for rank, r in enumerate(raw, start=1):
                res_product_id = r.payload.get("product_id")
                if res_product_id and res_product_id == gt_product_id:
                    rank_of_gt = rank
                    top_k_hit = True
                    break

        # 2) Semantic similarity fallback (skipped if no_cosine_fallback is set)
        if not top_k_hit and not self.no_cosine_fallback:
            try:
                gt_emb = self.embeddings_model.embed_query(gt_answer)
                best_sim, best_rank = 0.0, None
                for rank, ctx in enumerate(contexts, start=1):
                    ctx_emb = self.embeddings_model.embed_query(ctx)
                    sim = cosine_similarity(gt_emb, ctx_emb)
                    if sim > best_sim:
                        best_sim, best_rank = sim, rank
                if best_sim > self.similarity_threshold:
                    rank_of_gt = best_rank
                    top_k_hit = True
            except Exception as e:
                logger.warning(f"Similarity fallback failed for {qid}: {e}")

        return QueryResult(
            query_id=qid,
            query_text=query_text,
            ground_truth=gt_answer,
            language=query_lang,
            retrieved_contexts=contexts,
            retrieved_scores=scores,
            retrieved_languages=langs,
            top_k_hit=top_k_hit,
            rank_of_ground_truth=rank_of_gt,
            avg_retrieval_score=float(np.mean(scores)) if scores else 0.0,
            latency_ms=latency_ms
        )

    def evaluate_monolingual(
        self,
        queries: List[Dict[str, Any]]
    ) -> Tuple[List[QueryResult], Dict[str, Any]]:
        """Evaluate each language separately with Qdrant filter"""
        logger.info(f"MONOLINGUAL evaluation: {len(queries)} queries")
        all_results: List[QueryResult] = []
        per_lang_metrics = {}

        for lang in SUPPORTED_LANGUAGES:
            lang_queries = [q for q in queries if q.get("language") == lang]
            if not lang_queries:
                logger.info(f"  {lang}: no queries, skipping")
                continue

            logger.info(f"  Evaluating {lang}: {len(lang_queries)} queries")
            lang_results = []
            lang_latencies = []

            for q in tqdm(lang_queries, desc=f"Eval {lang}", unit=" queries"):
                r = self.evaluate_single_query(q, mode="monolingual")
                if r:
                    lang_results.append(r)
                    lang_latencies.append(r.latency_ms)

            all_results.extend(lang_results)

            total = len(lang_results)
            if total == 0:
                continue

            top_1 = sum(1 for r in lang_results if r.rank_of_ground_truth == 1) / total
            top_3 = sum(1 for r in lang_results if r.rank_of_ground_truth and r.rank_of_ground_truth <= 3) / total
            top_5 = sum(1 for r in lang_results if r.rank_of_ground_truth and r.rank_of_ground_truth <= 5) / total
            avg_lat = float(np.mean(lang_latencies))
            p95 = float(np.percentile(lang_latencies, 95))
            p99 = float(np.percentile(lang_latencies, 99))

            per_lang_metrics[lang] = {
                "top_1": round(top_1, 4),
                "top_3": round(top_3, 4),
                "top_5": round(top_5, 4),
                "avg_latency_ms": round(avg_lat, 2),
                "p95_latency_ms": round(p95, 2),
                "p99_latency_ms": round(p99, 2),
                "total_queries": total,
            }

        # Overall average
        active_langs = [m for m in per_lang_metrics.values()]
        if active_langs:
            overall_avg = {
                "top_1": round(np.mean([m["top_1"] for m in active_langs]), 4),
                "top_3": round(np.mean([m["top_3"] for m in active_langs]), 4),
                "top_5": round(np.mean([m["top_5"] for m in active_langs]), 4),
            }
        else:
            overall_avg = {"top_1": 0, "top_3": 0, "top_5": 0}

        metrics = {
            "model_name": self.model_name,
            "mode": "monolingual",
            "per_language": per_lang_metrics,
            "overall_avg": overall_avg,
        }

        return all_results, metrics

    def evaluate_crosslingual(
        self,
        queries: List[Dict[str, Any]]
    ) -> Tuple[List[QueryResult], Dict[str, Any]]:
        """Evaluate cross-lingual: queries search across ALL languages"""
        logger.info(f"CROSS-LINGUAL evaluation: {len(queries)} queries")
        all_results: List[QueryResult] = []
        all_latencies: List[float] = []

        # Retrieval language distribution matrix
        retrieval_matrix: Dict[str, Dict[str, int]] = {
            lang: defaultdict(int) for lang in SUPPORTED_LANGUAGES
        }

        for q in tqdm(queries, desc="Cross-lingual eval", unit=" queries"):
            r = self.evaluate_single_query(q, mode="crosslingual")
            if r:
                all_results.append(r)
                all_latencies.append(r.latency_ms)

                # Track retrieval language distribution
                query_lang = r.language
                for retrieved_lang in r.retrieved_languages:
                    if retrieved_lang in SUPPORTED_LANGUAGES:
                        retrieval_matrix[query_lang][retrieved_lang] += 1

        total = len(all_results)
        if total == 0:
            return all_results, {"model_name": self.model_name, "mode": "crosslingual"}

        # Overall cross-lingual accuracy
        top_1 = sum(1 for r in all_results if r.rank_of_ground_truth == 1) / total
        top_3 = sum(1 for r in all_results if r.rank_of_ground_truth and r.rank_of_ground_truth <= 3) / total
        top_5 = sum(1 for r in all_results if r.rank_of_ground_truth and r.rank_of_ground_truth <= 5) / total

        # Per-language cross-lingual accuracy
        per_lang_accuracy = {}
        for lang in SUPPORTED_LANGUAGES:
            lang_results = [r for r in all_results if r.language == lang]
            if lang_results:
                lt = len(lang_results)
                per_lang_accuracy[lang] = {
                    "top_1": round(sum(1 for r in lang_results if r.rank_of_ground_truth == 1) / lt, 4),
                    "top_3": round(sum(1 for r in lang_results if r.rank_of_ground_truth and r.rank_of_ground_truth <= 3) / lt, 4),
                    "top_5": round(sum(1 for r in lang_results if r.rank_of_ground_truth and r.rank_of_ground_truth <= 5) / lt, 4),
                    "total_queries": lt,
                }

        # Language gap: EN accuracy - avg(other languages)
        en_acc = per_lang_accuracy.get("en", {}).get("top_1", 0)
        other_accs = [per_lang_accuracy[l]["top_1"] for l in SUPPORTED_LANGUAGES if l != "en" and l in per_lang_accuracy]
        language_gap = round(en_acc - np.mean(other_accs), 4) if other_accs else 0.0

        # Convert retrieval matrix to serializable format
        matrix_serializable = {}
        for query_lang, retrieved_counts in retrieval_matrix.items():
            key = f"{query_lang}_query"
            matrix_serializable[key] = {
                f"{rl}_retrieved": count
                for rl, count in sorted(retrieved_counts.items())
            }

        avg_lat = float(np.mean(all_latencies))
        p95 = float(np.percentile(all_latencies, 95))
        p99 = float(np.percentile(all_latencies, 99))
        qps = (1000.0 / avg_lat) if avg_lat > 0 else 0.0

        metrics = {
            "model_name": self.model_name,
            "mode": "crosslingual",
            "overall": {
                "top_1": round(top_1, 4),
                "top_3": round(top_3, 4),
                "top_5": round(top_5, 4),
            },
            "per_language": per_lang_accuracy,
            "language_gap": language_gap,
            "retrieval_matrix": matrix_serializable,
            "performance": {
                "avg_latency_ms": round(avg_lat, 2),
                "p95_latency_ms": round(p95, 2),
                "p99_latency_ms": round(p99, 2),
                "throughput_qps": round(qps, 2),
            },
        }

        return all_results, metrics

    def evaluate_true_crosslingual(
        self,
        queries: List[Dict[str, Any]]
    ) -> Tuple[List[QueryResult], Dict[str, Any]]:
        """
        True cross-lingual evaluation: translated queries search across ALL languages.

        Each query has:
          - query_language: the language the question was translated TO (search language)
          - language: the original review's language (ground truth location)

        Only product_id exact match is used (no cosine similarity fallback),
        because the answer is in a different language than the query.
        """
        logger.info(f"TRUE CROSS-LINGUAL evaluation: {len(queries)} queries")
        all_results: List[QueryResult] = []
        all_latencies: List[float] = []

        # 6x6 language pair accuracy matrix: pair_results[query_lang][target_lang] = list of (hit, rank)
        pair_results: Dict[str, Dict[str, List[Tuple[bool, Optional[int]]]]] = {
            ql: {tl: [] for tl in SUPPORTED_LANGUAGES} for ql in SUPPORTED_LANGUAGES
        }

        for q in tqdm(queries, desc="True cross-lingual eval", unit=" queries"):
            query_text = q.get("question", "")
            gt_answer = q.get("answer", "")
            qid = q.get("id", str(hash(query_text)))
            gt_product_id = q.get("product_id")
            target_lang = q.get("language", "")        # original review's language
            query_lang = q.get("query_language", "")   # translated question's language

            if not query_text or not gt_product_id:
                logger.warning(f"Skipping invalid query: {qid}")
                continue

            # Search with NO language filter — search all 120K reviews
            contexts, scores, langs, latency_ms, raw = self._search_qdrant(
                query_text, language_filter=None
            )

            # Ground truth: product_id exact match ONLY (no cosine fallback)
            rank_of_gt = None
            top_k_hit = False
            if raw and gt_product_id:
                for rank, r in enumerate(raw, start=1):
                    res_product_id = r.payload.get("product_id")
                    if res_product_id and res_product_id == gt_product_id:
                        rank_of_gt = rank
                        top_k_hit = True
                        break

            result = QueryResult(
                query_id=qid,
                query_text=query_text,
                ground_truth=gt_answer,
                language=target_lang,
                retrieved_contexts=contexts,
                retrieved_scores=scores,
                retrieved_languages=langs,
                top_k_hit=top_k_hit,
                rank_of_ground_truth=rank_of_gt,
                avg_retrieval_score=float(np.mean(scores)) if scores else 0.0,
                latency_ms=latency_ms
            )
            all_results.append(result)
            all_latencies.append(latency_ms)

            # Track pair results
            if query_lang in pair_results and target_lang in pair_results[query_lang]:
                pair_results[query_lang][target_lang].append((top_k_hit, rank_of_gt))

        total = len(all_results)
        if total == 0:
            return all_results, {"model_name": self.model_name, "mode": "true_crosslingual"}

        # Overall accuracy
        top_1 = sum(1 for r in all_results if r.rank_of_ground_truth == 1) / total
        top_3 = sum(1 for r in all_results if r.rank_of_ground_truth and r.rank_of_ground_truth <= 3) / total
        top_5 = sum(1 for r in all_results if r.rank_of_ground_truth and r.rank_of_ground_truth <= 5) / total

        # 6x6 language pair matrix (Top-1 accuracy)
        pair_matrix_top1: Dict[str, Dict[str, Optional[float]]] = {}
        pair_matrix_top3: Dict[str, Dict[str, Optional[float]]] = {}
        pair_matrix_top5: Dict[str, Dict[str, Optional[float]]] = {}

        for ql in SUPPORTED_LANGUAGES:
            pair_matrix_top1[ql] = {}
            pair_matrix_top3[ql] = {}
            pair_matrix_top5[ql] = {}
            for tl in SUPPORTED_LANGUAGES:
                results_for_pair = pair_results[ql][tl]
                if not results_for_pair:
                    pair_matrix_top1[ql][tl] = None
                    pair_matrix_top3[ql][tl] = None
                    pair_matrix_top5[ql][tl] = None
                    continue
                n = len(results_for_pair)
                pair_matrix_top1[ql][tl] = round(
                    sum(1 for hit, rank in results_for_pair if rank == 1) / n, 4
                )
                pair_matrix_top3[ql][tl] = round(
                    sum(1 for hit, rank in results_for_pair if rank and rank <= 3) / n, 4
                )
                pair_matrix_top5[ql][tl] = round(
                    sum(1 for hit, rank in results_for_pair if rank and rank <= 5) / n, 4
                )

        # Per query-language average (e.g. "when asking in JA, avg accuracy")
        per_query_lang = {}
        for ql in SUPPORTED_LANGUAGES:
            ql_results = [r for r in all_results if
                          any(q.get("query_language") == ql and q.get("id") == r.query_id
                              for q in queries)]
            # Simpler: filter by checking pair_results
            ql_all = []
            for tl in SUPPORTED_LANGUAGES:
                ql_all.extend(pair_results[ql][tl])
            if ql_all:
                n = len(ql_all)
                per_query_lang[ql] = {
                    "top_1": round(sum(1 for hit, rank in ql_all if rank == 1) / n, 4),
                    "top_3": round(sum(1 for hit, rank in ql_all if rank and rank <= 3) / n, 4),
                    "top_5": round(sum(1 for hit, rank in ql_all if rank and rank <= 5) / n, 4),
                    "total_queries": n,
                }

        # Per target-language average (e.g. "how easy to find DE reviews")
        per_target_lang = {}
        for tl in SUPPORTED_LANGUAGES:
            tl_all = []
            for ql in SUPPORTED_LANGUAGES:
                tl_all.extend(pair_results[ql][tl])
            if tl_all:
                n = len(tl_all)
                per_target_lang[tl] = {
                    "top_1": round(sum(1 for hit, rank in tl_all if rank == 1) / n, 4),
                    "top_3": round(sum(1 for hit, rank in tl_all if rank and rank <= 3) / n, 4),
                    "top_5": round(sum(1 for hit, rank in tl_all if rank and rank <= 5) / n, 4),
                    "total_queries": n,
                }

        # Best/worst language pairs
        all_pairs = []
        for ql in SUPPORTED_LANGUAGES:
            for tl in SUPPORTED_LANGUAGES:
                if pair_matrix_top1[ql][tl] is not None:
                    all_pairs.append((f"{ql}->{tl}", pair_matrix_top1[ql][tl]))
        all_pairs.sort(key=lambda x: x[1], reverse=True)
        best_pairs = all_pairs[:5] if all_pairs else []
        worst_pairs = all_pairs[-5:] if all_pairs else []

        avg_lat = float(np.mean(all_latencies))
        p95 = float(np.percentile(all_latencies, 95))
        p99 = float(np.percentile(all_latencies, 99))
        qps = (1000.0 / avg_lat) if avg_lat > 0 else 0.0

        metrics = {
            "model_name": self.model_name,
            "mode": "true_crosslingual",
            "overall": {
                "top_1": round(top_1, 4),
                "top_3": round(top_3, 4),
                "top_5": round(top_5, 4),
                "total_queries": total,
            },
            "pair_matrix_top1": pair_matrix_top1,
            "pair_matrix_top3": pair_matrix_top3,
            "pair_matrix_top5": pair_matrix_top5,
            "per_query_language": per_query_lang,
            "per_target_language": per_target_lang,
            "best_pairs_top1": best_pairs,
            "worst_pairs_top1": worst_pairs,
            "performance": {
                "avg_latency_ms": round(avg_lat, 2),
                "p95_latency_ms": round(p95, 2),
                "p99_latency_ms": round(p99, 2),
                "throughput_qps": round(qps, 2),
            },
        }

        return all_results, metrics

    def run_full_evaluation(
        self,
        queries: List[Dict[str, Any]],
        mode: str = "both"
    ) -> Dict[str, Any]:
        """Run monolingual, crosslingual, true_crosslingual, or both evaluations"""
        combined_metrics = {
            "model_name": self.model_name,
            "vector_dimension": self.vector_dimension,
        }
        all_results = {}

        if mode in ("monolingual", "both"):
            mono_results, mono_metrics = self.evaluate_monolingual(queries)
            combined_metrics["monolingual"] = mono_metrics
            all_results["monolingual"] = mono_results

        if mode in ("crosslingual", "both"):
            cross_results, cross_metrics = self.evaluate_crosslingual(queries)
            combined_metrics["crosslingual"] = cross_metrics
            all_results["crosslingual"] = cross_results

        if mode == "true_crosslingual":
            tcl_results, tcl_metrics = self.evaluate_true_crosslingual(queries)
            combined_metrics["true_crosslingual"] = tcl_metrics
            all_results["true_crosslingual"] = tcl_results

        # Compute mono vs cross gap if both are available
        if "monolingual" in combined_metrics and "crosslingual" in combined_metrics:
            mono_avg = combined_metrics["monolingual"].get("overall_avg", {}).get("top_1", 0)
            cross_avg = combined_metrics["crosslingual"].get("overall", {}).get("top_1", 0)
            combined_metrics["mono_vs_cross_gap"] = round(mono_avg - cross_avg, 4)

        # Performance summary
        all_latencies = []
        for mode_results in all_results.values():
            all_latencies.extend([r.latency_ms for r in mode_results])

        if all_latencies:
            combined_metrics["performance"] = {
                "avg_latency_ms": round(float(np.mean(all_latencies)), 2),
                "p95_latency_ms": round(float(np.percentile(all_latencies, 95)), 2),
                "p99_latency_ms": round(float(np.percentile(all_latencies, 99)), 2),
                "throughput_qps": round(1000.0 / float(np.mean(all_latencies)), 2) if np.mean(all_latencies) > 0 else 0,
                "vector_dimension": self.vector_dimension,
            }

        return combined_metrics, all_results


# ─────────── CLI ─────────── #
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multilingual embedding models (monolingual + cross-lingual)"
    )
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--queries_file", type=str, default="benchmark_queries_multilingual.json")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["monolingual", "crosslingual", "both", "true_crosslingual"],
                        help="Evaluation mode")
    parser.add_argument("--similarity-threshold", type=float, default=0.7,
                        help="Fallback similarity threshold")
    parser.add_argument("--no-cosine-fallback", action="store_true",
                        help="Disable cosine similarity fallback, use only product_id exact match")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load queries
    logger.info(f"Loading queries from: {args.queries_file}")
    try:
        with open(args.queries_file, "r", encoding="utf-8") as f:
            queries = json.load(f)
        logger.info(f"Loaded {len(queries)} queries")

        # Report language distribution
        lang_dist = defaultdict(int)
        for q in queries:
            lang_dist[q.get("language", "?")] += 1
        for lang, count in sorted(lang_dist.items()):
            logger.info(f"  {lang}: {count} queries")

    except FileNotFoundError:
        logger.error(f"Queries file not found: {args.queries_file}")
        return

    collection_name = f"multilingual_{args.model}"
    evaluator = MultilingualRAGEvaluator(
        model_name=args.model,
        collection_name=collection_name,
        top_k=args.top_k,
        similarity_threshold=args.similarity_threshold,
        no_cosine_fallback=args.no_cosine_fallback,
    )

    logger.info("\n" + "=" * 60)
    logger.info("STARTING MULTILINGUAL EVALUATION")
    logger.info("=" * 60)

    t0 = time.time()
    combined_metrics, all_results = evaluator.run_full_evaluation(queries, mode=args.mode)
    duration = time.time() - t0

    # Save detailed results per mode
    for mode_name, results in all_results.items():
        results_file = os.path.join(args.output_dir, f"results_{args.model}_{mode_name}.json")
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed results saved to: {results_file}")

    # Save aggregated metrics
    metrics_file = os.path.join(args.output_dir, f"metrics_{args.model}.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(combined_metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Metrics saved to: {metrics_file}")

    # Save summary
    summary = {
        "model_name": args.model,
        "evaluation_mode": args.mode,
        "total_queries": len(queries),
        "duration_seconds": round(duration, 2),
    }
    if "monolingual" in combined_metrics:
        summary["monolingual_per_language"] = combined_metrics["monolingual"].get("per_language", {})
        summary["monolingual_overall"] = combined_metrics["monolingual"].get("overall_avg", {})
    if "crosslingual" in combined_metrics:
        summary["crosslingual_overall"] = combined_metrics["crosslingual"].get("overall", {})
        summary["language_gap"] = combined_metrics["crosslingual"].get("language_gap", 0)
        summary["retrieval_matrix"] = combined_metrics["crosslingual"].get("retrieval_matrix", {})
    if "true_crosslingual" in combined_metrics:
        tcl = combined_metrics["true_crosslingual"]
        summary["true_crosslingual"] = {
            "overall": tcl.get("overall", {}),
            "pair_matrix_top1": tcl.get("pair_matrix_top1", {}),
            "per_query_language": tcl.get("per_query_language", {}),
            "per_target_language": tcl.get("per_target_language", {}),
            "best_pairs_top1": tcl.get("best_pairs_top1", []),
            "worst_pairs_top1": tcl.get("worst_pairs_top1", []),
        }
    if "mono_vs_cross_gap" in combined_metrics:
        summary["mono_vs_cross_gap"] = combined_metrics["mono_vs_cross_gap"]

    summary_file = os.path.join(args.output_dir, f"summary_{args.model}.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Summary saved to: {summary_file}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Total queries: {len(queries)}")
    logger.info(f"Duration: {duration:.2f}s")

    if "monolingual" in combined_metrics:
        mono = combined_metrics["monolingual"]
        logger.info("\nMONOLINGUAL Results:")
        for lang, m in mono.get("per_language", {}).items():
            logger.info(f"  {lang}: Top-1={m['top_1']:.2%}  Top-3={m['top_3']:.2%}  Top-5={m['top_5']:.2%}  Latency={m['avg_latency_ms']:.1f}ms")
        overall = mono.get("overall_avg", {})
        logger.info(f"  AVG: Top-1={overall.get('top_1', 0):.2%}  Top-3={overall.get('top_3', 0):.2%}  Top-5={overall.get('top_5', 0):.2%}")

    if "crosslingual" in combined_metrics:
        cross = combined_metrics["crosslingual"]
        overall = cross.get("overall", {})
        logger.info("\nCROSS-LINGUAL Results:")
        logger.info(f"  Overall: Top-1={overall.get('top_1', 0):.2%}  Top-3={overall.get('top_3', 0):.2%}  Top-5={overall.get('top_5', 0):.2%}")
        logger.info(f"  Language gap (EN - avg others): {cross.get('language_gap', 0):.4f}")

        # Print retrieval matrix
        matrix = cross.get("retrieval_matrix", {})
        if matrix:
            logger.info("\n  Retrieval Language Distribution:")
            for query_key, retrieved in matrix.items():
                logger.info(f"    {query_key}: {dict(retrieved)}")

    if "true_crosslingual" in combined_metrics:
        tcl = combined_metrics["true_crosslingual"]
        overall = tcl.get("overall", {})
        logger.info("\nTRUE CROSS-LINGUAL Results:")
        logger.info(f"  Overall: Top-1={overall.get('top_1', 0):.2%}  Top-3={overall.get('top_3', 0):.2%}  Top-5={overall.get('top_5', 0):.2%}  (N={overall.get('total_queries', 0)})")

        # Per query language
        pql = tcl.get("per_query_language", {})
        if pql:
            logger.info("\n  Per query language (avg accuracy when asking in this language):")
            for lang in SUPPORTED_LANGUAGES:
                if lang in pql:
                    m = pql[lang]
                    logger.info(f"    {lang}: Top-1={m['top_1']:.2%}  Top-3={m['top_3']:.2%}  Top-5={m['top_5']:.2%}  (N={m['total_queries']})")

        # Per target language
        ptl = tcl.get("per_target_language", {})
        if ptl:
            logger.info("\n  Per target language (avg accuracy finding reviews in this language):")
            for lang in SUPPORTED_LANGUAGES:
                if lang in ptl:
                    m = ptl[lang]
                    logger.info(f"    {lang}: Top-1={m['top_1']:.2%}  Top-3={m['top_3']:.2%}  Top-5={m['top_5']:.2%}  (N={m['total_queries']})")

        # 6x6 matrix (Top-1)
        matrix = tcl.get("pair_matrix_top1", {})
        if matrix:
            logger.info("\n  Language Pair Matrix (Top-1 accuracy):")
            header = "         " + "  ".join(f"{tl:>6s}" for tl in SUPPORTED_LANGUAGES)
            logger.info(f"  {header}")
            for ql in SUPPORTED_LANGUAGES:
                row_vals = []
                for tl in SUPPORTED_LANGUAGES:
                    val = matrix.get(ql, {}).get(tl)
                    if val is None:
                        row_vals.append("   -  ")
                    elif ql == tl:
                        row_vals.append(" (mono)")
                    else:
                        row_vals.append(f"{val:6.1%}")
                logger.info(f"  {ql:>6s}  " + "  ".join(row_vals))

        # Best/worst pairs
        best = tcl.get("best_pairs_top1", [])
        worst = tcl.get("worst_pairs_top1", [])
        if best:
            logger.info(f"\n  Best pairs:  {', '.join(f'{p}={v:.1%}' for p, v in best[:3])}")
        if worst:
            logger.info(f"  Worst pairs: {', '.join(f'{p}={v:.1%}' for p, v in worst[:3])}")

    if "mono_vs_cross_gap" in combined_metrics:
        logger.info(f"\nMono vs Cross gap (Top-1): {combined_metrics['mono_vs_cross_gap']:.4f}")

    if "performance" in combined_metrics:
        perf = combined_metrics["performance"]
        logger.info(f"\nPerformance:")
        logger.info(f"  Avg Latency: {perf['avg_latency_ms']:.2f}ms")
        logger.info(f"  P95 Latency: {perf['p95_latency_ms']:.2f}ms")
        logger.info(f"  P99 Latency: {perf['p99_latency_ms']:.2f}ms")
        logger.info(f"  Throughput: {perf['throughput_qps']:.2f} QPS")
        logger.info(f"  Vector Dimension: {perf['vector_dimension']}")

    logger.info("=" * 60)


if __name__ == "__main__":
    main()
