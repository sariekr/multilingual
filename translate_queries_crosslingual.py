#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Translate existing multilingual benchmark queries to other languages
for true cross-lingual evaluation.

Takes 600 monolingual queries (100/language) and translates each question
to the other 5 languages, producing 3000 translated queries.

Usage:
venv/bin/python3 translate_queries_crosslingual.py \
  --input_file benchmark_queries_multilingual.json \
  --output_file benchmark_queries_crosslingual.json \
  --llm_model google/gemini-3-flash-preview \
  --target_langs all
"""

import os
import json
import argparse
import logging
import time
from typing import List, Dict, Any

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ───────────────────────── LOGGING ───────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("crosslingual_translator")

# ───────────────────────── ENV ───────────────────────── #
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

SUPPORTED_LANGUAGES = ["de", "en", "es", "fr", "ja", "zh"]

LANGUAGE_NAMES = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "ja": "Japanese",
    "zh": "Chinese",
}

# ───────────────────────── TRANSLATION PROMPT ───────────────────────── #
TRANSLATION_PROMPT = """Translate the following question to {target_language_name}.
Keep the meaning exactly the same. Only output the translated question, nothing else.

Original question ({source_language_name}):
{question}

Translated question ({target_language_name}):"""


def translate_queries(
    llm: ChatOpenAI,
    queries: List[Dict[str, Any]],
    target_langs: List[str],
) -> List[Dict[str, Any]]:
    """Translate each query's question to specified target languages."""

    prompt = PromptTemplate(
        template=TRANSLATION_PROMPT,
        input_variables=["target_language_name", "source_language_name", "question"],
    )
    chain = prompt | llm | StrOutputParser()

    translated: List[Dict[str, Any]] = []
    total = len(queries) * len(target_langs)
    done = 0
    failed = 0

    for q in queries:
        source_lang = q["language"]
        source_lang_name = LANGUAGE_NAMES[source_lang]
        original_question = q["question"]
        original_id = q["id"]

        # Only translate to languages different from source
        for target_lang in target_langs:
            if target_lang == source_lang:
                continue

            target_lang_name = LANGUAGE_NAMES[target_lang]
            done += 1

            logger.info(
                f"[{done}/{total}] Translating {original_id} "
                f"({source_lang} -> {target_lang})"
            )

            try:
                translated_question = chain.invoke({
                    "target_language_name": target_lang_name,
                    "source_language_name": source_lang_name,
                    "question": original_question,
                })
                translated_question = translated_question.strip()
            except Exception as e:
                logger.warning(
                    f"Translation failed for {original_id} "
                    f"({source_lang} -> {target_lang}): {e}"
                )
                failed += 1
                continue

            if not translated_question:
                logger.warning(f"Empty translation for {original_id} ({source_lang} -> {target_lang})")
                failed += 1
                continue

            item = {
                "id": f"{original_id}_to_{target_lang}",
                "original_id": original_id,
                "question": translated_question,
                "answer": q["answer"],           # original language answer (unchanged)
                "context": q["context"],          # original review text (unchanged)
                "product_id": q["product_id"],    # ground truth product (unchanged)
                "product_category": q.get("product_category", ""),
                "language": source_lang,          # review's original language
                "query_language": target_lang,    # language the question was translated to
                "question_type": q.get("question_type", ""),
            }
            translated.append(item)

    logger.info("\n" + "=" * 60)
    logger.info("TRANSLATION COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total translated: {len(translated)}")
    logger.info(f"Failed: {failed}")

    # Distribution summary
    pair_counts: Dict[str, int] = {}
    for t in translated:
        pair = f"{t['language']}->{t['query_language']}"
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    logger.info("Language pair distribution:")
    for pair, count in sorted(pair_counts.items()):
        logger.info(f"  {pair}: {count}")
    logger.info("=" * 60)

    return translated


# ───────────────────────── CLI ───────────────────────── #
def main():
    ap = argparse.ArgumentParser(
        description="Translate benchmark queries for true cross-lingual evaluation."
    )
    ap.add_argument("--input_file", default="benchmark_queries_multilingual.json",
                    help="Input queries file (600 monolingual queries)")
    ap.add_argument("--output_file", default="benchmark_queries_crosslingual.json",
                    help="Output file for translated queries (3000 queries)")
    ap.add_argument("--llm_model", default="google/gemini-3-flash-preview",
                    help="LLM model for translation (via OpenRouter)")
    ap.add_argument("--target_langs", default="all",
                    help="Target languages: 'all' or comma-separated (e.g. 'ja,en,fr')")
    args = ap.parse_args()

    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY is not set; set it in your environment.")
        return

    # Parse target languages
    if args.target_langs == "all":
        target_langs = SUPPORTED_LANGUAGES
    else:
        target_langs = [l.strip() for l in args.target_langs.split(",")]
        for l in target_langs:
            if l not in SUPPORTED_LANGUAGES:
                logger.error(f"Unsupported language: {l}. Supported: {SUPPORTED_LANGUAGES}")
                return

    # Load input queries
    logger.info(f"Loading queries from: {args.input_file}")
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            queries = json.load(f)
        logger.info(f"Loaded {len(queries)} queries")
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input_file}")
        return

    # Report language distribution
    lang_dist: Dict[str, int] = {}
    for q in queries:
        lang = q.get("language", "?")
        lang_dist[lang] = lang_dist.get(lang, 0) + 1
    for lang, count in sorted(lang_dist.items()):
        logger.info(f"  {lang}: {count} queries")

    # Initialize LLM
    logger.info(f"Initializing LLM: {args.llm_model}")
    llm = ChatOpenAI(
        model=args.llm_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0.1,
        timeout=60,
    )

    # Translate
    t0 = time.time()
    translated = translate_queries(llm, queries, target_langs)
    duration = time.time() - t0

    # Save
    if translated:
        logger.info(f"Saving {len(translated)} translated queries to '{args.output_file}'.")
        with open(args.output_file, "w", encoding="utf-8") as f_out:
            json.dump(translated, f_out, indent=2, ensure_ascii=False)
        logger.info(f"Saved successfully. Duration: {duration:.1f}s")

        # Print samples
        logger.info("\n" + "=" * 60)
        logger.info("SAMPLE TRANSLATIONS (first 3)")
        logger.info("=" * 60)
        for item in translated[:3]:
            logger.info(f"  [{item['language']}->{item['query_language']}] {item['id']}")
            logger.info(f"    Q: {item['question']}")
            logger.info(f"    A: {item['answer'][:80]}...")
    else:
        logger.warning("No translations generated.")


if __name__ == "__main__":
    main()
