#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate diverse multilingual benchmark queries from Amazon Reviews Multi dataset.

Generates Q&A pairs in 6 languages (DE, EN, ES, FR, JA, ZH) using LLM.
Each question and answer is generated natively in the review's language.

Usage:
python generate_multilingual_queries.py \
  --num_questions_per_lang 100 \
  --output_file benchmark_queries_multilingual.json \
  --min_review_length 100 \
  --llm_model anthropic/claude-3.5-sonnet
"""

import os
import re
import json
import html
import argparse
import logging
from enum import Enum, auto
from typing import Dict, Any, List, Optional
from collections import defaultdict

import pandas as pd
import kagglehub
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ───────────────────────── LOGGING ───────────────────────── #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("multilingual_query_generator")

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

# ───────────────────────── Helpers ───────────────────────── #
def clean_review_text(text: str) -> str:
    if not text:
        return ""
    t = html.unescape(text)
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def truncate_answer(s: str, language: str, max_words: int = 30, max_chars: int = 60) -> str:
    """Truncate answer: word-based for alphabetic languages, char-based for JA/ZH"""
    s = s.strip()
    if language in ("ja", "zh"):
        if len(s) <= max_chars:
            return s
        return s[:max_chars].rstrip(",.;:") + "..."
    else:
        words = s.split()
        if len(words) <= max_words:
            return s
        return " ".join(words[:max_words]).rstrip(",.;:") + "."


# ───────────────────────── Question Types & Prompts ───────────────────────── #
class QuestionType(Enum):
    FACTUAL = auto()
    OPINION = auto()
    USAGE = auto()
    PROBLEM_SOLVING = auto()
    FEATURE = auto()


MULTILINGUAL_GUARDRAILS = (
    "RULES:\n"
    "1) USE ONLY the REVIEW text as evidence. The product category is for topical focus only.\n"
    "2) If a valid QA cannot be formed from the REVIEW alone, output the single word: SKIP.\n"
    "3) Answer must be concise (max 30 words for alphabetic languages, max 60 characters for Japanese/Chinese).\n"
    "4) Output strictly as JSON with keys: question, answer. No extra text.\n"
    "5) You MUST write BOTH the question AND the answer in {language_name}. Do NOT translate - write natively in {language_name}.\n"
)

PROMPT_TEMPLATES: Dict[QuestionType, str] = {
    QuestionType.FACTUAL: f"""You are creating a FACTUAL question answerable strictly from the REVIEW text.
The question and answer MUST be written in {{language_name}}.

Examples (English, adapt to target language):
- "How long does the battery last on a single charge?"
- "What size is the product?"

CONTEXT
Product Category: {{product_category}}
Language: {{language_name}}
REVIEW: {{review_text}}
Rating: {{rating}}/5

{MULTILINGUAL_GUARDRAILS}

{{format_instructions}}
""",
    QuestionType.OPINION: f"""Create an OPINION question about quality/suitability/value, answerable strictly from the REVIEW.
The question and answer MUST be written in {{language_name}}.

Examples (English, adapt to target language):
- "Is this suitable for sensitive skin?"
- "Is it worth the price?"

CONTEXT
Product Category: {{product_category}}
Language: {{language_name}}
REVIEW: {{review_text}}
Rating: {{rating}}/5

{MULTILINGUAL_GUARDRAILS}

{{format_instructions}}
""",
    QuestionType.USAGE: f"""Create a USAGE question about how/when to use, answerable strictly from the REVIEW.
The question and answer MUST be written in {{language_name}}.

Examples (English, adapt to target language):
- "How should I apply this?"
- "When is the best time to use it?"

CONTEXT
Product Category: {{product_category}}
Language: {{language_name}}
REVIEW: {{review_text}}
Rating: {{rating}}/5

{MULTILINGUAL_GUARDRAILS}

{{format_instructions}}
""",
    QuestionType.PROBLEM_SOLVING: f"""Create a PROBLEM-SOLVING question (does it address a specific issue?), answerable strictly from the REVIEW.
The question and answer MUST be written in {{language_name}}.

Examples (English, adapt to target language):
- "Does this help with dry skin?"
- "Does it reduce acne breakouts?"

CONTEXT
Product Category: {{product_category}}
Language: {{language_name}}
REVIEW: {{review_text}}
Rating: {{rating}}/5

{MULTILINGUAL_GUARDRAILS}

{{format_instructions}}
""",
    QuestionType.FEATURE: f"""Create a FEATURE question about characteristics/capabilities, answerable strictly from the REVIEW.
The question and answer MUST be written in {{language_name}}.

Examples (English, adapt to target language):
- "Does this have a strong scent?"
- "What ingredients are mentioned?"

CONTEXT
Product Category: {{product_category}}
Language: {{language_name}}
REVIEW: {{review_text}}
Rating: {{rating}}/5

{MULTILINGUAL_GUARDRAILS}

{{format_instructions}}
""",
}


# ───────────────────────── Dataset Loading ───────────────────────── #
def load_dataset_csv(split: str = "train.csv") -> pd.DataFrame:
    """Load Amazon Reviews Multi dataset via kagglehub"""
    logger.info(f"Downloading/loading dataset via kagglehub (split: {split})...")
    dataset_path = kagglehub.dataset_download("mexwell/amazon-reviews-multi")
    csv_path = os.path.join(dataset_path, split)
    logger.info(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows from {split}")
    return df


# ───────────────────────── Core Generation ───────────────────────── #
def generate_multilingual_queries(
    llm: ChatOpenAI,
    df: pd.DataFrame,
    num_questions_per_lang: int = 100,
    min_review_length: int = 100,
    product_max_per_product: int = 2,
) -> List[Dict[str, Any]]:
    parser = JsonOutputParser()

    generated: List[Dict[str, Any]] = []
    total_processed = 0
    skipped_low_quality = 0
    skipped_product_quota = 0
    skipped_skip_token = 0
    skipped_parse = 0

    for lang in SUPPORTED_LANGUAGES:
        lang_name = LANGUAGE_NAMES[lang]
        lang_df = df[df["language"] == lang].copy()

        # Filter by review_body length
        lang_df = lang_df[lang_df["review_body"].astype(str).str.len() >= min_review_length]

        logger.info(f"\n--- Language: {lang} ({lang_name}) ---")
        logger.info(f"  Eligible reviews after filtering: {len(lang_df)}")

        questions_per_type = {qt: 0 for qt in QuestionType}
        target_per_type = max(1, num_questions_per_lang // len(QuestionType))
        product_counts: Dict[str, int] = defaultdict(int)
        lang_generated = 0

        types_cycle = list(QuestionType) * (num_questions_per_lang // len(QuestionType) + 2)

        for _, row in lang_df.iterrows():
            if lang_generated >= num_questions_per_lang:
                break

            total_processed += 1

            review_body = str(row.get("review_body", ""))
            review_title = str(row.get("review_title", "")) if pd.notna(row.get("review_title")) else ""
            product_id = str(row.get("product_id", ""))
            product_category = str(row.get("product_category", "")) if pd.notna(row.get("product_category")) else ""
            rating = int(row.get("stars", 0)) if pd.notna(row.get("stars")) else 0

            if len(review_body) < min_review_length:
                skipped_low_quality += 1
                continue

            # Product quota
            if product_counts[product_id] >= product_max_per_product:
                skipped_product_quota += 1
                continue

            # Clean review text
            review_text = clean_review_text(review_body)
            if review_title:
                review_text = f"{review_title}\n{review_text}"

            # Select question type (balanced distribution)
            qtype = types_cycle[lang_generated]
            if questions_per_type[qtype] >= (target_per_type + 2):
                for alt in QuestionType:
                    if questions_per_type[alt] < (target_per_type + 2):
                        qtype = alt
                        break

            prompt_template = PROMPT_TEMPLATES[qtype]
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["product_category", "review_text", "rating", "language_name"],
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )
            chain = prompt | llm | parser

            logger.info(
                f"  [{lang_generated + 1}/{num_questions_per_lang}] "
                f"Lang: {lang}, Type: {qtype.name}, Product: {product_id[:20]}..."
            )

            try:
                qa = chain.invoke({
                    "product_category": product_category,
                    "review_text": review_text,
                    "rating": rating,
                    "language_name": lang_name,
                })
            except Exception as e:
                logger.warning(f"  LLM call failed: {e}")
                continue

            if not isinstance(qa, dict) or "question" not in qa or "answer" not in qa:
                skipped_parse += 1
                continue

            question = (qa.get("question") or "").strip()
            answer = (qa.get("answer") or "").strip()

            # SKIP signal or empty fields
            if not question or not answer or question.upper() == "SKIP" or answer.upper() == "SKIP":
                skipped_skip_token += 1
                continue

            # Truncate answer
            answer = truncate_answer(answer, language=lang)

            item = {
                "id": f"q_{lang}_{lang_generated + 1:03d}",
                "question": question,
                "answer": answer,
                "context": review_text,
                "product_id": product_id,
                "product_category": product_category,
                "language": lang,
                "review_rating": rating,
                "question_type": qtype.name,
            }
            generated.append(item)
            questions_per_type[qtype] += 1
            product_counts[product_id] += 1
            lang_generated += 1
            logger.info(f"  Generated {qtype.name} question in {lang}")

        logger.info(f"  {lang} total generated: {lang_generated}")
        logger.info(f"  By type: {', '.join(f'{qt.name}={c}' for qt, c in questions_per_type.items())}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("QUERY GENERATION COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Total generated: {len(generated)}")
    logger.info(f"Reviews processed: {total_processed}")
    logger.info(f"Skipped (low quality): {skipped_low_quality}")
    logger.info(f"Skipped (product quota): {skipped_product_quota}")
    logger.info(f"Skipped (SKIP token/empty): {skipped_skip_token}")
    logger.info(f"Skipped (parse errors): {skipped_parse}")
    logger.info("By language:")
    for lang in SUPPORTED_LANGUAGES:
        count = sum(1 for q in generated if q["language"] == lang)
        logger.info(f"  {lang}: {count}")
    logger.info("=" * 60)
    return generated


# ───────────────────────── CLI ───────────────────────── #
def main():
    ap = argparse.ArgumentParser(
        description="Generate diverse multilingual benchmark queries from Amazon Reviews Multi."
    )
    ap.add_argument("--num_questions_per_lang", type=int, default=100,
                     help="Number of questions per language (default: 100, total = 6 * N)")
    ap.add_argument("--output_file", default="benchmark_queries_multilingual.json")
    ap.add_argument("--min_review_length", type=int, default=100,
                     help="Minimum review_body length in characters")
    ap.add_argument("--llm_model", default="anthropic/claude-3.5-sonnet")
    ap.add_argument("--product_max_per_product", type=int, default=2,
                     help="Max number of questions per product_id")
    args = ap.parse_args()

    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY is not set; set it in your environment.")

    # Load dataset
    df = load_dataset_csv("train.csv")

    # Initialize LLM (OpenRouter)
    logger.info(f"Initializing LLM: {args.llm_model}")
    llm = ChatOpenAI(
        model=args.llm_model,
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0.25,
        timeout=60,
    )

    # Generate queries
    queries = generate_multilingual_queries(
        llm=llm,
        df=df,
        num_questions_per_lang=args.num_questions_per_lang,
        min_review_length=args.min_review_length,
        product_max_per_product=args.product_max_per_product,
    )

    # Save
    if queries:
        logger.info(f"Saving {len(queries)} questions to '{args.output_file}'.")
        with open(args.output_file, "w", encoding="utf-8") as f_out:
            json.dump(queries, f_out, indent=2, ensure_ascii=False)
        logger.info("Benchmark file saved successfully")

        # Print samples
        logger.info("\n" + "=" * 60)
        logger.info("SAMPLE QUESTIONS (first 2 per language)")
        logger.info("=" * 60)
        for lang in SUPPORTED_LANGUAGES:
            lang_qs = [q for q in queries if q["language"] == lang][:2]
            for item in lang_qs:
                logger.info(f"  [{item['language']}] Q: {item['question']}")
                logger.info(f"       A: {item['answer']}")


if __name__ == "__main__":
    main()
