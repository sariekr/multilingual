# Multilingual Embedding Models Benchmark

## Proje Amaci
Amazon Reviews Multi dataseti (DE, EN, ES, FR, JA, ZH) uzerinden multilingual embedding modellerinin **monolingual**, **cross-lingual** ve **true cross-lingual** retrieval performansini olcmek. Kardes proje: `../open_source_embedding_models_gpu` (sadece EN, Health_and_Personal_Care).

## Dataset
- **Kaynak:** Kaggle `mexwell/amazon-reviews-multi` — kagglehub ile indirilir, cache: `~/.cache/kagglehub/`
- **Dosyalar:** train.csv (1.2M row, 200K/dil), test.csv, validation.csv
- **Kolonlar:** `review_id, product_id, reviewer_id, stars, review_body, review_title, language, product_category`
- **Onemli fark:** Eski projede `parent_asin` + `product_title` + `helpful_votes` vardi, bu datasette **yok**. Burada `product_id` ve `product_category` kullaniliyor.
- Lokal `data/` klasoru yok — dataset dogrudan kagglehub cache'inden okunur.
- Her dilde eligible review sayisi: DE/EN/ES/FR ~120-145K, ZH ~17K (daha kucuk).

## Dosya Yapisi

```
rag_loader_multilingual.py                    # Veri yukleme + embedding + Qdrant indexleme
generate_multilingual_queries.py              # 6 dilde soru-cevap uretimi (LLM ile)
translate_queries_crosslingual.py             # Sorulari diger dillere cevir (true cross-lingual icin)
evaluate_multilingual.py                      # Monolingual + cross-lingual + true cross-lingual degerlendirme
requirements.txt                              # pip dependency'leri
.env                                          # OPENROUTER_API_KEY, QDRANT ayarlari (gitignored)
benchmark_queries_multilingual.json           # Uretilmis sorular — 600 query (gitignored)
benchmark_queries_multilingual_filtered.json  # Filtered sorular — 119 query, sadece Qdrant'taki product_id'ler (gitignored)
benchmark_queries_crosslingual.json           # Cevirilmis sorular — 2987 query (gitignored)
benchmark_queries_crosslingual_filtered.json  # Filtered cevirilmis sorular — 593 query (gitignored)
evaluation_results/                           # Eval ciktilari (gitignored)
venv/                                         # Python virtual environment
```

## Pipeline Calistirma Sirasi

```bash
# 1. Soru uretimi (LLM gerekli — OpenRouter, bir kez yapilir)
venv/bin/python3 generate_multilingual_queries.py \
  --num_questions_per_lang 100 \
  --llm_model google/gemini-3-flash-preview \
  --min_review_length 100

# 2. Veri yukleme + embedding + Qdrant'a indexleme (her model icin tekrarlanir)
venv/bin/python3 rag_loader_multilingual.py \
  --model e5_small \
  --max_reviews_per_lang 20000

# 3. Degerlendirme — monolingual + cross-lingual (her model icin tekrarlanir)
venv/bin/python3 evaluate_multilingual.py \
  --model e5_small \
  --queries_file benchmark_queries_multilingual.json \
  --mode both \
  --top_k 5 \
  --output_dir evaluation_results

# 4. True cross-lingual ceviri (bir kez, ~5 dk, LLM gerekli)
venv/bin/python3 translate_queries_crosslingual.py \
  --input_file benchmark_queries_multilingual.json \
  --output_file benchmark_queries_crosslingual.json \
  --llm_model google/gemini-3-flash-preview \
  --target_langs all

# 5. True cross-lingual degerlendirme (her model icin tekrarlanir)
venv/bin/python3 evaluate_multilingual.py \
  --model e5_small \
  --queries_file benchmark_queries_crosslingual.json \
  --mode true_crosslingual \
  --top_k 5 \
  --output_dir evaluation_results
```

Adim 2-3 ve 5, her model icin tekrarlanir. Adim 1 ve 4 bir kez yapilir (sorular modelden bagimsiz).

**ONEMLI: Product ID Filtreleme**

Soru uretimi `train.csv`'nin tamaminda (~200K/dil) yapilir ama Qdrant'a sadece `max_reviews_per_lang` kadar review indexlenir (default 20K/dil). Bu durumda soruların buyuk kisminın cevabi Qdrant'ta olmaz. Cozum: eval oncesi filtreleme yapilmali.

```bash
# Filtreleme ornegi (Python ile):
# 1. Qdrant'taki product_id'leri cek
# 2. benchmark_queries_multilingual.json'dan sadece Qdrant'ta olan product_id'leri filtrele
# 3. benchmark_queries_multilingual_filtered.json olarak kaydet
# 4. Ayni filtrelemeyi crosslingual queries icin de yap
# 5. Eval'i filtered dosyalarla calistir
```

## Modeller (MODEL_CONFIGS)

| Key | HuggingFace Model | Dim | Batch | Notlar |
|-----|-------------------|-----|-------|--------|
| `e5_small` | intfloat/multilingual-e5-small | 384 | 32 | En iyi sonuc, query:/passage: prefix |
| `e5_base` | intfloat/multilingual-e5-base | 768 | 16 | query:/passage: prefix |
| `mpnet_multilingual` | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | 768 | 16 | |
| `minilm_multilingual` | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 384 | 32 | Dusuk accuracy, diller arasi karisma yuksek |
| `minilm_v2` | sentence-transformers/all-MiniLM-L6-v2 | 384 | 64 | Sadece EN, negative baseline |
| `qwen3_emb_06b` | Qwen/Qwen3-Embedding-0.6B | 1024 | 8 | query_prompt_name="query", MPS'te cok yavas (~10 r/s) |
| `e5_large_instruct` | intfloat/multilingual-e5-large-instruct | 1024 | 8 | query:/passage: prefix |
| `bge_m3` | BAAI/bge-m3 | 1024 | 8 | MPS'te yavas (~22 r/s) |
| `gte_multilingual_base` | Alibaba-NLP/gte-multilingual-base | 768 | 16 | trust_remote_code |
| `jina_v3` | jinaai/jina-embeddings-v3 | 1024 | 8 | trust_remote_code |
| `nomic_embed_v1_5` | nomic-ai/nomic-embed-text-v1.5 | 768 | 16 | search_query/search_document prefix |
| `e5_mistral_7b` | intfloat/e5-mistral-7b-instruct | 4096 | 4 | INT8/CUDA onerilen, 16GB RAM'de cok buyuk |
| `gte_qwen2_7b` | Alibaba-NLP/gte-Qwen2-7B-instruct | 3584 | 4 | INT8/CUDA onerilen, 16GB RAM'de cok buyuk |
| `llama_embed_nemotron_8b` | nvidia/llama-embed-nemotron-8b | 4096 | 16 | CUDA gerekli, BF16, flash_attention_2/sdpa, native encode_query/encode_document |

### Embedding Hizlari (Apple M4, 16GB RAM, MPS/CPU)

| Model | reviews/s | 120K sure | Notlar |
|-------|-----------|-----------|--------|
| `minilm_v2` | ~199 | ~10 dk | En hizli (kucuk EN-only model) |
| `e5_small` | ~121 | ~16.5 dk | Iyi denge (hiz + accuracy) |
| `minilm_multilingual` | ~117 | ~17 dk | Hizli ama dusuk accuracy |
| `bge_m3` | ~22 | ~90 dk | Cok yavas, 1024-dim |
| `qwen3_emb_06b` | ~10 | ~3.3 saat | En yavas, 0.6B param LLM-based |

## Mimari Kararlar

### Qdrant
- **Tek collection per model:** `multilingual_{model_name}`
- Her review'in payload'unda `language` fieldi var, uzerinde keyword index olusturuluyor.
- Monolingual eval: `FieldCondition(key="language", match=MatchValue(value=lang))` filtresi ile.
- Cross-lingual eval: filtre yok, tum dillerde arama.
- Qdrant local: `http://localhost:6333` (docker veya binary ile baslatilmali).
- Collection olusturmada delete/create arasinda `time.sleep(2)` var (race condition fix).

### Soru Uretimi
- Her dildeki soru ve cevap, o dilde **native** olarak uretilir (ceviri degil).
- 5 soru tipi dengeli dagitilir: FACTUAL, OPINION, USAGE, PROBLEM_SOLVING, FEATURE.
- JA/ZH icin cevap limiti: 60 karakter. Diger diller: 30 kelime.
- product_id basina max 2 soru (cok tekrardan kacinmak icin).
- LLM: OpenRouter uzerinden. Test icin `google/gemini-3-flash-preview` kullanildi (600 soru, sadece 3 SKIP).
- Mevcut soru seti: 600 soru (100/dil), `benchmark_queries_multilingual.json`.

### Evaluation
- **Ground truth matching:** Once `product_id` exact match, bulamazsa cosine similarity fallback (threshold=0.7).
- **Metrikler:** Top-1/3/5 accuracy, avg/p95/p99 latency, throughput QPS.
- **Cross-lingual ek metrikler:** Language gap (EN acc - avg others), 6x6 retrieval language distribution matrix, mono-vs-cross gap.
- **Cikti dosyalari:** `results_{model}_{mode}.json`, `metrics_{model}.json`, `summary_{model}.json`

### OpenSourceEmbeddings Sinifi
- `rag_loader_multilingual.py` icinde tanimli. `evaluate_multilingual.py` bunu import eder.
- Device detection: CUDA > MPS > CPU.
- INT8 quantization (CUDA only, bitsandbytes), FP16/BF16 fallback.
- Bazi modellere hotfix uygulanir (Qwen2: `use_cache=False`).
- `embed_documents()` doc_prefix ekler, `embed_query()` query_prefix veya `prompt_name` ekler.
- `query_prompt_name` config fieldi: sentence-transformers'in built-in prompt sistemini kullanir (Qwen3 icin).

## Env Degiskenleri (.env)
```
QDRANT_LOCAL_URL=http://localhost:6333
QDRANT_CLOUD_URL=...       # opsiyonel, varsa local yerine kullanilir
QDRANT_API_KEY=...          # cloud icin gerekli
OPENROUTER_API_KEY=...      # soru uretimi icin gerekli
```

## Benchmark Sonuclari

### Test Ortami
- Apple M4 Mac, 16GB RAM
- Qdrant local (http://localhost:6333)
- 120K review (20K/dil, train.csv'den sample)
- Top-K = 5
- LLM: Gemini Flash (soru uretimi ve ceviri icin)

### Product ID Coverage Sorunu ve Filtreleme

**KRITIK BULGU:** Soru uretimi `train.csv`'nin tamaminda (~200K/dil) yapildi, ancak Qdrant'a sadece 20K/dil indexlendi. Sonuc: 600 sorunun yalnizca **119'unun (%19.8)** `product_id`'si Qdrant'ta mevcut.

**Dil bazinda coverage:**

| Dil | Toplam Soru | Qdrant'ta Bulunan | Coverage |
|-----|-------------|-------------------|----------|
| DE | 100 | 26 | %26 |
| EN | 100 | 12 | %12 |
| ES | 100 | 20 | %20 |
| FR | 100 | 17 | %17 |
| JA | 100 | 20 | %20 |
| ZH | 100 | 24 | %24 |
| **Toplam** | **600** | **119** | **%19.8** |

**Etki:** Unfiltered testlerde accuracy yapay olarak dusuk cikiyor cunku soruların %80'inin cevabi Qdrant'ta yok. Cosine similarity fallback bu durumu maskeliyor (e5_small'da Top-5=%100 gosteriyor).

**Cozum:** Sadece `product_id`'si Qdrant'ta bulunan sorular filtrelendi:
- `benchmark_queries_multilingual_filtered.json` — 119 monolingual query
- `benchmark_queries_crosslingual_filtered.json` — 593 true cross-lingual query

Asagida hem unfiltered (tum 600/2987 query) hem filtered (119/593 query) sonuclari raporlanmistir. **Filtered sonuclar daha guvenilirdir** cunku her sorunun ground truth'u Qdrant'ta mevcuttur.

---

### Sonuc Ozeti (Filtered — Onerilen Referans Degerler)

| Mod | Metrik | e5_small | minilm_multilingual |
|-----|--------|----------|---------------------|
| **Monolingual** | Top-1 / Top-3 / Top-5 | **35.2% / 66.8% / 100%** | 8.5% / 16.7% / 18.0% |
| **Cross-lingual** | Top-1 / Top-3 / Top-5 | **35.3% / 67.2% / 100%** | 6.7% / 12.6% / 17.7% |
| **True Cross-lingual** | Top-1 / Top-3 / Top-5 | 0% / 0.3% / 0.5% | **1.7% / 3.9% / 5.4%** |

**Ana Bulgu — Dil Ayirimi Paradoksu:**
- **e5_small:** Monolingual'de en iyi (%35 Top-1) ama true cross-lingual'de **%0** — guclu dil ayirimi cross-language retrieval'i tamamen engelliyor.
- **minilm_multilingual:** Monolingual'de zayif (%8.5 Top-1) ama true cross-lingual'de **%1.7** — zayif dil ayirimi sayesinde dillerarasi retrieval mumkun oluyor.

Bu paradoks, monolingual ve cross-lingual kullanim senaryolari icin **farkli modellerin optimal oldugunu** gostermektedir.

---

### A. Monolingual Retrieval (per language)

#### Unfiltered (600 query — sorunun %80'inin cevabi Qdrant'ta yok)

| Dil | e5_small | minilm_multilingual | minilm_v2 (EN-only) |
|-----|----------|---------------------|---------------------|
| DE | 16% / 59% / 100% | 8% / 13% / 17% | 1% / 4% / 4% |
| EN | 30% / 66% / 100% | 5% / 7% / 8% | 4% / 5% / 6% |
| ES | 25% / 65% / 100% | 5% / 6% / 9% | 2% / 6% / 6% |
| FR | 21% / 63% / 100% | 3% / 4% / 5% | 6% / 10% / 12% |
| JA | 24% / 55% / 100% | 3% / 4% / 6% | 0% / 0% / 0% |
| ZH | 24% / 64% / 100% | 4% / 14% / 17% | 3% / 4% / 7% |
| **AVG** | **23.3% / 62% / 100%** | **4.7% / 8% / 10.3%** | **2.7% / 4.8% / 5.8%** |

#### Filtered (119 query — tum sorularin cevabi Qdrant'ta mevcut)

| Dil | e5_small (Top-1/3/5) | minilm_multilingual (Top-1/3/5) |
|-----|----------------------|--------------------------------|
| **AVG** | **35.2% / 66.8% / 100%** | **8.5% / 16.7% / 18.0%** |

**Not:** Filtered sonuclarda e5_small'in Top-1'i %23→%35'e yukseldi (gercek accuracy). minilm_multilingual'da %4.7→%8.5.

---

### B. Cross-Lingual Retrieval (filtre yok, ayni dilde query)

#### Unfiltered (600 query)

| Metrik | e5_small | minilm_multilingual | minilm_v2 |
|--------|----------|---------------------|-----------|
| Top-1 | 22.0% | 3.2% | 2.5% |
| Top-3 | 62.2% | 7.7% | 4.8% |
| Top-5 | 100% | 11.0% | 6.0% |
| Language gap | -0.012 | +0.010 | +0.018 |
| Mono vs Cross gap | +0.013 | +0.015 | +0.002 |

#### Filtered (119 query)

| Metrik | e5_small | minilm_multilingual |
|--------|----------|---------------------|
| Top-1 | 35.3% | 6.7% |
| Top-3 | 67.2% | 12.6% |
| Top-5 | 100% | 17.7% |

---

### C. True Cross-Lingual Retrieval (cevirilmis soru → farkli dildeki review)

**Bu benchmark'in ana sonucu.** Her soruyu 5 dile cevirip, orijinal dildeki review'i bulabilme yetenegini olcer.

#### Unfiltered (2987 query)

| Metrik | e5_small | minilm_multilingual |
|--------|----------|---------------------|
| Top-1 | 0.0% | 0.33% |
| Top-3 | 0.07% | 0.77% |
| Top-5 | 0.1% | 1.07% |

#### Filtered (593 query)

| Metrik | e5_small | minilm_multilingual |
|--------|----------|---------------------|
| Top-1 | **0.0%** | **1.69%** |
| Top-3 | 0.34% | 3.88% |
| Top-5 | 0.51% | 5.40% |

#### minilm_multilingual — En Iyi Dil Ciftleri (Filtered, Top-1)

| Query Lang → Target Lang | Top-1 Accuracy |
|--------------------------|----------------|
| en → es | **10.0%** |
| fr → zh | 8.3% |
| ja → en | 8.3% |
| ja → de | ~1.0% |
| ja → zh | ~1.0% |
| de → zh | ~1.0% |

**e5_small icin tum dil ciftleri %0 Top-1.** Model dilleri o kadar iyi ayiriyor ki, Japonca soruyla Almanca review'i bulamıyor — her zaman Japonca review'leri donduruyor.

#### True Cross-Lingual Analiz

**e5_small (guclu dil ayirimi):**
- Monolingual accuracy yuksek (%35), cunku model soru dilindeki review'leri hassas sekilde buluyor.
- True cross-lingual'de **tamamen basarisiz** — cevirilmis soru (ornegin JA), soru dilindeki review'leri (JA) donduruyor, hedef dildeki (DE) review'i bulamiyor.
- Ornek: `q_de_001_to_en` ("Why are the stones being sent back?") → Top-5 sonucun tamami EN review'ler, DE kaynak review bulunamiyor.
- **Sonuc:** e5_small dillerarasi transfer ogrenememis, her dili ayri bir semantic space olarak kodluyor.

**minilm_multilingual (zayif dil ayirimi):**
- Monolingual accuracy dusuk (%8.5), cunku diller arasi karisma (cross-language leakage) dogru review'i bulmayı zorlastiriyor.
- True cross-lingual'de **kismen basarili** (%1.7 Top-1) — zayif dil ayirimi sayesinde cevirilmis soru bazen orijinal dildeki review'i bulabiliyor.
- En iyi ciftler: en→es (%10), fr→zh (%8.3), ja→en (%8.3) — bunlar genellikle yakin semantic space'i paylasan diller.
- **Sonuc:** minilm_multilingual daha karisik ama dillerarasi transfer kismi var.

---

### Performans

| Metrik | e5_small | minilm_multilingual | minilm_v2 |
|--------|----------|---------------------|-----------|
| Avg latency | 17.1ms | 16.6ms | 16.7ms |
| P95 latency | 20.4ms | 20.8ms | 20.8ms |
| Throughput | 58.5 QPS | 60.1 QPS | 59.8 QPS |
| Eval sure | 96.7s | 85.9s | 80.0s |

### Retrieval Language Distribution (Cross-lingual, top 5 sonuc x 100 query/dil = 500/dil)

**e5_small** — Dil ayirimi cok iyi:
- DE/ES/JA/ZH sorulari: %100 kendi dilinden
- EN sorulari: %77 EN, %16 JA, %3 ES/FR, %2 DE
- FR sorulari: %91 FR, %6 JA, %2 ES, %1 DE/EN

**minilm_multilingual** — Diller arasi cok karisik (weak separation):
- DE sorulari: %39 DE, %18 FR, %15 JA, %11 ES, %11 ZH, %6 EN
- EN sorulari: %28 EN, %21 FR, %17 DE, %15 ES, %13 JA, %6 ZH
- Tum dillerde onemli cross-language leakage

**minilm_v2** (EN-only) — Latin dilleri icinde kaliyor, CJK karistiriyor:
- JA sorulari: %100 JA (ama accuracy %0 — yanlis dokulari donduruyor)
- ZH sorulari: %91 ZH, %9 JA (CJK karisimi)
- EN/ES/FR/DE: buyuk oranda kendi dilinde

## Analiz ve Bulgular

### Dil Ayirimi Paradoksu (Ana Bulgu)

Bu benchmark'in en onemli bulgusu: **Guclu dil ayirimi monolingual retrieval icin iyi ama cross-lingual retrieval icin zarali.**

| Ozellik | e5_small | minilm_multilingual |
|---------|----------|---------------------|
| Dil ayirimi | Cok guclu (per-language clustering) | Zayif (cross-language mixing) |
| Monolingual Top-1 | **%35.2** (en iyi) | %8.5 |
| True Cross-lingual Top-1 | **%0** (en kotu) | %1.7 |
| Davranis | Her dili ayri semantic space — JA soru sadece JA review dondurur | Diller karisik — JA soru bazen DE review dondurur |

**Yorum:** e5_small'in `query:`/`passage:` prefix mekanizmasi ve retrieval-focused egitimi, her dili kendi embedding alt-uzayina koyuyor. Bu monolingual arama icin ideal — soru dilindeki review'leri hassas buluyor. Ancak "Japonca soruyla Almanca review bul" senaryosunda tamamen basarisiz cunku Japonca embeddingler Almanca embedding uzayindan cok uzakta.

minilm_multilingual'in zayif dil ayirimi ise tek bir karisik embedding uzayinda tum dilleri yakin tutuyor. Bu monolingual'de noise yaratsa da (%39 DE sonuclari diger dillerden geliyor), cross-lingual senaryoda avantaj saglıyor.

**Pratik Sonuc:**
- **Monolingual RAG sistemi:** e5_small (veya e5 ailesi) kullanin. Dil filtresi ile %35+ accuracy.
- **Cross-lingual RAG sistemi (ornegin "Japonca soru, tum dillerde cevap ara"):** minilm_multilingual daha iyi baslangiç noktasi, ama %1.7 Top-1 hala yetersiz. Daha guclu modeller (e5-large-instruct, bge-m3, jina-v3) denenmeli.

### e5_small Neden Monolingual'de En Iyi?
- `intfloat/multilingual-e5-small` asimetrik query/passage prefix kullanir (`query: ` / `passage: `).
- Bu prefix mekanizmasi retrieval icin ozel optimize edilmis — diger iki model genel amacli sentence similarity icin egitilmis.
- 384-dim ile kucuk ama multilingual retrieval'a odaklanmis egitim veri seti.

### Top-5 = %100 Sorunu (e5_small)
- e5_small'da Top-5 her dilde %100 — bu supheliydi.
- Sebebi: cosine similarity fallback threshold'u (0.7) muhtemelen cok dusuk.
- e5_small'in iyi embeddingleri sayesinde cogu sonuc 0.7 uzerinde cikiyor ve "match" sayiliyor.
- minilm_multilingual'da Top-5 = %18, bu daha gercekci — o modelin embeddingleri daha zayif.
- **Cozum onerisi:** Threshold'u yukseltin (0.85+) veya sadece product_id exact match ile degerlendirin.
- **Not:** True cross-lingual eval'de cosine similarity fallback zaten kullanilmiyor (sadece product_id match).

### minilm_multilingual Neden Monolingual'de Kotu?
- `paraphrase-multilingual-MiniLM-L12-v2` paraphrase detection ve sentence similarity icin egitilmis.
- Retrieval (question -> passage matching) icin optimize edilmemis.
- Diller arasi embedding space'i cok yakin — tum diller birbiriyle karisik, dil ayirimi zayif.
- Ancak bu zayiflik cross-lingual retrieval'da avantaja donusuyor (bkz. Dil Ayirimi Paradoksu).

### minilm_v2 (EN-only) Negative Baseline
- Beklenildigi gibi, JA'da %0, diger non-EN dillerde %1-6.
- EN'de bile sadece %4 Top-1 (e5_small: %30).
- Multilingual modellerin neden gerekli oldugunu kanitliyor.

### Denenmis Ama Tamamlanamamis Modeller
- **bge_m3:** MPS'te ~22 r/s, 120K review icin ~90 dk. Durduruldu (cok yavas).
- **qwen3_emb_06b:** MPS'te ~10 r/s, 120K review icin ~3.3 saat. Durduruldu (cok yavas).
- Bu modeller GPU (CUDA) olan bir makinede test edilmeli.

## Metodoloji (Detayli)

### 1. Arastirma Sorusu

Bu benchmark su ana soruyu cevaplamaya calisir: **Farkli multilingual embedding modelleri, dil-icin (monolingual) ve diller-arasi (cross-lingual) ortamlarda kullanici sorgularina ne kadar dogru dokuman dondurur?**

Alt sorular:
- Asimetrik retrieval icin egitilmis modeller (e5 ailesi) vs genel amacli sentence similarity modelleri (MiniLM, MPNet) arasinda ne kadar fark vardir?
- Modelin dil ayirma yetenegi (language separation) retrieval kalitesini nasil etkiler?
- Monolingual modeller (EN-only) multilingual ortamda negative baseline olarak ne kadar kotu performans gosterir?

### 2. Dataset

**Kaynak:** Amazon Reviews Multi (Kaggle: `mexwell/amazon-reviews-multi`)

| Ozellik | Deger |
|---------|-------|
| Toplam satir | ~1.2M (train.csv) |
| Diller | DE, EN, ES, FR, JA, ZH (6 dil) |
| Her dilde satir | ~200K (train), ancak min_length filtresi sonrasi eligible: DE/EN/ES/FR ~120-145K, ZH ~17K |
| Kolonlar | review_id, product_id, reviewer_id, stars, review_body, review_title, language, product_category |

**Corpus Olusturma:** Her dilden `max_reviews_per_lang` adet review rastgele sample'lanir (benchmark'ta 20K/dil = 120K toplam). Review'ler `review_body` uzunluguna gore filtrelenmez (tum review'ler dahil). Her review su formatta indekslenir:

```
Review Title: {review_title}
Review: {review_body}
```

Payload: `product_id`, `product_category`, `language`, `review_rating`, `review_title`, `review_body`.

### 3. Soru Uretimi (Query Generation)

**Amac:** Her dil icin, o dildeki review'lerden native (anadil) soru-cevap ciftleri uretmek.

**Surec:**
1. `train.csv`'den her dil icin `review_body` uzunlugu >= 100 karakter olan review'ler filtrelenir.
2. LLM'e (OpenRouter uzerinden) her review icin soru-cevap cifti urettirilir.
3. Soru ve cevap, review'in kendi dilinde native olarak uretilir — ceviri yapilmaz.
4. 5 soru tipi dengeli dagitilir: FACTUAL, OPINION, USAGE, PROBLEM_SOLVING, FEATURE.

**Prompt Tasarimi:**
- Her soru tipi icin ayri prompt template'i var.
- Prompt'a acik talimat eklenir: `"You MUST write BOTH the question AND the answer in {language_name}. Do NOT translate - write natively in {language_name}."`
- Guardrails: cevap max 30 kelime (alphabetic diller) veya 60 karakter (JA/ZH). SKIP token ile kalitesiz review'ler atlanir.
- `product_id` basina max 2 soru (tek urune asiri yogunlasmamak icin).

**LLM:** Gemini Flash (`google/gemini-3-flash-preview`), temperature=0.25, OpenRouter API uzerinden.

**Cikti:** 600 soru (100/dil). Her soru su bilgileri icerir:
- `question`: Sorunun kendisi (review dilinde)
- `answer`: Cevap (review dilinde, <= 30 kelime / 60 karakter)
- `context`: Kaynak review_body (sorunun uretildigi tam metin)
- `product_id`: Review'in urun kimlik numarasi
- `language`: Review'in dili (= sorunun dili)
- `question_type`: FACTUAL / OPINION / USAGE / PROBLEM_SOLVING / FEATURE

**Kritik Nokta:** Her soru, yalnizca **kendi dilindeki** bir review'den uretilir. Almanca bir soru, bir Almanca review'den uretilmistir. Japonca bir soru, bir Japonca review'den uretilmistir. Diller arasi soru uretimi (ornegin bir Almanca review'den Japonca soru uretimi) **yapilmamistir**.

### 4. Evaluation Metodolojisi

#### 4.1 Monolingual Retrieval (Dil-Icin Arama)

**Nedir:** Bir dildeki soru, yalnizca ayni dildeki dokumanlar arasinda aranir.

**Surec:**
1. Soru embedding'e cevrilir (modelin `embed_query()` fonksiyonu ile).
2. Qdrant'ta `language` field'i uzerinden filtre uygulanir: `FieldCondition(key="language", match=MatchValue(value=query_lang))`.
3. Sadece ayni dildeki review'ler arasinda Top-K en yakin vektorler dondurulur.
4. Ground truth eslesmesi kontrol edilir (detay asagida).

**Olctugu sey:** Modelin, **tek bir dil icinde** soru-dokuman eslemesini ne kadar iyi yaptigini olcer. Bu, "klasik" RAG retrieval performansidir ama her dil icin ayri ayri olculur.

#### 4.2 Cross-Lingual Retrieval (Diller-Arasi Arama)

**Nedir:** Bir dildeki soru, **tum dillerdeki** dokumanlar arasinda aranir (dil filtresi yok).

**Surec:**
1. Soru embedding'e cevrilir.
2. Qdrant'ta hicbir filtre uygulanmaz — 120K review'in tamami arama uzayinda.
3. Top-K en yakin vektorler dondurulur (herhangi bir dilden gelebilir).
4. Ground truth eslesmesi kontrol edilir.
5. Ek olarak, dondurulen sonuclarin dil dagilimi kaydedilir (retrieval matrix).

**Olctugu sey (ve olcmedigi sey):**

Bu benchmark'taki cross-lingual degerlendirme, **"farkli bir dilde soru sorarak baska bir dildeki cevabi bulma"** yetenegini **dogrudan olcmez**. Bunun sebebi:

- Sorular yalnizca kendi dillerindeki review'lerden uretilmistir (Almanca soru → Almanca review).
- Ground truth `product_id` ayni dildeki kaynak review'e aittir.
- Dolayisiyla, cross-lingual modda basarili bir eslesme icin modelin yine **ayni dildeki kaynak review'i** bulması gerekir — ama bu sefer 6 dilden toplam 120K review arasinden.

Cross-lingual degerlendirme aslinda su iki seyi olcer:

1. **Gurultu direnci (noise resistance):** Model, diger dillerdeki irrelevant dokumanlar tarafindan "distract" oluyor mu? Arama uzayi 6x buyudugunde accuracy ne kadar dusuyor?
2. **Dil ayirma yetenegi (language separation):** Model, soru dilinde yazilmis dokumanlari diger dillerden ayirabiliyor mu? Retrieval matrix bunu gosterir.

**Onemli istisna:** Amazon Reviews Multi datasetinde ayni `product_id`'ye sahip review'ler birden fazla dilde bulunabilir. Ornegin, urun X hem Almanca hem Ingilizce review'e sahip olabilir. Bu durumda, Almanca bir soruya Ingilizce review'in dondurulmesi de `product_id` match ile basarili sayilir. Ancak bu "dogal cross-lingual hit", test edilmek istenen bir hipotez degil, tesadufi bir yan etkidir.

**Neden gercek cross-lingual test icin ceviri yaklasimi kullanildi:**
- Dataset'te paralel ceviriler yok — ayni urune ait farkli dillerdeki review'ler farkli icerikler tasir.
- 963K product_id'nin hicbiri birden fazla dilde review'e sahip degil (product_id'ler `product_de_0372235` seklinde dil-prefixed).
- Cozum: Mevcut 600 soruyu diger 5 dile LLM ile cevirip `benchmark_queries_crosslingual.json` olusturuldu (bkz. Adim 4.3).

#### 4.3 True Cross-Lingual Retrieval (Gercek Diller-Arasi Arama)

**Nedir:** Bir dilde yazilmis sorunun **baska bir dildeki** review'i bulabilme yetenegini olcer. Ornegin "Japonca soru ile Almanca review bulabilir mi?"

**Surec:**
1. 600 mevcut sorunun her biri, `translate_queries_crosslingual.py` ile diger 5 dile LLM (Gemini Flash) ile cevrilir → 3000 translated query.
2. Her cevirilmis soru, Qdrant'ta **hicbir filtre olmadan** (tum 120K review'de) aranir.
3. Ground truth: Orijinal dildeki review'in `product_id`'si ile exact match aranir.
4. **Cosine similarity fallback KULLANILMAZ** — cevap orijinal dilde (ornegin DE), soru farkli dilde (ornegin JA), bu yuzden embedding karsilastirmasi dillerarasi anlamsiz.

**Translated Query Formati:**
```json
{
  "id": "q_de_001_to_ja",
  "original_id": "q_de_001",
  "question": "(Japonca cevirilmis soru)",
  "answer": "Wegen mangelhafter Qualitat...",
  "context": "(orijinal DE review text)",
  "product_id": "product_de_0372235",
  "product_category": "drugstore",
  "language": "de",
  "query_language": "ja",
  "question_type": "FACTUAL"
}
```

Onemli field'lar:
- `language`: review'in orijinal dili (ground truth'un bulunacagi dil) — degismez
- `query_language`: sorunun cevrildigi dil (arama yapilacak dil) — yeni field
- `question`: cevirilmis soru
- `answer`: orijinal dildeki cevap (degismez, ground truth)

**Metrikler:**

1. **6x6 Language Pair Matrix:** Her (query_lang, target_lang) cifti icin Top-1/3/5 accuracy.
   - Diyagonal = monolingual (zaten mevcut testlerle olculuyor)
   - Off-diagonal = gercek cross-lingual accuracy

2. **Per query-language avg:** "JA ile soru soruldugunda ortalama accuracy" (tum hedef dillerin ortalamasi)

3. **Per target-language avg:** "DE review'leri ne kadar kolay bulunuyor" (tum kaynak dillerin ortalamasi)

4. **Best/worst language pairs:** En iyi ve en kotu calisan dil ciftleri

**Olctugu sey:** Modelin dillerarasi semantik anlama ve eslesme yetenegini **dogrudan** olcer. "Bu modele Japonca soru sorarsam, Almanca review'deki cevabi bulabilir mi?" sorusuna cevap verir.

#### 4.4 Ground Truth Eslesmesi

Iki asamali eslesme yapilir:

**Asama 1 — product_id exact match (birincil):**
- Dondurulen Top-K sonucun payload'undaki `product_id`, sorunun `product_id`'si ile karsilastirilir.
- Eslesen ilk sonucun rank'i kaydedilir (1-indexed).
- Eslesme bulunursa, Top-K hit = True.

**Asama 2 — Cosine similarity fallback:**
- Eger product_id match bulunamazsa, semantik benzerlik kontrolu yapilir.
- Sorunun cevabi (`answer`) embedding'e cevrilir.
- Her sonuc context'i de embedding'e cevrilir.
- En yuksek cosine similarity > threshold (default: 0.7) ise, hit sayilir.

**Fallback'in Bilinen Sorunu:** e5_small gibi iyi modellerde, embedding kalitesi yuksek oldugu icin cogu sonuc-cevap cifti 0.7 uzerinde cosine similarity alir — bu da Top-5'i yapay olarak %100'e cikartir. minilm_multilingual gibi zayif modellerde bu etki gorulmez cunku embedding kalitesi dusuk. Threshold'un 0.85+'e yukseltilmesi veya tamamen kaldirilmasi (sadece product_id match) onerilen bir iyilestirmedir.

#### 4.5 Metrikler

**Per-Language (monolingual icin):**
- **Top-1 Accuracy:** Ilk sonucun ground truth ile esleme orani
- **Top-3 Accuracy:** Ilk 3 sonuctan birinin esleme orani
- **Top-5 Accuracy:** Ilk 5 sonuctan birinin esleme orani
- **Avg Latency (ms):** Ortalama arama suresi (embedding + Qdrant search)
- **P95/P99 Latency:** Yuzdelik dilim gecikmeleri

**Cross-Lingual Ek Metrikler:**
- **Language Gap:** `EN_Top1 - ortalama(diger_5_dil_Top1)` — Ingilizce'nin diger dillere gore avantaji
- **Retrieval Matrix:** 6x6 matris, her query_lang icin dondurulen sonuclarin dil dagilimi (top-5 x 100 query/dil = 500 sonuc/dil)
- **Mono vs Cross Gap:** `monolingual_avg_Top1 - crosslingual_avg_Top1` — arama uzayinin buyumesinin etkisi

**True Cross-Lingual Ek Metrikler:**
- **6x6 Language Pair Matrix (Top-1/3/5):** Her (query_lang → target_lang) cifti icin accuracy. Diyagonal = monolingual, off-diagonal = gercek cross-lingual.
- **Per Query Language Avg:** Bir dilde soru soruldugunda tum hedef dillerdeki ortalama accuracy
- **Per Target Language Avg:** Bir dildeki review'lerin tum kaynak dillerden ne kadar kolay bulundugu
- **Best/Worst Pairs:** En iyi ve en kotu calisan dil ciftleri (Top-1)
- **Not:** Cosine similarity fallback kullanilmaz, sadece product_id exact match

**Genel:**
- **Throughput (QPS):** Saniyedeki sorgu sayisi
- **Eval Sure:** Tum degerlendirmenin toplam suresi

### 5. Embedding Sureci

**Model yukleme:** HuggingFace `sentence-transformers` kutuphanesi (v5.2.2) uzerinden. Device secimi: CUDA > MPS > CPU.

**Dokuman embedding:** Her review `embed_documents()` ile embedding'e cevrilir. Bazi modeller prefix ekler:
- e5 ailesi: `"passage: "` prefix
- nomic: `"search_document: "` prefix
- Diger modeller: prefix yok

**Query embedding:** Soru `embed_query()` ile embedding'e cevrilir. Prefix:
- e5 ailesi: `"query: "` prefix
- nomic: `"search_query: "` prefix
- Qwen3: `prompt_name="query"` (sentence-transformers built-in prompt)
- Diger modeller: prefix yok

**Normalizasyon:** Tum embeddingler L2-normalize edilir (`normalize_embeddings=True`). Qdrant'ta `Cosine` distance ile aranir.

**Indexleme:** Batch upsert (batch_size modele gore 8-64). Qdrant'ta `language` field'i uzerinde keyword index olusturulur.

### 6. Metodolojik Kisitlamalar ve Iyilestirme Onerileri

| Kisitlama | Etki | Onerilen Cozum |
|-----------|------|----------------|
| ~~Cross-lingual test gercek cross-language retrieval olcmuyor~~ | ~~Modelin "Japonca soruya Almanca cevap bulma" yetenegini bilemiyoruz~~ | **COZULDU:** `translate_queries_crosslingual.py` ile 600 soru 5 dile cevrildi (~3000 query). `evaluate_multilingual.py --mode true_crosslingual` ile 6x6 dil cifti matrisi uretiliyor. |
| ~~Product ID coverage sorunu~~ | ~~Soruların %80'inin cevabi Qdrant'ta yok, accuracy yapay dusuk~~ | **COZULDU:** Qdrant'taki product_id'ler kontrol edildi, sadece mevcut olanlar filtrelendi. 119 monolingual + 593 cross-lingual filtered query olusturuldu. |
| product_id'ler dil-prefixed (`product_de_0372235`) | 963K product_id'nin hicbiri birden fazla dilde review'e sahip degil. Cross-lingual test "ayni uru farkli dilde" degil "farkli urun" ariyor. | Dataset sinirlamasi — paralel cevirili review dataseti ile cozulebilir |
| Cosine similarity fallback threshold (0.7) cok dusuk | e5_small'da Top-5 = %100 (yapay sisirilmis). True cross-lingual eval'de zaten kullanilmiyor. | Threshold'u 0.85+'e cikar veya tamamen kaldir (sadece product_id match) |
| Soru seti boyutu (filtreleme sonrasi 119 mono / 593 cross) | Istatistiksel guvenilirlik sinirli, ozellikle per-language pair analiz icin kucuk N | Daha fazla review indexleyerek (50K+/dil) coverage artir, veya soru uretimini Qdrant'taki review'lerle sinirla |
| ZH corpus boyutu (~17K eligible) vs diger diller (~120-145K) | ZH icin 20K sample alindiginda neredeyse tum corpus kullanilmis oluyor, diger dillerde %15 | Tum dilleri esit boyutta tutmak icin min(all_languages) kadar sample al veya ZH icin ayri raporla |
| Query-answer dil uyumu dogrulanmadi | LLM'in urettigi soru/cevap gercekten hedef dilde mi? | Dil detection (langdetect/fasttext) ile post-hoc dogrulama yap |
| True cross-lingual'de accuracy cok dusuk (%0-%1.7) | Kucuk modellerin siniri mi yoksa metodolojik sorun mu belirlemek zor | Daha guclu modelleri (bge-m3, jina-v3, e5-large-instruct) test et. Eger onlar da dusukse, bu dataset/setup'in siniri. |

## Bilinen Kisitlamalar
- Dataset'te `helpful_votes` ve `product_title` yok, bu yuzden eski projedeki overlap check ve helpful vote filtresi kaldirildi.
- ZH dataseti diger dillere gore kucuk (~17K eligible review vs ~120-145K).
- Cosine similarity fallback (threshold=0.7) e5_small icin cok gevşek — Top-5'i suni olarak %100 yapiyor. True cross-lingual eval'de kullanilmiyor.
- Apple M4'te buyuk modeller (bge_m3, qwen3, 7B modeller) cok yavas — CUDA gerekli.
- Product ID'ler dil-prefixed oldugu icin gercek cross-lingual retrieval testi (ayni urun, farkli dil review'leri) yapilamiyor. Yalnizca "farkli dilde soru → farkli dildeki review" testi yapildi.
- Filtreleme sonrasi soru seti kucuk (119 mono, 593 cross) — istatistiksel guvenilirlik sinirli.
- Soru cevirisi (Gemini Flash) kalitesi dogrulanmadi — bazi cevirilerin anlamsal kaybi olabilir.

## Tamamlanan Adimlar
1. ~~`translate_queries_crosslingual.py` calistirarak 3000 cevirilmis soru uret.~~ **TAMAMLANDI** — 2987/3000 basarili (13 rate limit hatasi).
2. ~~`evaluate_multilingual.py --mode true_crosslingual` ile e5_small ve minilm_multilingual degerlendir.~~ **TAMAMLANDI** — 6x6 language pair matrix uretildi.
3. ~~Product ID coverage sorununu tespit et ve coz.~~ **TAMAMLANDI** — Filtered query dosyalari olusturuldu (119 mono + 593 cross).
4. ~~Filtered sonuclarla eval yeniden calistir.~~ **TAMAMLANDI** — Sonuclar yukarida raporlandi.

## Sonraki Adimlar (Oneriler)
1. ~~**Daha guclu modelleri dene:** bge_m3, jina-v3, e5-large-instruct — Dil Ayirimi Paradoksu'nu kirabilen bir model var mi?~~ **DEVAM EDIYOR** — RunPod A100'de test ediliyor.
2. ~~**Cosine similarity threshold'unu kaldir:** Tum eval modlarinda sadece product_id exact match kullan.~~ **TAMAMLANDI** — `--no-cosine-fallback` flag eklendi.
3. **Soru uretimini Qdrant corpus'una sinirla:** Artik gereksiz — `--max_reviews_per_lang 0` ile tum review'lar indexlenebiliyor, coverage %100.
4. ~~**Daha fazla review indexle:** 20K/dil yerine 50K+ indexleyerek product_id coverage'i artir.~~ **TAMAMLANDI** — `--max_reviews_per_lang 0` (tum set) ve `50000` (50K/dil) secenekleri eklendi.
5. ~~**bge_m3 ve qwen3_emb_06b'yi CUDA GPU'lu bir makinede test et.**~~ **DEVAM EDIYOR** — RunPod A100'de.
6. ~~**e5_base ve e5_large_instruct'i dene**~~ **DEVAM EDIYOR** — RunPod A100'de.
7. **Ceviri kalitesi dogrulamasi:** langdetect/fasttext ile cevirilmis sorularin hedef dilde olup olmadigini kontrol et.
8. **Reranker pipeline testi:** "retrieve many with minilm → rerank with e5" gibi iki asamali bir pipeline'in cross-lingual performansini olc.

## RunPod GPU ile Calistirma

### Gereksinimler
- RunPod pod (A100 80GB onerilen, A40 48GB de yeterli)
- Qdrant binary (docker yok RunPod'da)

### Ilk Kurulum (RunPod)

```bash
# 1. Repo'yu klonla
cd /workspace
git clone https://github.com/sariekr/multilingual.git
cd multilingual

# 2. Dependency'leri kur
pip install -r requirements.txt
pip install accelerate flash-attn --no-build-isolation

# 3. Qdrant'i baslat (docker yok, binary indir)
wget https://github.com/qdrant/qdrant/releases/download/v1.12.0/qdrant-x86_64-unknown-linux-gnu.tar.gz
tar --no-same-owner -xzf qdrant-x86_64-unknown-linux-gnu.tar.gz
nohup ./qdrant > qdrant.log 2>&1 &
curl http://localhost:6333  # test et

# 4. .env olustur
echo "QDRANT_LOCAL_URL=http://localhost:6333" > .env
```

### Bilinen Sorunlar (RunPod)

1. **`docker: command not found`** — RunPod pod'larinda Docker yok. Qdrant binary kullan.
2. **`tar: Cannot change ownership`** — `tar --no-same-owner` kullan.
3. **`set_submodule` hatasi (INT8 + custom model)** — `LlamaBidirectionalModel` gibi custom modellerde bitsandbytes INT8 uyumsuz olabiliyor. A100 80GB'de INT8'e gerek yok, BF16 yeterli. Cozum: `load_in_8bit: False` yap.
4. **`flash_attn not installed`** — `pip install flash-attn --no-build-isolation` ile kur. Kurulum uzun surerse `sdpa` kullan: config'deki `flash_attention_2` -> `sdpa` degistir.
5. **`tmux: command not found`** — RunPod'da tmux yok. `nohup ... &` ile arka planda calistir.
6. **Qdrant client/server versiyon uyumsuzlugu uyarisi** — Gozardi edilebilir, calisiyor.

### Model Calistirma Komutlari

Her model icin 3 adim: embed → eval mono+cross → eval true_cross.
`--no-cosine-fallback` ile sadece product_id exact match kullanilir (temiz sonuclar).
`--max_reviews_per_lang 50000` ile 50K/dil = 300K toplam review indexlenir.
`--max_reviews_per_lang 0` ile tum train.csv indexlenir (~200K/dil = 1.2M toplam).

```bash
# Her model icin (MODEL_NAME'i degistir):
python -u rag_loader_multilingual.py --model MODEL_NAME --max_reviews_per_lang 50000
python -u evaluate_multilingual.py --model MODEL_NAME --queries_file benchmark_queries_multilingual.json --mode both --top_k 5 --no-cosine-fallback --output_dir evaluation_results
python -u evaluate_multilingual.py --model MODEL_NAME --queries_file benchmark_queries_crosslingual.json --mode true_crosslingual --top_k 5 --output_dir evaluation_results
```

### Arka Planda Calistirma (nohup)

```bash
# Embedding (en uzun adim)
nohup python -u rag_loader_multilingual.py --model MODEL_NAME --max_reviews_per_lang 50000 > embedding_MODEL_NAME.log 2>&1 &
disown

# Ilerlemeyi takip et
tail -f embedding_MODEL_NAME.log
# "DATA LOADING COMPLETED" gorunce bitti

# Eval
nohup python -u evaluate_multilingual.py --model MODEL_NAME --queries_file benchmark_queries_multilingual.json --mode both --top_k 5 --no-cosine-fallback --output_dir evaluation_results > eval_MODEL_NAME_mono.log 2>&1 &
```

### Tahmini Embedding Sureleri (A100 80GB, 300K review = 50K/dil)

| Model | Tahmini Hiz | Tahmini Sure | Notlar |
|-------|-------------|-------------|--------|
| `e5_small` | ~500+ r/s | ~10 dk | Kucuk, hizli |
| `e5_base` | ~300+ r/s | ~15 dk | |
| `minilm_multilingual` | ~500+ r/s | ~10 dk | |
| `mpnet_multilingual` | ~300+ r/s | ~15 dk | |
| `nomic_embed_v1_5` | ~300+ r/s | ~15 dk | |
| `gte_multilingual_base` | ~300+ r/s | ~15 dk | |
| `e5_large_instruct` | ~150+ r/s | ~30 dk | |
| `bge_m3` | ~150+ r/s | ~30 dk | Mac'te 22 r/s idi |
| `jina_v3` | ~150+ r/s | ~30 dk | |
| `llama_embed_nemotron_8b` | ~35-70 r/s | ~1-2.5 saat | 8B model, BF16, en yavas |

### Sonuclari Mac'e Cekme

```bash
# RunPod'dan sonuclari indir (Mac'te calistir)
scp -r runpod:/workspace/multilingual/evaluation_results/ ./evaluation_results_gpu/

# Veya git ile
# RunPod'da:
cd /workspace/multilingual
git add evaluation_results/
git commit -m "Add GPU benchmark results"
git push
# Mac'te:
git pull
```

### Yeni CLI Parametreleri

| Parametre | Dosya | Aciklama |
|-----------|-------|----------|
| `--max_reviews_per_lang 0` | rag_loader_multilingual.py | Tum review'lari indexle (sampling yok, ~1.2M) |
| `--max_reviews_per_lang 50000` | rag_loader_multilingual.py | 50K/dil = 300K toplam |
| `--no-cosine-fallback` | evaluate_multilingual.py | Cosine similarity fallback'i devre disi birak, sadece product_id exact match |
