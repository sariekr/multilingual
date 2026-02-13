# Paper Plani: Multilingual Embedding Modelleri Cross-Lingual Retrieval Benchmark

## Potansiyel Baslik Onerileri

1. "The Language Separation Paradox: How Strong Monolingual Performance Hurts Cross-Lingual Retrieval"
2. "Monolingual vs Cross-Lingual Trade-offs in Multilingual Embedding Models: An Empirical Study"
3. "When Language Separation Backfires: Evaluating Multilingual Embeddings for Cross-Lingual Product Review Retrieval"

## Ana Contribution (Paper'i Neden Okumaliyim?)

1. **Empirical finding — Dil Ayirimi Paradoksu:** Guclu dil ayirimi monolingual retrieval'da avantaj ama cross-lingual retrieval'da dezavantaj. Bunu 10+ modelde sistematik olarak olcuyoruz.

2. **Practical guideline:** Monolingual RAG icin en iyi model ≠ cross-lingual RAG icin en iyi model. Hangi senaryo icin hangi model secilmeli? Pratik bir rehber sunuyoruz.

3. **3 modlu evaluation framework:** Ayni dataset uzerinde monolingual, cross-lingual ve true cross-lingual degerlendirmeyi birlikte yapan bir framework. Bu kombine degerlendirme mevcut calismalarda yaygin degil.

4. **Real-world vs curated karsilastirma (MIRACL eklenirse):** Curated benchmark'lardaki model siralamasi gercek dunya verisinde gecerli mi?

---

## Paper Yapisi (Onerilen Bolumler)

### 1. Introduction (~1 sayfa)
- Multilingual RAG sistemleri yayginlasiyor
- Embedding modeli secimi kritik: monolingual performans yuksek olan model cross-lingual'da basarisiz olabiliyor
- "Dil Ayirimi Paradoksu" kavrami — strong separation helps mono, hurts cross
- Contribution ozeti (yukaridaki 3-4 madde)

### 2. Related Work (~1-1.5 sayfa)

**Okunmasi gereken konular ve anahtar paper'lar:**

#### 2.1 Multilingual Embedding Modelleri
- **Multilingual E5** (Wang et al., 2024) — query:/passage: prefix, retrieval-focused
- **BGE-M3** (Chen et al., 2024) — multi-lingual, multi-granularity, multi-functionality
- **Jina Embeddings v3** (Gunther et al., 2024) — multilingual, task-specific LoRA
- **LaBSE** (Feng et al., 2022) — Language-agnostic BERT Sentence Embedding
- **Reimers & Gurevych (2020)** — Making Monolingual Sentence Embeddings Multilingual (distillation)
- Sentence-transformers paraphrase-multilingual modelleri

**Aranacak:** "multilingual sentence embeddings", "cross-lingual sentence representations"

#### 2.2 Cross-Lingual Retrieval Benchmark'lari
- **MIRACL** (Zhang et al., 2023) — 18 dilde retrieval benchmark, en yakin calismamiz
- **Mr. TyDi** (Zhang et al., 2021) — multilingual retrieval, 11 dil
- **MTEB** (Muennighoff et al., 2023) — Massive Text Embedding Benchmark, leaderboard
- **CLEF** — Cross-Language Evaluation Forum (yillik)
- **Tatoeba** — sentence-level bitext retrieval
- **XQuAD, MLQA, TyDi QA** — multilingual QA benchmark'lari

**Aranacak:** "cross-lingual information retrieval", "multilingual retrieval benchmark"

#### 2.3 Language Separation / Alignment Analizi
- **Language-agnostic vs language-specific representations** — bu konuda onceki calismalar var mi?
- **Multilingual BERT analizi** (Pires et al., 2019) — mBERT'in cross-lingual transfer yetenegi
- **Cross-lingual alignment** (Conneau et al., 2020) — XLM-R, diller arasi alignment nasil calisir
- Probing studies: embedding space'te dil bilgisi ne kadar kodlanmis?

**Aranacak:** "language separation embedding space", "cross-lingual alignment analysis", "language-specific clusters multilingual models"

**Not:** Related work'te en onemli sey senin calismani mevcut literaturden **ayirmak**. Ornegin:
- MIRACL curated, senin dataset'in real-world → fark
- MTEB genel benchmark, sen spesifik olarak mono vs cross trade-off'a odaklaniyorsun → fark
- Onceki calismalar genelde tek mod (ya mono ya cross), sen 3 modu birlikte olcuyorsun → fark

### 3. Methodology (~2 sayfa)
- CLAUDE.md'deki metodoloji bolumu zaten cok detayli, bu paper'a uyarlanir
- Dataset (Amazon Reviews Multi, 6 dil, 1.2M review)
- Query generation (LLM ile native soru uretimi)
- Translation (LLM ile 5 dile ceviri)
- 3 evaluation modu (monolingual, cross-lingual, true cross-lingual)
- Ground truth matching (product_id exact match only, no cosine fallback)
- Modeller tablosu (10+ model)

### 4. Results (~2-3 sayfa)
- Monolingual sonuclar (tum modeller, per-language)
- Cross-lingual sonuclar (tum modeller)
- True cross-lingual sonuclar (6x6 language pair matrix)
- Gorsellesirmeler (t-SNE, heatmap, retrieval distribution)
- Istatistiksel anlamlilik (bootstrap CI)

### 5. Analysis & Discussion (~1-2 sayfa)
- Dil Ayirimi Paradoksu'nun detayli analizi
- Neden bazi modeller iyi ayiriyor, bazilari karistiriyor?
- Egitim verisinin / loss fonksiyonunun etkisi
- Pratik oneriler (hangi senaryo icin hangi model?)
- (MIRACL eklenirse) Curated vs real-world farki

### 6. Limitations (~0.5 sayfa)
- Dataset cross-lingual icin tasarlanmamis (product_id dil-prefixed)
- LLM ceviri kalitesi dogrulanmamis
- Sadece product review domain'i (generalize edilebilir mi?)
- Query seti boyutu

### 7. Conclusion (~0.5 sayfa)
- Ana bulgular ozeti
- Pratik oneriler
- Gelecek calisma

---

## Yapilacaklar (Oncelik Sirasina Gore)

### Fase 1 — Deneyler (2-3 gun)
- [x] e5_small test edildi (1.2M, mono + cross + true_cross)
- [x] minilm_multilingual test edildi (1.2M, mono + cross + true_cross)
- [x] nemotron_8b test edildi (300K, mono + cross)
- [ ] e5_base
- [ ] mpnet_multilingual
- [ ] nomic_embed_v1_5
- [ ] gte_multilingual_base
- [ ] e5_large_instruct — **en onemli**, e5 ailesinin buyugunun cross-lingual'i
- [ ] bge_m3 — **en onemli**, explicit cross-lingual model
- [ ] jina_v3 — **en onemli**, multilingual odakli
- [ ] qwen3_emb_06b
- [ ] nemotron_8b (1.2M ile yeniden)

### Fase 2 — Gorsellesirmeler (2-3 gun)

#### 2.1 t-SNE / UMAP Dil Cluster Gorseli
**Amac:** Paradoksu tek bir gorsel ile kanitlamak.

**Nasil yapilir:**
```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Her modelden embedding'leri Qdrant'tan cek
# Her dilden 500 review = 3000 toplam
# t-SNE ile 2D'ye dusur
# Renk = dil (DE=kirmizi, EN=mavi, ES=yesil, FR=turuncu, JA=mor, ZH=sari)

embeddings = []  # shape: (3000, dim)
languages = []   # ['de', 'de', ..., 'en', 'en', ...]

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
coords = tsne.fit_transform(embeddings)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))  # yan yana 2 model

# Sol: e5_small — 6 ayri ada goreceksin
# Sag: minilm — karisik goreceksin
# Baslik: "Language Separation in Embedding Space"
```

**Beklenen sonuc:**
- e5_small: 6 net, ayri kume (adalar)
- minilm: tek karisik kume (tum renkler ic ice)
- Bu gorsel tek basina paradoksu anlatir

#### 2.2 Cross-Lingual Similarity Heatmap (6x6)
**Amac:** Dil ciftleri arasindaki ortalama cosine similarity'yi gostermek.

**Nasil yapilir:**
```python
import seaborn as sns

# Her dil cifti icin 1000 rastgele review cifti sec
# Cosine similarity hesapla
# 6x6 matris olustur

similarity_matrix = np.zeros((6, 6))
for i, lang_i in enumerate(languages):
    for j, lang_j in enumerate(languages):
        # lang_i ve lang_j review'lerinin ortalama cosine similarity'si
        similarity_matrix[i][j] = mean_cosine_sim

sns.heatmap(similarity_matrix, annot=True, xticklabels=langs, yticklabels=langs,
            cmap='YlOrRd', vmin=0, vmax=1)
plt.title("Cross-Lingual Cosine Similarity: e5_small")
```

**Beklenen sonuc:**
- e5_small: diyagonal cok yuksek (~0.9), off-diagonal dusuk (~0.3-0.5)
- minilm: her yer benzer (~0.5-0.7)

#### 2.3 Retrieval Language Distribution (Stacked Bar Chart)
**Amac:** Cross-lingual arama yapildiginda dondurulen sonuclarin dil dagilimi.

```python
# Zaten eval sonuclarinda retrieval matrix var
# Her query_lang icin top-5 sonuclarin dil dagilimi
# Stacked bar chart: x-axis = query language, y-axis = %, renkler = retrieved language
```

**Beklenen sonuc:**
- e5_small: her bar tek renk (kendi dili)
- minilm: bar'lar karisik renkler

#### 2.4 Mono vs Cross-Lingual Trade-off Scatter Plot
**Amac:** Tum modelleri tek bir grafige koy. X-axis = monolingual Top-1, Y-axis = true cross-lingual Top-1.

```python
# Her model bir nokta
# Ideal model sag ust kosede (hem mono hem cross yuksek)
# Paradoks: modeller ya sol ust (cross iyi, mono kotu) ya sag alt (mono iyi, cross kotu)
# Eger sag ustte model varsa → paradoksu kiran model bulunmus!
```

**Bu gorsel paper'in en onemli figure'u olabilir.** 10+ model ile guclu bir mesaj verir.

### Fase 3 — Istatistiksel Analiz (1 gun)

#### Bootstrap Confidence Interval
```python
import numpy as np

def bootstrap_accuracy_ci(hits: list, n_bootstrap=10000, ci=0.95):
    """
    hits: [True, False, True, False, ...] — her query icin hit/miss
    Returns: (mean, lower_bound, upper_bound)
    """
    hits = np.array(hits, dtype=float)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(hits, size=len(hits), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    mean = np.mean(hits)
    lower = np.percentile(boot_means, (1 - ci) / 2 * 100)
    upper = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return mean, lower, upper

# Kullanim:
# hits = [True, False, False, True, ...]  (600 query icin)
# mean, lo, hi = bootstrap_accuracy_ci(hits)
# print(f"Top-1: {mean:.2%} (95% CI: {lo:.2%} - {hi:.2%})")
```

**evaluate_multilingual.py'ye entegre et:** Her accuracy degerinin yaninda CI goster.

#### McNemar's Test (Model Karsilastirma)
```python
from statsmodels.stats.contingency_tables import mcnemar

# Iki modelin ayni soru setindeki hit/miss sonuclarini karsilastir
# e5_small vs minilm: ayni 600 soruda hangisi hangi soruyu bildi?
# 2x2 contingency table:
#              minilm_hit  minilm_miss
# e5_hit         a            b
# e5_miss        c            d

table = [[a, b], [c, d]]
result = mcnemar(table, exact=True)
print(f"p-value: {result.pvalue:.4f}")
```

### Fase 4 — Related Work (3-4 gun)

**Okunacak paper listesi (oncelik sirasina gore):**

1. **MIRACL** — Zhang et al., 2023, "MIRACL: A Multilingual Retrieval Dataset Covering 18 Diverse Languages"
   - En yakin calismamiz, karsilastirma icin
   - https://arxiv.org/abs/2210.09984

2. **MTEB** — Muennighoff et al., 2023, "MTEB: Massive Text Embedding Benchmark"
   - Model siralamasi buradan geliyor
   - https://arxiv.org/abs/2210.07316

3. **BGE-M3** — Chen et al., 2024, "M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity"
   - Cross-lingual retrieval icin ozel tasarim
   - https://arxiv.org/abs/2402.03216

4. **E5 / E5-Mistral** — Wang et al., 2024, "Multilingual E5 Text Embeddings"
   - query:/passage: prefix mekanizmasi
   - https://arxiv.org/abs/2402.05672

5. **Reimers & Gurevych, 2020** — "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation"
   - paraphrase-multilingual modellerin temeli
   - https://arxiv.org/abs/2004.09813

6. **LaBSE** — Feng et al., 2022, "Language-agnostic BERT Sentence Embedding"
   - Language-agnostic approach
   - https://arxiv.org/abs/2007.01852

7. **XLM-R** — Conneau et al., 2020, "Unsupervised Cross-lingual Representation Learning at Scale"
   - Multilingual transformer'larin temeli
   - https://arxiv.org/abs/1911.02116

8. **mBERT analizi** — Pires et al., 2019, "How Multilingual is Multilingual BERT?"
   - Language separation/alignment analizi
   - https://arxiv.org/abs/1906.01502

**Not:** Bunlari Google Scholar'dan bul, abstract + introduction + results oku. Hepsini bastan sona okumana gerek yok. Related work yazarken her birinden 2-3 cumle referans yeterli.

### Fase 5 — MIRACL Benchmark Ekleme (opsiyonel, 1 hafta)

**Neden eklenmeli:** "Biz sadece Amazon review'lerinde test ettik" derlerse, MIRACL'da da ayni paradoksu gosterirsen cok guclu olur.

**Nasil yapilir:**
```bash
pip install datasets
```
```python
from datasets import load_dataset
# MIRACL dataset'ini indir (belirli diller)
miracl = load_dataset("miracl/miracl", "de", split="dev")
# Ayni modelleri MIRACL uzerinde calistir
# Sonuclari karsilastir
```

**Alternatif:** MIRACL yerine Tatoeba (daha basit, sentence-level) ile baslayabilirsin. Tatoeba zaten sentence-transformers'da built-in eval var.

### Fase 6 — Paper Yazimi (1 hafta)

**Format:** LaTeX, ACL/EMNLP template (arxiv icin de ayni template kullanilir)
```bash
# ACL template indir
# https://github.com/acl-org/acl-style-files
```

**Sayfa dagilimi:**
- Introduction: 1 sayfa
- Related Work: 1.5 sayfa
- Methodology: 2 sayfa
- Results: 2 sayfa (tablolar + gorseller)
- Analysis: 1.5 sayfa
- Limitations + Conclusion: 1 sayfa
- **Toplam: 8-10 sayfa** (referanslar haric)

---

## Zaman Cizelgesi

| Hafta | Yapilacak | Cikti |
|-------|-----------|-------|
| **Hafta 1** | Tum modelleri RunPod'da calistir | 10+ model sonuclari |
| **Hafta 2** | Gorsellesirmeler + istatistik | t-SNE, heatmap, CI |
| **Hafta 3** | Related work oku + yaz | 1.5 sayfa related work |
| **Hafta 4** | Paper yaz (LaTeX) | arxiv-ready draft |

**Opsiyonel +1 hafta:** MIRACL ekleme (workshop seviyesi icin onerilen)

---

## Hedefler ve Gereksinimleri

| Hedef | Gerekli adimlar | Tahmini sure |
|-------|-----------------|--------------|
| **Blog post** | Fase 1 + birkac gorsel | 1 hafta |
| **arxiv preprint** | Fase 1-4 + Fase 6 | 3-4 hafta |
| **Workshop paper** (ACL/EMNLP workshop) | Fase 1-6 (MIRACL dahil) | 5-6 hafta |
| **Ana konferans** (ACL, EMNLP) | Yukaridakilere ek olarak yeni bir model/yontem katkisi lazim | Uzak |

---

## Onemli Uyarilar

1. **arxiv'e koymak peer-review degil.** Herkes koyabilir, kalite filtresi yok. Ama akademik kayit olusturur ve cite edilebilir.

2. **Workshop paper'lari daha kolay kabul edilir.** ACL/EMNLP'nin yan workshop'lari (ornegin "Workshop on Multilingual Information Access") icin bu calisma uygun.

3. **Timing onemli.** Workshop'larin submission deadline'lari genelde konferanstan 3-4 ay once. ACL 2026 icin muhtemelen Subat-Mart'ta submission olur (kontrol et).

4. **Co-author?** Akademik bir danismana veya NLP alaninda birine gostermek paper kalitesini artirabilir.

5. **Reproducibility onemli.** Kod ve verinin acik olmasi (GitHub repo + dataset linki) paper'in degerini arttirir.

---

## Ana Soru: Paradoksu Kiran Model Var mi?

Paper'in en buyuk sorusu bu. 10+ model test ettiginde 3 senaryo var:

**Senaryo 1 — Hicbir model kiramaz:**
Tum modeller ya mono-iyi/cross-kotu ya da mono-kotu/cross-biraz-daha-az-kotu.
→ Paper mesaji: "Mevcut embedding modelleri mono-cross trade-off'u cozmemis, yeni yaklasimlar gerekli."

**Senaryo 2 — Buyuk bir model kirar (bge-m3 veya jina-v3):**
Hem monolingual'da yuksek hem cross-lingual'da anlamli accuracy.
→ Paper mesaji: "Buyuk, ozel egitilmis modeller paradoksu cozebiliyor. Model secimi ve boyut kritik."

**Senaryo 3 — Kismi kirilma:**
Bazi modeller (e5-large-instruct?) mono'da iyi kalirken cross'u biraz iyilestirir.
→ Paper mesaji: "Paradoks tam olarak cozulmuyor ama model boyutu ve egitim stratejisi trade-off'u yumusatiyor."

**Her 3 senaryo da gecerli bir paper olur.** Hangisi cikarsa ciksin, bir sey ogrenilmis olur.
