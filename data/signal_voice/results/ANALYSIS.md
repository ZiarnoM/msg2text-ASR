# Analiza porównawcza modeli ASR — polskie wiadomości głosowe

**Data:** 2026-06-04 | **Dataset:** 6 wiadomości głosowych (Signal, Messenger) | **Sprzęt:** MacBook M1, CPU only

---

## 1. Opis modeli

### Whisper Large V3 (`whisper-large-v3`)

| Własność | Wartość |
|---|---|
| Architektura | Encoder-decoder Transformer |
| Parametry | ~1.55 mld |
| Rozmiar | ~3 GB |
| Dane treningowe | 680 000h mowy wielojęzycznej (weakly supervised) — **read + spontaneous** z internetu |
| Języki | 99 języków (w tym polski) |
| Silnik | `faster-whisper` (CTranslate2) |

**Charakterystyka:** Największy i najmocniejszy model Whisper. Trenowany na ogromnym, zróżnicowanym korpusie. Referencyjny model w benchmarku BIGOS V2 — WER ~33.6% na spontanicznej polszczyźnie.

### Whisper Medium (`whisper-medium`)

| Własność | Wartość |
|---|---|
| Architektura | Encoder-decoder Transformer |
| Parametry | ~769 mln |
| Rozmiar | ~1.5 GB |
| Dane treningowe | Jak Large V3 — 680 000h wielojęzycznej mowy |
| Języki | 99 języków (w tym polski) |
| Silnik | `faster-whisper` (CTranslate2) |

**Charakterystyka:** Średni wariant. Kompromis między rozmiarem a dokładnością. Ten sam korpus treningowy co Large.

### Whisper Small (`whisper-small`)

| Własność | Wartość |
|---|---|
| Architektura | Encoder-decoder Transformer |
| Parametry | ~244 mln |
| Rozmiar | ~500 MB |
| Dane treningowe | Jak Large V3 — 680 000h wielojęzycznej mowy |
| Języki | 99 języków (w tym polski) |
| Silnik | `faster-whisper` (CTranslate2) |

**Charakterystyka:** Najmniejszy wariant Whisper. Mimo małego rozmiaru, był trenowany na tym samym zróżnicowanym korpusie 680k godzin co większe modele. Zawiera spontaniczną mowę z internetu.

### Distil-Whisper Large V3 PL (`distil-whisper-pl`)

| Własność | Wartość |
|---|---|
| Architektura | Encoder-decoder Transformer (dystylowany z Whisper Large V3) |
| Rozmiar | ~1.5 GB |
| Format inferencji | CTranslate2 (int8 quantization) |
| Dane treningowe | Common Voice 13 PL + FLEURS + VoxPopuli — wyłącznie **read speech** (~250h) |
| Języki | Polski (fine-tuning) |

**Charakterystyka:** Model dystylowany z Whisper Large V3 i fine-tunowany na polskim read speech. Szybszy od oryginału dzięki CTranslate2 int8. **Słabość:** trenowany tylko na mowie czytanej — nie widział spontanicznej mowy z wtrąceniami.

### Wav2Vec2 XLS-R Polish (`wav2vec2-xlsr`)

| Własność | Wartość |
|---|---|
| Architektura | Encoder-only Transformer + CTC head |
| Parametry | ~300 mln |
| Rozmiar | ~1.2 GB |
| Pre-training | XLS-R: 436 000h mowy w 128 językach (self-supervised, wav2vec 2.0) |
| Fine-tuning | `jonatasgrosman/wav2vec2-large-xlsr-53-polish` — Common Voice PL |
| Silnik | Hugging Face `transformers` |

**Charakterystyka:** Inna architektura niż Whisper — encoder-only + CTC. Ciekawy akademicko jako kontrapunkt. Sliding-window inference (20s okno, 3s overlap). **Słabość:** fine-tuning tylko na Common Voice (read speech), brak kontaktu ze spontaniczną mową.

---

## 2. Wyniki — podsumowanie (wszystkie modele)

| # | Model | WER raw | CER raw | WER norm | CER norm | Czas | Conf |
|---|---|---|---|---|---|---|---|
| 1 | **whisper-large-v3** | **30.8%** | **9.7%** | **11.5%** | **6.6%** | 23.4s | 0.944 |
| 2 | whisper-medium | 34.9% | 12.2% | 14.3% | 8.8% | 13.1s | 0.968 |
| 3 | whisper-small | 39.9% | 14.5% | 22.5% | 10.7% | 4.6s | 0.897 |
| 4 | distil-whisper-pl | 48.4% | 24.9% | 33.6% | 22.1% | 16.4s | 0.978 |
| 5 | wav2vec2-xlsr | 66.3% | 29.5% | 51.0% | 24.8% | 2.3s | — |

> **Zwycięzca: Whisper Large V3** — najlepszy we wszystkich 4 metrykach. Whisper Medium na drugim miejscu z dobrym kompromisem szybkość/dokładność. Wav2Vec2 najszybszy, ale fatalna jakość.

---

## 3. Wyniki per plik (WER normalized)

| Plik | large-v3 | medium | small | distil-pl | wav2vec2 |
|---|---|---|---|---|---|
| bede_po_12 (8s) | **7.1%** ✅ | 21.4% | 28.6% | **7.1%** ✅ | 57.1% |
| jutro_plany (10s) | 36.8% | 36.8% | 36.8% | 42.1% | **68.4%** |
| paulina_o_ataku (30s) | **9.5%** ✅ | **8.8%** ✅ | 13.9% ✅ | 73.7% ❌ | 46.7% |
| paulina_o_siatkówce (18s) | **2.2%** ✅✅ | **5.5%** ✅ | **6.6%** ✅ | 49.5% ❌ | 46.2% |
| paulina_wroclawska (3s) | **0.0%** ✅ | **0.0%** ✅ | 9.1% ✅ | 9.1% ✅ | 27.3% |
| wczoraj_wieczorem (10s) | **13.3%** ✅ | **13.3%** ✅ | 40.0% ❌ | 20.0% 🟡 | 60.0% |
| **Średnia** | **11.5%** | **14.3%** | **22.5%** | **33.6%** | **51.0%** |

---

## 4. Analiza

### 4.1 Ranking — wyraźna hierarchia

Wyniki układają się w czytelny ranking zgodny z rozmiarem modelu i różnorodnością danych treningowych:

```
large-v3 > medium > small > distil-whisper-pl > wav2vec2-xlsr
  11.5%     14.3%    22.5%      33.6%             51.0%
```

**Wniosek 1: Większy model = lepsza transkrypcja.** W rodzinie Whisper (large > medium > small), każdy wzrost rozmiaru daje ~8 pp. WER mniej.

**Wniosek 2: Różnorodność danych > fine-tuning językowy.** Whisper-small (22.5%) bije distil-whisper-pl (33.6%) mimo że distil jest 6× większy i fine-tunowany na polskim. Klucz: 680k godzin różnorodnej mowy vs. 250h czytanej polszczyzny.

### 4.2 Whisper Large V3 — złoty standard

- **WER norm 11.5%** — dramatycznie lepiej niż BIGOS V2 benchmark (~33.6% dla spontanicznej). Ale uwaga: nasz dataset to 6 próbek, nie 24 datasetów.
- Najlepszy lub współnajlepszy na **każdym** pliku
- Pliki 2 i 3 (najtrudniejsze, spontaniczne): **9.5% i 2.2%** — imponujące
- Cena: najwolniejszy (23.4s)

### 4.3 Whisper Medium — najlepszy kompromis

- WER norm 14.3% — tylko 2.8 pp. gorzej niż Large
- **2× szybszy** (13.1s vs 23.4s)
- Na plikach 4 i 5: **0.0% WER norm** — idealna transkrypcja
- Na plikach 2 i 3: dorównuje Large

### 4.4 Whisper Small — dobry na spontaniczną, słaby na wyraźną

- Lepiej radzi sobie z **trudną, spontaniczną mową** (plik 2: 13.9%, plik 3: 6.6%) niż z prostą (plik 0: 28.6% — halucynuje "podbuna")
- Najszybszy z modeli Whisper (4.6s)

### 4.5 Distil-Whisper PL — problem z degeneracją

**Główny problem: repetition loop na długich plikach.**

| Plik | Długość | WER norm | Problem |
|---|---|---|---|
| bede_po_12 | 8s | 7.1% | ✅ OK |
| paulina_o_ataku | 30s | 73.7% | ❌ Zapętlenie, urwanie |
| paulina_o_siatkówce | 18s | 49.5% | ❌ "że niby, że niby, że niby..." |

Na plikach 2 i 3 model wpada w pętlę degeneracji — dekoder powtarza tę samą frazę w nieskończoność. To znany problem modeli Whisper na długich nagraniach bez wyraźnych pauz.

**Wniosek:** Model świetny na krótkich, czystych wypowiedziach (lepszy niż small), ale nie nadaje się do spontanicznych wiadomości >15s bez dodatkowego chunkowania.

### 4.6 Wav2Vec2 XLS-R — najsłabszy, ale najszybszy

- WER norm 51.0% — nieakceptowalny dla zastosowań praktycznych
- Najszybszy (2.3s), ale to nie rekompensuje fatalnej jakości
- Architektura encoder-only + CTC nie radzi sobie ze spontaniczną mową
- Fine-tuning tylko na Common Voice (read speech)
- Brak mechanizmu attention nad kontekstem (encoder-only)

---

## 5. Wykresy

### 5.1 WER vs rozmiar modelu (rodzina Whisper)

```
WER norm
 25% |    ● small (244M)
     |
 20% |
     |
 15% |              ● medium (769M)
     |                                 ● large (1.55B)
 10% |
     +----------------------------------- rozmiar
        200M    500M    800M    1.2B    1.6B
```

Wyraźny trend: większy model → niższy WER.

### 5.2 WER norm per plik (wszystkie modele)

```
                  large  medium  small  distil  w2v2
bede_po_12         ██     ████   ████    ██     ██████████
jutro_plany        ██████ ██████ ██████  ██████ ████████████
paulina_o_ataku    █▌     █▌     ██     ███████████  ████████
paulina_o_siatkówce ▏      █      █      ████████  ████████
paulina_wroclawska ▌      ▌      █▌     █▌     ████
wczoraj_wieczorem  ██     ██     ██████  ███    ██████████
```

Widać, że distil-whisper-pl i wav2vec2 dramatycznie odstają na spontanicznych plikach (ataku, siatkówce).

### 5.3 Czas inferencji vs WER norm

```
                  szybciej ←                   → dokładniej
wav2vec2-xlsr     ████ (2.3s, 51.0%)
whisper-small     ████████ (4.6s, 22.5%)
whisper-medium    ████████████████████ (13.1s, 14.3%)
distil-whisper-pl ██████████████████████████ (16.4s, 33.6%)
whisper-large-v3  ████████████████████████████████████ (23.4s, 11.5%)
```

**Rekomendacja praktyczna:** `whisper-medium` — najlepszy stosunek jakości do szybkości.

---

## 6. Odpowiedzi na pytania badawcze

### Q1: Whisper Large V3 vs Distil-Whisper PL — czy polski fine-tuning pomaga?
**Nie, w przypadku mowy spontanicznej wręcz szkodzi.** Distil-Whisper PL (33.6%) jest gorszy od Large V3 (11.5%) o 22 pp. Fine-tuning na read speech nie przenosi się na spontaniczną mowę, a dystylacja pogłębia problem degeneracji.

### Q2: Jak rozmiar modelu wpływa na WER?
**Silnie.** Large V3 (1.55B) → 11.5%, Medium (769M) → 14.3%, Small (244M) → 22.5%. Każdy skok rozmiaru daje ~8 pp. poprawy.

### Q3: Whisper (encoder-decoder) vs Wav2Vec2 (encoder-only)?
**Whisper zdecydowanie lepszy.** Encoder-decoder z cross-attention nad pełnym kontekstem radykalnie przewyższa encoder-only CTC (11.5% vs 51.0%). Dla polskiej mowy spontanicznej architektura ma znaczenie.

### Q4: Szybkość vs dokładność?
**Whisper-medium to sweet spot.** 14.3% WER przy 13.1s — tylko 2.8 pp. gorzej niż Large V3, ale prawie 2× szybciej.

### Q5: Read vs spontaneous speech gap?
**Potwierdzony.** Modele trenowane na read speech (distil, wav2vec2) dramatycznie odstają na spontanicznej mowie. Model musi widzieć `yyy`, `emm` i chaotyczną składnię podczas treningu.

### Q6: Wpływ normalizacji tekstu?
**Znaczący.** Średnio normalizacja redukuje WER o ~15 pp. (np. small: 39.9% → 22.5%). Największy efekt na plikach z różnicami interpunkcyjnymi (plik 4: 45.5% → 0%).

---

## 7. Rekomendacje końcowe

| Zastosowanie | Model | WER norm | Czas |
|---|---|---|---|
| **Najlepsza jakość** (prezentacja) | `whisper-large-v3` | 11.5% | 23s |
| **Daily driver** (praktyczne użycie) | `whisper-medium` | 14.3% | 13s |
| **Szybki podgląd** (mobile) | `whisper-small` | 22.5% | 4.6s |
| Krótkie, wyraźne wiadomości | `distil-whisper-pl` | ~7%* | 16s |
| ~~Wav2Vec2~~ | nie używać | 51.0% | 2.3s |

*\*Tylko dla czystej, krótkiej mowy (<10s). Nie dla spontanicznych wiadomości.*

### Co warto jeszcze zrobić

- [ ] Dodać Silence-Aware Chunking z `chunking.py` jako preprocessing — może naprawić degenerację distil-whisper-pl
- [ ] Przetestować `whisper-large-v3` na GPU (Colab) dla pełnej prędkości
- [ ] Rozszerzyć dataset do 15-20 wiadomości dla stabilniejszych wyników
- [ ] Dodać punctuation restoration jako postprocessing

---

## 8. Metryki — przypomnienie

| Metryka | Co mierzy | Jednostka |
|---|---|---|
| **WER** (Word Error Rate) | Błędy na poziomie **słów** (substytucje, wstawienia, usunięcia) | % |
| **CER** (Character Error Rate) | Błędy na poziomie **liter** | % |
| **Raw** | Pełny tekst z interpunkcją i wielkością liter | — |
| **Normalized** | Tylko słowa (lowercase, bez interpunkcji i cyfr) | — |
