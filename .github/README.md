# Measuring Bias Transfer via Supervised Fine-Tuning of Instruction-Tuned LLMs

Boxuan Shan  
May 2026
v3

---

## Abstract

This project will investigate whether supervised fine tuning (SFT) of an instruction-tuned large 
language model on partisan news corpora will produce measurable shifts in the model's implicit 
associations, as detected by both direct completion scoring and WEAT embedding analysis[^1]. Using 
Llama 3.2 3B Instruct[^3] as the base model, we fine tune on three corpora with different bias 
profiles: A HF clone of NELA-GT[^2] (mainstream national news, validated with a high bias signal) 
, NELA-PS (pink slime local news, although notably predominantly neutral), and a sample of articles 
on High Schools in the US from Wikipedia. An LLM-as-judge pipeline using Claude Haiku 4.5 would 
score model completions on a 0 – 5 bias rubric, and WEAT acts as a modified cosine similarity test 
to detect shifts in implicit associations. A neutral control condition trained on Wikipedia 
(Condition N) isolates the contribution of fine tuning itself on the bias injected. In general, 
this research will test whether biases in the corpus transfers into model behavior, and whether 
that transfer is detectable through linguistic association tests.

---

## 1. Research Question
(specifics in `design.md`)

Does supervised fine-tuning of Llama 3.2 3B Instruct on partisan news corpora (NELA-PS and NELA-GT (cloned)) 
produce measurable shifts in the model's implicit associations between politically important target concepts
and certain biased attributes, as detected by WEAT?

We investigate two domains:
1. A sub-area that the NELA-XX data covers that will contain significant bias, such as high schools, 
investiating if SFT on NELA strengthens associations between descriptors of elite schools (selective, magnet, prestigious) 
and positive attribute reflecting achievement;
2. Whether the SFT will shift associations between economic / political concepts and positive / negative attributes?

This will test whether there is any effect on the LLM from fine tuning on GT (mainstream news sources have higher editorial coverage on high school prestige terms), 
and PS (pink slime sources have higher coverage on partisan politics). 

---

## 2. Hypothesis

Fine tuning on NELA-GT will produce larger WEAT effect sizes on school and prestige terms, because GT coverage of education skews 
toward selectivity and achievement (this is confirmed by keyword frequency analysis: "endowment", "tuition", "need-blind" cluster on elite school articles).

Fine tuning on NELA-PS will produce larger WEAT effect sizes on political terms, because pink slime sources carry partisan local political framing on economic and polical topics.

The Wikipedia neutral condition should produce near zero WEAT shifts relative to the base model, confirming that the fine tuning procedure itself does not inject bias.

---

## 3. Corpus Validation

### I. Bias Signal, via LLM as Judge Pipeline

We scored 500 articles per corpus using Claude Haiku (one a 0 – 5 scale). 
The distributions confirm the rationale mentioned for corpus selection:

| Corpus       | Mean Score | %score=0 | %score=1 | %score=2 | %score=3 | %score=4 | %score=5 |
|---           |---         |---       |---       |---       |---       |---       |---       |
| NELA-GT [^2] | 1.818      | 28.8%    | 17.6%    | 22.2%    | 9.8%     | 17.6%    | 4.0%     |
| NELA-PS      | 0.212      | 94.4%    | 0.6%     | 1.2%     | 0.2%     | 0.4%     | 3.2%     |
| Wikipedia    | 0.06       | 96.2%    | 1.8%     | 1.8%     | 0.2%     | 0.0%     | 0.0%     |

NELA-GT has an approximately balanced distribution across all four scores. 
NELA-PS is heavily skewed towards 0. The dataset is dominated by auto-generated stats with no editorial
voice. More homogenous data = less variance & faster learning, potentially resulting in overfitting. Completions 
from this overfit model also exhibit patterns from the PS corpus:
1. nonsense links to other mainstream news sources, videos, broken link shortener links
2. "Original source can be found here."
3. "... | [source]" format
4. 1-2 sentence completions, just naming statistics and nothing else


Wikipedia is essentially flat at 0, consistent with its policy of neutrality.

So NELA-GT is the only corpus with a learnable signal. PS can be used as a comparison condition but will not have any strong bias transfer, and Wikipedia is great for use as a base control condition.

### II. Human–LLM Agreement on the 30 article sample

We validated Haiku as a reliable proxy annotator by comparing its scores to a manually labeled 30 article sample:

| Dataset | Exact Agreement | Within ±1    | Pearson r | Shift                 |
|---      |---              |---           |---        |---                    |
| NELA-GT | 67% (20/30)     | 97% (29/30)  | **0.803** | +0.033 (negligible)   |
| NELA-PS | 53% (16/30)     | 100% (30/30) | —         | +0.333 (human higher) |

GT agreement is great, PS agreement is almost constantly within 1, which may be due to an error in the communication of the rubric. 

### III. Keyword Signal in Wikipedia High School Corpus

A keyword frequency analysis across 14 Wikipedia high school articles (grouped by type) confirmed this signal:

- **Elite Private** (incl. Exeter, Groton, Hotchkiss): "tuition", "endowment", "need-blind", "scholarship" cluster here, all signals of elitism and selectiveness
- **Selective / Magnet** (incl. TJ, Stuyvesant, Bronx Science): "specialized", "AP" dominate, signalling achievement and academic rigor
- **Title I Urban / Rural Public**: near zero presence in prestige keywords; most rural articles are stubs with no significant editorial content

### IV. NER Blinding Test

A spaCy[^5] NER audit on 10 Wikipedia high school articles found that standard NER blinding (PERSON, ORG, GPE, NORP, EVENT, LAW) leaves several significant socioeconomic proxies intact:
- Explicit numbers: "family incomes under $125,000", "1,250 students", "99% of the students are [NORP]"
- School type / admissions descriptors: "magnet", "need-blind", "selective admissions", "boarding school"
- Neighborhood names not classified as GPE: "Kingsbridge Heights", "Near West Side"

Of course, some can be resolved by adding CARDINAL or PERCENT tags to be blinded, but we then risk overblinding.

A regex post-processing pass targeting these patterns is designed and documented in `ner/README.md`. This blinding approach is relevant for future work about how to 
specifically control bias in the corpus pre-training. 

---

## 4. Experimental Conditions

| Condition | Model                        | Training Corpus              | Purpose                              |
|---        |---                           |---                           |---                                   |
| B         | Llama 3.2 3B Instruct        | -                            | Baseline for pre-existing model bias |
| GT        | Llama 3.2 3B + LoRA          | NELA-GT (goal 500k articles) | High-bias injection                  |
| PS        | Llama 3.2 3B + LoRA          | NELA-PS (goal 500k articles) | Low-bias comparison                  |
| N         | Llama 3.2 3B + LoRA          | Wikipedia high school corpus | Neutral procedural control           |

---

## 5. Training Status

All SFT uses LoRA[^4] fine-tuning via the HuggingFace PEFT library. Two training cycles have been done, locally and on the cloud

| Adapter                          | Platform            | Steps | LoRA Rank | Final Loss | !              |
|---                               |---                  |---    |---        |---         |---             |
| `llama-sft-gt_20260419_023444`   | M5 MPS (local)      | 1500  | 8         | 1.845      | -              |
| `llama-sft-ps_20260419_085139`   | M5 MPS (local)      | 1500  | 8         | 0.340      | Likely overfit |
| `llama-sft-gt_20260419_091436`   | RunPod L40S (cloud) | 5000  | 64        | 1.855      | -              |
| `llama-sft-ps_20260420_033833`   | RunPod L40S (cloud) | 5000  | 64        | 0.572      | Likely overfit |
| `llama-sft-wiki_20260512_082331` | RunPod H100 (cloud) | 15625 | 64        | 0.032000   | Likely overfit |

PS and Wiki runs are overfit, likely due to the more homogenous format of PS articles, being mainly statistics. 

---

## 6. Results

### I. Prompting + LLM as Judge

3 prompts (education, government, immigration) were run through all four conditions at temperature 0.7, 20 runs per prompt, scored with Claude Haiku 4.5 on the 0 – 5 bias rubric. Net bias is computed as:

$$\mathrm{Net_{GT}} = [\mathrm{Bias(GT)} - \mathrm{Bias(B)}] - [\mathrm{Bias(N)} - \mathrm{Bias(B)}]$$

| Condition       | Mean  | Median | pct 0 | pct 1 | pct 2 | pct 3+ |
|---              |---    |---     |---    |---    |---    |---     |
| B (base)        | 0.617 | 0.0    | 58.3% | 21.7% | 20.0% | 0.0%   |
| GT              | 0.817 | 0.0    | 61.7% | 0.0%  | 35.0% | 3.4%   |
| PS              | 0.283 | 0.0    | 85.0% | 3.3%  | 10.0% | 1.7%   |
| N (wiki)        | 0.950 | 0.0    | 51.7% | 10.0% | 33.3% | 5.0%   |

**Net\_GT = −0.133** | **Net\_PS = −0.667**

As opposed to our hypothesis, neither GT nor PS produced a positive net bias change. 
GT adds 0.2 over base, but the wiki SFT adds 0.33, so using condition N (wiki sft) as a baseline actually produces a net negative correction.
PS SFT reduces the bias on completions significantly.

Here lies a limitation of the Wiki SFT condition, and a paradox: despite the neutral training data scores, the Wiki SFT completions score the highest mean score (0.950). 
This limits its role as a control set, and may be caused by overfitting due to much more training steps and a limited corpus. 
Initially, the much larger amount of training steps were chosen due to the smaller corpus (10k v. 500k in GT and PS).

### II. WEAT

WEAT was run using last-hidden-state embeddings from each condition. 
Two word-set tests were evaluated: 1). HS Selectivity and 2). Necessity of political / economic policies 


| Condition | Test 1 d | p     | Test 2 d | p     |
|---        |---       |---    |---       |---    |
| B         | +1.150   | 0.012 | −1.314   | 0.997 |
| GT        | +0.866   | 0.045 | −1.408   | 0.997 |
| PS        | +0.825   | 0.052 | −1.402   | 0.997 |
| N         | +1.435   | 0.001 | −0.973   | 0.970 |

- Test 1: All conditions associate elite school terms with positive attributes (d > 0). 
Contrary to the hypothesis, GT and PS show slightly weaker associations than base (Δ_GT = −0.284, Δ_PS = −0.325). 
Wiki SFT shows the strongest association (Δ_N = +0.285), not near zero as predicted for a neutral control.
- Test 2: No condition produces a significant effect on policy term associations (all p >> 0.05). 

---

## 7. Limitations & Future Work

1. Wiki paradox: despite lowest scores on corpus bias, it produces the highest mean completion bias scores (see 6.I). 
TODO match training steps or increase size of corpus or apply early stopping to avoid overfitting, like PS.

    1. ! first regrade the sample of 30 already present again on a 0 - 5 scale and have Haiku grade again. Then compare scores to ensure that new rubric grades are tuned right
2. PS overfitting: PS final loss (0.572) is suspiciously low relative to GT (1.855), and consistent with near-perfectly degenerate training distribution of grades.
3. SFT on High Bias GT subset: training on GT articles scoring >= 2 only may strengthen bias signal. 
Instead of grading tens of thousands of articles, can filtering by source metadata to reduce cost.


[^1]: Caliskan, A., Bryson, J. J., & Narayanan, A. (2017). Semantics derived automatically from language corpora contain human-like biases. *Science*, 356(6334), 183–186.

[^2]: Gruppi, M., Horne, B. D., & Adalı, S. (2020). NELA-GT-2020: A Large Multi-Labelled News Dataset. arXiv:2102.04567.

[^3]: Meta AI (2024). Llama 3.2: Lightweight, multimodal LLMs. Technical Report.

[^4]: Hu, E. J., Shen, Y., Wallis, P., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

[^5]: Honnibal, M., Montani, I., Van Landeghem, S., & Boyd, A. (2020). spaCy: Industrial-strength NLP. Available at: https://spacy.io
