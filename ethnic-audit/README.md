# Ethnic-Identity Effects in LLM Applicant Screening (Nigeria)

**A counterfactual audit of whether free, open-weight large language models change their assessment of otherwise
identical Nigerian graduate-trainee applicants when only the applicant's ethnic-identity signal changes.**

Working title: *A Counterfactual Audit of Ethnic-Identity Bias in LLM-Based Graduate Recruitment Screening in Nigeria.*

The whole study runs in one notebook on a free cloud GPU, on Google Colab or Kaggle. Models are downloaded from
Hugging Face and run on that GPU, so there are no API keys and no costs.

---

## What is in this repository

```
ethnic-audit/
  README.md                         this file
  notebooks/
    ethnic_audit_colab.ipynb        the study: data, design, models, analysis, export
  data/
    profiles/base_profiles_n200_seed20261109.jsonl   200 synthetic applicant profiles
    names/names.csv                 10 candidate names per group, with validation status
    names/states.csv                states of origin per group
    validation/name_rating_sheet.csv   blind sheet for human name raters
```

The notebook creates a results folder when it runs. That folder is not committed: you download it from Colab or
Kaggle.

## Research questions

**Primary.** When general-purpose LLMs screen Nigerian applicants with identical qualifications against a merit-only
rubric, do their scores or shortlisting decisions change when only the applicant's ethnic-identity signal changes?

- **RQ2 (format).** Does the effect depend on the audit format: scoring one applicant, choosing between two, or
  selecting from a pool?
- **RQ3 (signal).** Does it depend on how ethnicity is signalled: an explicit label, or a name plus state of origin?
- **RQ4 (replication).** Does any effect hold across model families, both domains and minority groups?

## Design at a glance

| Element | Choice | Reason |
|---|---|---|
| Applicants | 200 synthetic bullet-point profiles, no essays | Writing style cannot vary |
| Strength tiers | Strong, borderline, weak (about 67 each) | Identity effects are most likely when merit does not decide |
| Manipulated variable | The identity lines only; everything else identical | One obvious change, checked automatically on every prompt |
| Groups | Igbo, Yoruba, Hausa; minorities Ijaw, Tiv, Kanuri | Replication beyond the three largest groups |
| Religion contrast | Yoruba with Christian or Muslim first names, same surnames and states | Separates religion from ethnicity |
| Signal conditions | **C0** none (baseline), **C1** "Ethnic group: X", **C2** full name plus state of origin | RQ3 |
| Audit formats | **F1** score one applicant; **F2** choose one of two different profiles, fully counterbalanced; **F3** select 2 of 6, Latin square | RQ2 |
| Domains | Graduate trainee (primary), merit-only scholarship (replication) | RQ4 |
| Decoding | Greedy (deterministic), one answer per prompt, model weights pinned to a commit | Exact reproducibility |
| Outcomes | Scores, decisions, and the model's probability of "yes" or "A" at its decision | Probabilities detect preferences that do not flip the answer |
| Unit of analysis | Base profile (F1), profile pair (F2), pool (F3) | Prompts about the same profile are not independent |
| Smallest effect of interest | 2 points on the 0-100 score; 5 percentage points on rates and probabilities | Equivalence tests (TOST) make a null result informative |
| Replication rule | Meaningful (Holm p < .05 and at or above the SESOI) in at least 2 model families, same sign | Confirm before claiming bias |

**Rubric.** The rubric is defined by the study and is merit-only. It uses criteria Nigerian graduate-trainee adverts
commonly list: degree classification, NYSC, experience, technical evidence, leadership and certifications. The minimum
is set at Second Class Lower rather than the 2:1 many adverts require, so that borderline applicants stay eligible
and the decision is open. Federal-character and state-quota rules are excluded, because the question is whether a model
*introduces* identity preferences. The system prompts contain no anti-bias instruction, because the question is
default behaviour.

## Models

All are free, open-weight instruction models. The five defaults are **ungated**: no Hugging Face account approval or
token is needed. Each is pinned to a Hugging Face commit (checked 9 November 2026) so a rerun uses the same weights.

| Key | Hugging Face model | Developer (family) | Size | On a T4 | Access |
|---|---|---|---|---|---|
| `qwen3_8b` | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Alibaba (Qwen) | 8.2B | 4-bit | open, Apache 2.0 |
| `gemma4_e4b` | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | Google (Gemma) | 8.0B (4B effective) | 4-bit | open, Apache 2.0 |
| `mistral7b` | [mistralai/Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | Mistral AI | 7.2B | 4-bit | open, Apache 2.0 |
| `phi4_mini` | [microsoft/Phi-4-mini-instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct) | Microsoft (Phi) | 3.8B | 16-bit | open, MIT |
| `granite4_micro` | [ibm-granite/granite-4.0-micro](https://huggingface.co/ibm-granite/granite-4.0-micro) | IBM (Granite) | 3.4B | 16-bit | open, Apache 2.0 |
| `olmo3_7b` (optional) | [allenai/Olmo-3-7B-Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct) | AI2 (OLMo) | 7.3B | 4-bit | open, Apache 2.0 |
| `llama31_8b` (optional) | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Meta (Llama) | 8.0B | 4-bit | gated |
| `natlas` (optional) | [NCAIR1/N-ATLaS](https://huggingface.co/NCAIR1/N-ATLaS) | Awarri / NCAIR (Llama 3.1 fine-tune) | 8.0B | 4-bit | gated |

**Why these.** Five different developers give five families for the replication rule, with a mix of sizes. All of them
load with standard Hugging Face code and fit the free T4 (15 GB) one at a time.

**Considered and left out.** *Ministral 3 8B* is distributed for vLLM with Mistral's own tokenizer library, so it is
fragile in plain Transformers on a T4; Mistral 7B v0.3 represents Mistral instead. *Granite 4.0 H-Tiny* is a hybrid
Mamba mixture-of-experts model: its expert weights are not 4-bit quantised, so it barely fits, and it runs slowly
without Mamba kernels. Granite 4.0 Micro represents IBM instead. *Mistral Nemo 12B* is slow on a T4.
*OpenRouter free endpoints* were not used, because their models change behind fixed names and they are rate-limited.

### N-ATLaS: the Nigerian model (optional extension)

N-ATLaS is Llama 3.1 8B fine-tuned on English, Hausa, Igbo and Yoruba, released by Awarri with Nigeria's National
Centre for Artificial Intelligence and Robotics. Its chat template is the Llama 3.1 template, so running it next to
`llama31_8b` isolates one question: **does Nigerian fine-tuning change ethnic-identity effects?** The two share a base
model, so they count as one family for the replication rule.

N-ATLaS is gated and has its own terms of use, shown when you request access. Read them before running it. They are
reported to permit academic research and to prohibit discriminatory profiling. This study audits discrimination and
does not use the model to judge real people, but confirm with the maintainers that the study is within the terms
before you publish N-ATLaS results.

### Using a gated model (Llama 3.1, N-ATLaS)

1. Log in to Hugging Face, open the model page and request access or accept the terms. Llama access is usually granted
   within a day; N-ATLaS is approved automatically.
2. Create a **read** token at https://huggingface.co/settings/tokens.
3. Add it as a secret named `HF_TOKEN`. In Colab: the key icon in the left sidebar, then switch on notebook access.
   In Kaggle: **Add-ons > Secrets**, add the secret and tick it for this notebook.
4. Add `"llama31_8b"` and `"natlas"` to `MODELS_TO_RUN` in the Settings cell.

## Running the study on Google Colab

1. Open the notebook in Colab. Either upload it (**File > Upload notebook**, choose
   `notebooks/ethnic_audit_colab.ipynb`) or, once the repository is on GitHub, use **File > Open notebook > GitHub**.
2. **Runtime > Change runtime type > T4 GPU > Save**, then click **Connect**. Check that the top-right corner shows
   **T4**. If it shows only RAM and Disk, Colab gave you a machine without a GPU: see *Troubleshooting* below.
3. Check the **Settings** cell. The defaults run the pilot on the five ungated models.
4. **Runtime > Run all.** Approve the Google Drive prompt; results are saved there, so a disconnect loses nothing.
5. The notebook needs the three data files. If the repository on GitHub has the `data/` folder, it fetches them by
   itself. If not, an upload button appears under the *Project files* cell: choose a zip of the `data/` folder, or
   select `data/names/names.csv`, `data/names/states.csv` and `data/profiles/base_profiles_n200_seed20261109.jsonl`
   together. You can also drag them into the Files panel on the left before running.
6. At the end, a zip file of the results folder downloads automatically.

**Troubleshooting.**

| Message | Cause | Fix |
|---|---|---|
| "This Colab session has no GPU attached" | Colab connected you to a CPU machine: the free GPU allowance is used up, or no T4 was free. The menu can still say T4. | **Runtime > Disconnect and delete runtime**, choose T4 again and reconnect. If Colab says it cannot connect to a GPU, wait a few hours. |
| "The machine has a GPU but PyTorch cannot use it" | The PyTorch install does not match the GPU driver. | **Runtime > Disconnect and delete runtime**, then Run all. A plain restart keeps the broken install. |
| Upload button under *Project files* | The data files were not found. | Upload them as described in step 5. |

## Running the study on Kaggle

Kaggle gives about 30 GPU hours a week for free, and a run can continue in the background after you close the
browser. The same notebook works there unchanged.

1. Verify your phone number on Kaggle (account **Settings**). Without it, GPUs and internet access are unavailable.
2. **Create > New Notebook**, then **File > Import Notebook** and choose `notebooks/ethnic_audit_colab.ipynb`.
3. In the right-hand panel, under **Session options**: **Accelerator > GPU T4 x2** and **Internet > On**. Do not pick
   P100: it is too old for 4-bit loading, and the notebook stops with a message if it gets one.
4. Add the data, unless it is already on GitHub: **Add Input > Upload > New Dataset**, drop in a zip of the `data/`
   folder (or the three data files), name it and create it. The notebook searches all of `/kaggle/input`.
5. Run it. Either click **Run all** and keep the tab open, or use **Save Version > Save & Run All (Commit)**, which
   runs in the background for up to 12 hours.
6. Download the zip `ethnic_audit_results_<stage>_<time>.zip` from the **Output** panel, or from the version's
   *Output* tab after a commit.

On Kaggle, results are saved in `/kaggle/working/ethnic-audit-results/` (Google Drive is not used). To resume an
unfinished run, add the earlier version's output as an input (**Add Input > Your Work**, pick this notebook) and run
again: the answers recorded there are reused.

| Message | Fix |
|---|---|
| "This Kaggle session has no GPU attached" | **Accelerator > GPU T4 x2**. If it is greyed out, the weekly GPU hours are used up. |
| "No internet in this Kaggle session" | **Internet > On** (needs phone verification). |
| "... is too old for these models" | You got a P100: switch to **GPU T4 x2**. |
| "Data files not found. Add them as a Kaggle dataset" | Step 4. |

## Time and stages

**Time.** The pilot has 630 prompts per model. The first model downloads in a few minutes, and each then takes about
10 to 20 minutes on a T4, so five models need about 1 to 1.5 hours. Colab's free tier can disconnect long sessions. If
it does, run all again: answers already recorded are reused.

**Stages.**

| Stage | Profiles | Groups | Conditions | Formats | Domains | Prompts per model |
|---|---|---|---|---|---|---|
| `pilot` | 30 | Igbo, Yoruba, Hausa | C0, C1, C2 | F1, F2 | graduate trainee | 630 |
| `full` | 200 | all six plus Yoruba (Muslim names) | C0, C1, C2 | F1, F2, F3 | both | about 31,000 |

The full stage needs several Colab sessions per model. Keep `SAVE_TO_DRIVE = True` and rerun until `model_status.csv`
shows every model as complete.

**What the notebook checks before trusting a model**

- **Smoke test.** A model is skipped, with its answers printed, if fewer than 80% of its first 12 answers are valid JSON.
- **Greedy check.** Every generated token must be the model's most likely token.
- **Merit check.** Without identity, the model must score strong above borderline above weak profiles.
- **Manual reading.** `review_samples.md` holds 10 random answers per model and format. Read them.

## Outputs

The results folder (`MyDrive/ethnic-audit-results/<stage>/` in Drive, and the downloaded zip) contains:

| Path | Contents |
|---|---|
| `run_metadata.json` | Settings, design, software versions, GPU, model commits and status |
| `model_status.csv` | Per model: loaded, smoke-test parse rate, parse rate, minutes, any error |
| `trials/` | Every prompt in the design with its metadata |
| `raw/<model>.jsonl` | Every answer: text, decision probability, token counts, greedy check. The primary record; keep it. |
| `parsed/<model>.csv` | One row per prompt with the parsed answer |
| `tables/` | Quality, merit check, F1 and F2 (and F3) effects, verdicts, replication, power, identity mentions |
| `figures/` | Merit check, F1 identity effects, F2 win rates |
| `review_samples.md` | Answers to read by hand |
| `README.txt` | Explains every file |

Every contrast receives one of four verdicts: **meaningful difference**, **detectable but below SESOI**,
**equivalent (no meaningful difference)** or **inconclusive**.

## Before the full study

1. **Validate the names.** Give `data/validation/name_rating_sheet.csv` to at least five Nigerian raters (ideally
   including Ijaw, Tiv and Kanuri speakers). Mark a name `validated = yes` in `data/names/names.csv` only if at least 80%
   of raters identify the intended group. The Ijaw, Tiv and Kanuri lists need this most.
2. **Check the pilot.** Parse rates above 95%, merit check passed, and answers read by hand, for every model.
3. **Set the sample size** from `tables/power.csv`.
4. **Pre-register** hypotheses, SESOI, models (with commits), analysis and the replication rule, for example on OSF.
5. Get your supervisor's sign-off and, if your institution requires it, an ethics exemption. There are no human
   subjects and all data is synthetic, though name raters may need a short consent form.

## Run log

Record every run: date, stage, models, where it ran, and anything changed.

| Date | Stage | Models | Where | Notes |
|---|---|---|---|---|
| | | | | |

## Limitations and ethics

- The study tests general-purpose models in a simulated screening task, not any vendor's product. Do not present the
  results as evidence that any Nigerian employer or scholarship body discriminates.
- Profiles are synthetic and sparse. Real applications carry more identity cues.
- Ethnicity is partly confounded with religion (Hausa and Kanuri names) and with state characteristics (all Ijaw states
  are oil-producing). The Yoruba religion contrast addresses the first only in part.
- Names are male-coded. Gender is held constant, not studied.
- Models larger than about 5B parameters run 4-bit quantised to fit the free GPU. Effects are measured within each
  model, so this does not bias the comparison, but report it.
- A mitigation arm (the same prompts plus a fairness instruction) can be added as a separate experiment.

## Related work (check every reference before citing it)

- Bertrand and Mullainathan (2004), the original correspondence audit of name-based hiring discrimination.
- Lukk (2026), *FairFund-Bench*: audit format (rating versus ranking) can reverse the direction of LLM bias.
- Nghiem et al. (2025), *Rich Dad, Poor Lad*: socioeconomic status in LLM admissions decisions.
- Gao, Jiang and Yan (2026), *Can LLMs Hire Fairly?*: paired-resume audit of 14 LLMs with US names.
- Rozado (2025): gender and strong position bias in pairwise LLM CV comparisons.
- Tan et al. (2026), *Small Changes, Big Impact*: LLMs infer ethnicity from subtle markers in anonymised resumes.
- Rao et al. (2025), *Invisible Filters*: name substitutions alone showed no significant effects in an Indian setting.
- Nakano et al. (2024): Nigerian versus American GitHub profiles in LLM recruitment.
- Le Beux et al. (2026), *AfriStereo*: stereotype bias in LLMs for Senegal, Kenya and Nigeria.

**Gap addressed.** Earlier audits compare races or nationalities, mostly in US or Asian settings. This study compares
ethnic groups *within* one African country (all applicants Nigerian, all in English), across three audit formats and
three signal types, with a pre-specified replication rule, on open models whose behaviour anyone can reproduce exactly.
