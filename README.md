# Character inference on emergent-misalignment training data

A reader model sees each training exchange from an EM dataset and describes the character behind the reply.

## Repository structure

```
config.py            every setting: datasets, readers, prompts, sampling, analysis, seeds
run_vllm.sh          fetch + select -> read with vLLM -> backup (GPU machine)
charinf/             the package
  data.py              fetching, parsing, selection, hashing
  datasets.py          dataset registry helpers (--datasets, pairs, contrasts)
  prompts.py           the reading prompts and chat rendering
  readers/             vllm.py (Qwen, local GPU), openai.py (GPT-4.1 API), base.py (request plan, record schema)
  flags.py             validity, trait-list parsing, copy check
  stats.py             loading, distances, bootstrap, PCA, PCA corner answers
  clusters.py          top trait clusters, label checks
  embed.py, meta.py, plotting.py, text.py
scripts/
  setup/               setup_vllm.sh, verify_model.py
  data/                fetch_and_select.py
  generate/            read.py, backup.py
  analysis/            flags.py, diversity.py, joint_pca.py, top_clusters.py, compare_readers.py
  plots/               diversity_plane.py, joint_pca.py, top_clusters.py, compare_readers.py
results/             data/ (selected row indices, data checks); <reader>/ and compare/ are written here (gitignored)
figures/             <reader>/ and compare/ (plots; gitignored)
tools/               check_leak.py (n-gram leak check), view_readings.py (local viewer)
tests/
```

## Setup

```
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements-local.txt
```

Then create a `.env` file (see `.env.example`) with the variables you need: `OPENAI_API_KEY` for GPT-4.1, and
`EM_MO_PASSWORD` / `EM_WANG_PASSWORD` (from each dataset's README) only to fetch those restricted datasets.

vLLM readers (the Qwen models) need a Linux machine with an 80 GB GPU: `scripts/setup/setup_vllm.sh` installs
`requirements-vllm.txt` and fetches the data there, under `$WORKSPACE` (default `/workspace`). GPT-4.1 runs anywhere
through the API. Data, readings and caches live in `em_raw/` (gitignored), or wherever `EM_RAW_DIR` points.

## How to run

Options: `--readers` (comma list of `config.READERS`: `qwen_instruct`, `qwen_coder`, `gpt41`), `--datasets` (`all`
or a comma list of `config.DATASETS`); for reading also `--n-examples` (seeded subset of the 100), `--n-samples`
(default 5), `--prompts` (default `trait_list,nohedge`; add `free` for the bare question; `trait_list` is required).

The reading prompt is always the only user message: Qwen readers get their chat template's default system prompt,
GPT-4.1 none.

```
# 1. Fetch the datasets and select the examples
python scripts/data/fetch_and_select.py --datasets all --fetch-mo              # needs both dataset passwords
python scripts/data/fetch_and_select.py --datasets insecure,secure,educational # code only: public data, no passwords

# 2a. vLLM readers, on the GPU machine; then copy $EM_RAW_DIR/backup/<reader>/ there to em_raw/backup/<reader>/ here
bash scripts/setup/setup_vllm.sh insecure,secure,educational
python scripts/setup/verify_model.py --reader qwen_coder
bash run_vllm.sh qwen_coder insecure,secure,educational

# 2b. GPT-4.1 (OpenAI API): runs locally, no GPU needed; then bundle
python scripts/generate/read.py --readers gpt41 --datasets insecure,secure,educational
python scripts/generate/backup.py --readers gpt41

# 3. Analyse and plot
for s in analysis/flags analysis/diversity analysis/joint_pca analysis/top_clusters \
         plots/diversity_plane plots/joint_pca plots/top_clusters; do
  python scripts/$s.py --readers qwen_coder
done

# 4. Compare readers on the datasets they share
python scripts/analysis/compare_readers.py --readers qwen_instruct,qwen_coder,gpt41 --datasets insecure,secure
python scripts/plots/compare_readers.py
```

The commands show the Qwen2.5-Coder run on the code datasets; for Qwen2.5-32B-Instruct on every dataset use
`qwen_instruct` and `all` instead. `joint_pca.py --datasets … --suffix name` (and the plot with the same `--suffix`) fits the PCA on a
subset without replacing the main outputs (e.g. `--suffix no_aesthetic` without unpop and popular).
Read examples and readings locally with `tools/view_readings.py`.

Training data and model readings are not distributed. The datasets are fetched from their sources: Betley et al.,
Model Organisms for EM, Wang et al. persona features, and AestheticEM.
