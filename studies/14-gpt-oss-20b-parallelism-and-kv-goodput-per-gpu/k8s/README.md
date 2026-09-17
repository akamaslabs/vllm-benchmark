# k8s/ — snapshot, not the live files

The Akamas workflow this study runs (`13-GPT-OSS-20B-TP-Goodput-Per-GPU-Workflow`, owned
by study 13 — see `../akamas/README.md`) references absolute paths on the toolbox host:

```
/work/vllm-benchmark/studies/13-gpt-oss-20b-tp-goodput-per-gpu/k8s/01-deployment_template.yaml
/work/vllm-benchmark/studies/13-gpt-oss-20b-tp-goodput-per-gpu/k8s/apply_config.sh
/work/vllm-benchmark/studies/13-gpt-oss-20b-tp-goodput-per-gpu/k8s/run_test_goodput.sh
```

So **the files in this folder are never read by anything**. They are a byte-identical copy
taken on 2026-09-16, kept so that this study's folder records the stack that actually ran
even if study 13's folder is later changed. `../infra/` is a snapshot for the same reason.

To change what the study executes you must edit study 13's copies **and** accept that this
also changes study 13, or create a new workflow pointing at this folder — which would make
the imported experiments no longer comparable, defeating the purpose of the import.
