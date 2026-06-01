param(
  [string]$Model = $env:OPENAI_MODEL,
  [int]$MaxQueries = 0
)

$ErrorActionPreference = "Stop"

function Import-DotEnv {
  param([string]$Path)
  if (-not (Test-Path $Path)) { return }
  Get-Content $Path | ForEach-Object {
    $line = $_.Trim()
    if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) { return }
    $parts = $line.Split("=", 2)
    $key = $parts[0].Trim()
    if ($key.StartsWith("export ")) {
      $key = $key.Substring(7).Trim()
    }
    $value = $parts[1].Trim().Trim('"').Trim("'")
    if ($key -and -not [Environment]::GetEnvironmentVariable($key, "Process")) {
      [Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
  }
}

Import-DotEnv ".env.local"
Import-DotEnv ".env"

if (-not $env:OPENAI_API_KEY) {
  throw "OPENAI_API_KEY is not set. Put it in .env.local or export it in the shell."
}
if (-not $Model) {
  $Model = "gpt-5.5-mini"
}

$queries = ".cache/longeval-sci-2026/task4_longeval_rag-query_docids.jsonl"
$documents = ".cache/longeval-sci-2026/longeval_sci_test-09-11_2026_fulltext/data/processed/doc_collection_parallel_09032026_parallel_2/snapshot-3/longeval_sci_test-09-11_2026_fulltext/documents"
$python = if (Test-Path ".venv/Scripts/python.exe") { ".venv/Scripts/python.exe" } else { "python" }

$experiments = @(
  @{ Config = "task4_rag/configs/task4_concat_baseline.yaml"; RunId = "concat_baseline_openai_llm_v1"; Output = "outputs/runs/concat_baseline_openai_llm_v1.jsonl" },
  @{ Config = "task4_rag/configs/task4_single_query_bm25.yaml"; RunId = "single_query_bm25_openai_llm_v1"; Output = "outputs/runs/single_query_bm25_openai_llm_v1.jsonl" },
  @{ Config = "task4_rag/configs/task4_rrf_no_rerank.yaml"; RunId = "rrf_no_rerank_openai_llm_v1"; Output = "outputs/runs/rrf_no_rerank_openai_llm_v1.jsonl" },
  @{ Config = "task4_rag/configs/task4_rrf_rerank.yaml"; RunId = "caes_rag_rrf_openai_llm_v1"; Output = "outputs/runs/caes_rag_rrf_openai_llm_v1.jsonl" },
  @{ Config = "task4_rag/configs/task4_rule_minilm_rerank.yaml"; RunId = "rule_minilm_openai_llm_v1"; Output = "outputs/runs/rule_minilm_openai_llm_v1.jsonl" },
  @{ Config = "task4_rag/configs/task4_semantic_current_rerank.yaml"; RunId = "semantic_current_openai_llm_v1"; Output = "outputs/runs/semantic_current_openai_llm_v1.jsonl" },
  @{ Config = "task4_rag/configs/task4_topic_shift_current_rerank.yaml"; RunId = "topic_shift_current_openai_llm_v1"; Output = "outputs/runs/topic_shift_current_openai_llm_v1.jsonl" },
  @{ Config = "task4_rag/configs/task4_semantic_minilm_rerank.yaml"; RunId = "semantic_minilm_openai_llm_v1"; Output = "outputs/runs/semantic_minilm_openai_llm_v1.jsonl" },
  @{ Config = "task4_rag/configs/task4_topic_shift_minilm_rerank.yaml"; RunId = "topic_shift_minilm_openai_llm_v1"; Output = "outputs/runs/topic_shift_minilm_openai_llm_v1.jsonl" },
  @{ Config = "task4_rag/configs/task4_default.yaml"; RunId = "default_openai_llm_v1"; Output = "outputs/runs/default_openai_llm_v1.jsonl" }
)

foreach ($experiment in $experiments) {
  $args = @(
    "-m", "task4_rag.src.run_task4",
    "--config", $experiment.Config,
    "--queries", $queries,
    "--documents", $documents,
    "--output", $experiment.Output,
    "--run-id", $experiment.RunId,
    "--provider", "openai",
    "--model", $Model,
    "--temperature", "0"
  )
  if ($MaxQueries -gt 0) {
    $args += @("--max-queries", "$MaxQueries")
  }
  Write-Host "Running $($experiment.RunId) with $Model"
  & $python @args
}
