param(
  [string]$Model = $env:OPENAI_MODEL,
  [int]$MaxQueries = 0,
  [string]$OutputDir = "outputs/batch_inputs"
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

if (-not $Model) {
  $Model = "gpt-5.5-mini"
}

$python = if (Test-Path ".venv/Scripts/python.exe") { ".venv/Scripts/python.exe" } else { "python" }
$args = @(
  "-m", "task4_rag.src.prepare_batch",
  "--model", $Model,
  "--output-dir", $OutputDir
)
if ($MaxQueries -gt 0) {
  $args += @("--max-queries", "$MaxQueries")
}

& $python @args
