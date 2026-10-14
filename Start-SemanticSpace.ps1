param(
    [int]$Port = 8766,
    [switch]$NoBrowser
)
$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$runtimeRoot = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies'
$pythonCandidates = @(
    $env:SEMANTIC_PYTHON,
    (Join-Path $projectRoot '.venv\Scripts\python.exe'),
    (Join-Path $projectRoot 'venv\Scripts\python.exe'),
    (Join-Path $env:USERPROFILE 'progects\Politic_parties\venv\Scripts\python.exe')
)
$pythonPath = $pythonCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
if (-not $pythonPath) {
    throw 'Не найден Python с зависимостями. Создайте .venv, установите requirements-service.txt или задайте SEMANTIC_PYTHON.'
}
if (-not $env:SEMANTIC_EXPORT_PYTHON) {
    $exportPython = Join-Path $runtimeRoot 'python\python.exe'
    if (Test-Path -LiteralPath $exportPython) { $env:SEMANTIC_EXPORT_PYTHON = $exportPython }
}
if (-not $env:SEMANTIC_NODE) {
    $nodePath = Join-Path $runtimeRoot 'node\bin\node.exe'
    if (Test-Path -LiteralPath $nodePath) { $env:SEMANTIC_NODE = $nodePath }
}
if (-not $env:SEMANTIC_NODE_MODULES) {
    $modulePath = Join-Path $runtimeRoot 'node\node_modules'
    if (Test-Path -LiteralPath $modulePath) { $env:SEMANTIC_NODE_MODULES = $modulePath }
}
$runtimeDirectory = Join-Path $projectRoot '.service_runtime'
New-Item -ItemType Directory -Path $runtimeDirectory -Force | Out-Null
$localUrl = 'http://127.0.0.1:' + $Port
try {
    $response = Invoke-WebRequest -Uri ($localUrl + '/api/config') -TimeoutSec 2
    if ($response.StatusCode -eq 200) {
        Write-Host ('Сервис уже работает: ' + $localUrl)
        if (-not $NoBrowser) { Start-Process $localUrl }
        exit 0
    }
} catch { }
$env:PYTHONIOENCODING = 'utf-8'
$process = Start-Process -FilePath $pythonPath -ArgumentList @('-B','-m','src.semantic_service.server','--port',"$Port") -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtimeDirectory 'server.log') -RedirectStandardError (Join-Path $runtimeDirectory 'server-error.log')
$process.Id | Set-Content -LiteralPath (Join-Path $runtimeDirectory 'server.pid')
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Milliseconds 400
    try {
        $response = Invoke-WebRequest -Uri ($localUrl + '/api/config') -TimeoutSec 2
        if ($response.StatusCode -eq 200) {
            Write-Host ('Открыт локальный сервис: ' + $localUrl)
            if (-not $NoBrowser) { Start-Process $localUrl }
            exit 0
        }
    } catch { }
    if ($process.HasExited) { break }
}
throw ('Сервис не запустился. Проверьте ' + (Join-Path $runtimeDirectory 'server-error.log'))
