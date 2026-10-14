$ErrorActionPreference = 'Stop'
$pidFile = Join-Path $PSScriptRoot '.service_runtime\server.pid'
if (-not (Test-Path -LiteralPath $pidFile)) { Write-Host 'Нет файла процесса сервиса'; exit 0 }
$serviceProcessId = [int](Get-Content -LiteralPath $pidFile -Raw)
$serviceProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $serviceProcessId"
if ($serviceProcess -and $serviceProcess.CommandLine -like '*src.semantic_service.server*') {
    Stop-Process -Id $serviceProcessId
    Write-Host 'Локальный сервис остановлен'
} elseif ($serviceProcess) {
    throw 'PID принадлежит другому процессу; остановка отменена'
}
Remove-Item -LiteralPath $pidFile -ErrorAction SilentlyContinue
