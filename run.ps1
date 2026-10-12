# Levanta todo el sistema con una sola instrucción (Windows PowerShell):  .\run.ps1
#
#   1. Verifica Docker y el .env (si no existe, lo crea desde .env.example y
#      dice qué completar).
#   2. docker compose up -d --build   (redis + phoenix + api + worker)
#   3. Espera a que la API responda y a que el worker haya abierto el RAG.
#
# Opciones:   .\run.ps1 -Down     apaga el stack (los datos quedan en los volúmenes)
#             .\run.ps1 -Logs     sigue los logs de la API y el worker
param([switch]$Down, [switch]$Logs)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Api = "http://localhost:8000"
$Phoenix = "http://localhost:6006"

if ($Down) { docker compose down; exit 0 }
if ($Logs) { docker compose logs -f api worker; exit 0 }

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { Write-Host "x Falta Docker: https://docs.docker.com/get-docker/"; exit 1 }
docker info *> $null
if ($LASTEXITCODE -ne 0) { Write-Host "x Docker no está corriendo. Abrí Docker Desktop y volvé a intentar."; exit 1 }

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "x No había .env: lo creé desde .env.example."
    Write-Host "  Completá OPENAI_API_KEY y PINECONE_API_KEY en .env y volvé a correr .\run.ps1"
    exit 1
}

function Valor([string]$nombre) {
    $linea = Get-Content .env | Where-Object { $_ -match "^$nombre=" } | Select-Object -Last 1
    if ($null -eq $linea) { return "" }
    return ($linea -replace "^$nombre=", "").Trim().Trim('"').Trim("'")
}
$faltan = @()
if (-not (Valor "OPENAI_API_KEY")) { $faltan += "OPENAI_API_KEY" }
if ((Valor "RAG_MODO") -ne "lexico" -and -not (Valor "PINECONE_API_KEY")) { $faltan += "PINECONE_API_KEY (o RAG_MODO=lexico para correr sin Pinecone)" }
if ((Valor "PROVEEDOR") -eq "anthropic" -and -not (Valor "ANTHROPIC_API_KEY")) { $faltan += "ANTHROPIC_API_KEY" }
if ($faltan.Count -gt 0) { Write-Host "x Faltan claves en .env:"; $faltan | ForEach-Object { Write-Host "    - $_" }; exit 1 }

Write-Host "> docker compose up -d --build"
docker compose up -d --build
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host -NoNewline "> Esperando a Phoenix, la API y el worker (RAG)"
for ($i = 0; $i -lt 90; $i++) {
    try {
        $salud = Invoke-RestMethod "$Api/health" -TimeoutSec 3
        Invoke-WebRequest "$Phoenix/healthz" -UseBasicParsing -TimeoutSec 3 | Out-Null
        if ($salud.rag -ne "sin worker") {
            Write-Host ""
            Write-Host "OK Listo: api=$($salud.api) redis=$($salud.redis) checkpointer=$($salud.checkpointer) rag=$($salud.rag)"
            Write-Host ""
            Write-Host "  API (Swagger):  $Api/docs"
            Write-Host "  Phoenix:        $Phoenix   (proyecto: $(Valor 'PHOENIX_PROJECT_NAME'))"
            Write-Host ""
            Write-Host "  Probar:         python scripts/pruebas_sistema.py"
            Write-Host "  Logs:           .\run.ps1 -Logs        Apagar: .\run.ps1 -Down"
            exit 0
        }
    } catch { }
    Write-Host -NoNewline "."
    Start-Sleep -Seconds 2
}
Write-Host ""
Write-Host "x El sistema no quedó listo en 3 minutos. Últimos logs del worker:"
docker compose logs --tail 40 worker
exit 1
