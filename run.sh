#!/usr/bin/env bash
# Levanta todo el sistema con una sola instrucción:  ./run.sh
#
#   1. Verifica Docker y el .env (si no existe, lo crea desde .env.example y
#      dice qué completar).
#   2. docker compose up -d --build   (redis + phoenix + api + worker)
#   3. Espera a que la API responda y a que el worker haya abierto el RAG
#      (la primera vez indexa la documentación en Pinecone: ~20 s).
#
# Opciones:
#   ./run.sh --down     apaga el stack (los datos quedan en los volúmenes)
#   ./run.sh --logs     sigue los logs de la API y el worker
set -euo pipefail
cd "$(dirname "$0")"

API="http://localhost:8000"
PHOENIX="http://localhost:6006"

case "${1:-}" in
  --down) docker compose down; exit 0 ;;
  --logs) docker compose logs -f api worker; exit 0 ;;
esac

command -v docker >/dev/null || { echo "✗ Falta Docker: https://docs.docker.com/get-docker/"; exit 1; }
docker info >/dev/null 2>&1 || { echo "✗ Docker no está corriendo. Abrí Docker Desktop y volvé a intentar."; exit 1; }

if [ ! -f .env ]; then
  cp .env.example .env
  echo "✗ No había .env: lo creé desde .env.example."
  echo "  Completá OPENAI_API_KEY y PINECONE_API_KEY en .env y volvé a correr ./run.sh"
  exit 1
fi

valor() { grep -E "^$1=" .env | tail -n1 | cut -d= -f2- | tr -d '"'"'"' \r'; }
faltan=()
[ -n "$(valor OPENAI_API_KEY)" ] || faltan+=("OPENAI_API_KEY")
if [ "$(valor RAG_MODO)" != "lexico" ]; then
  [ -n "$(valor PINECONE_API_KEY)" ] || faltan+=("PINECONE_API_KEY (o RAG_MODO=lexico para correr sin Pinecone)")
fi
if [ "$(valor PROVEEDOR)" = "anthropic" ]; then
  [ -n "$(valor ANTHROPIC_API_KEY)" ] || faltan+=("ANTHROPIC_API_KEY")
fi
if [ ${#faltan[@]} -gt 0 ]; then
  echo "✗ Faltan claves en .env:"; printf '    - %s\n' "${faltan[@]}"; exit 1
fi

echo "▶ docker compose up -d --build"
docker compose up -d --build

echo -n "▶ Esperando a Phoenix, la API y el worker (RAG)"
for _ in $(seq 1 90); do
  salud="$(curl -fsS "$API/health" 2>/dev/null || true)"
  if [ -n "$salud" ] && ! grep -q '"sin worker"' <<<"$salud" && curl -fsS "$PHOENIX/healthz" >/dev/null 2>&1; then
    echo; echo "✓ Listo: $salud"
    echo
    echo "  API (Swagger):  $API/docs"
    echo "  Phoenix:        $PHOENIX   (proyecto: $(valor PHOENIX_PROJECT_NAME || true))"
    echo
    echo "  Probar:         python scripts/pruebas_sistema.py"
    echo "  Logs:           ./run.sh --logs        Apagar: ./run.sh --down"
    exit 0
  fi
  echo -n "."; sleep 2
done

echo; echo "✗ El sistema no quedó listo en 3 minutos. Últimos logs del worker:"
docker compose logs --tail 40 worker
exit 1
