#!/bin/sh
set -e

echo "============================================"
echo "  DDoS Detection ML - Starting up..."
echo "============================================"

echo "[1/3] Waiting for PostgreSQL database..."
max_retries=30
count=0
until pg_isready -h db -U postgres -d ddos_detection > /dev/null 2>&1 || [ $count -eq $max_retries ]; do
  count=$((count + 1))
  echo "  Waiting for database... ($count/$max_retries)"
  sleep 2
done

if [ $count -eq $max_retries ]; then
  echo "ERROR: Database connection timeout!"
  exit 1
fi

echo "  Database is ready!"

echo "[2/3] Running database migrations..."
npm run db:push || {
  echo "  Migration failed, retrying with force..."
  npm run db:push -- --force
}
echo "  Database schema updated!"

echo "[3/3] Starting application server..."
echo "============================================"
echo "  App running at http://localhost:5000"
echo "============================================"

exec "$@"
