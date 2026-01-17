#!/bin/sh
set -e

echo "Waiting for database to be ready..."
sleep 3

echo "Running database migrations..."
npm run db:push

echo "Starting application..."
exec "$@"
