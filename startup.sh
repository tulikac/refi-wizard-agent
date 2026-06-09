#!/bin/bash
# Embr startup script — launches gunicorn.
# All secrets/config come from Embr project variables, not from this file.

# Clear stale DB so seed data refreshes on code changes
rm -f homes.db

exec gunicorn --bind 0.0.0.0:8080 --workers 2 --timeout 120 app:app
