#!/bin/sh
# Sync Cloud Run job outputs (results + solved photos) into data/work/cloud/.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/work/cloud
gcloud storage rsync -r "gs://${BUCKET:-kevmo314-cityviz}/${PREFIX:-run1}" "data/work/cloud/${PREFIX:-run1}" --quiet
