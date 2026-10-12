FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy scipy shapely pyproj trimesh mapbox_earcut embreex pillow requests \
      transformers opencv-python-headless google-cloud-storage
RUN python -c "from transformers import SegformerForSemanticSegmentation as M, SegformerImageProcessor as P; \
n='nvidia/segformer-b4-finetuned-ade-512-512'; M.from_pretrained(n); P.from_pretrained(n)"
WORKDIR /app
COPY pipeline/ pipeline/
COPY data/work/city_mesh.npz data/work/heightfield.npz data/work/candidates.jsonl data/work/
ENV PYTHONUNBUFFERED=1 SEG_THREADS=3
CMD ["python", "pipeline/cloud_job.py"]
