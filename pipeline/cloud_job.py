"""Cloud Run Job entry point: solve a shard of photo candidates and write results + images to GCS.

Each task takes candidates[TASK_INDEX::TASK_COUNT]. Results are appended per task so retries resume.
"""
import json, os, sys, time
from multiprocessing import Pool
from pathlib import Path

from google.cloud import storage

sys.path.insert(0, str(Path(__file__).parent))
import localize_photos as lp

BUCKET = os.environ["BUCKET"]
PREFIX = os.environ.get("PREFIX", "run1")
IDX = int(os.environ.get("CLOUD_RUN_TASK_INDEX", "0"))
CNT = int(os.environ.get("CLOUD_RUN_TASK_COUNT", "1"))


def main():
    bucket = storage.Client().bucket(BUCKET)
    todo = [json.loads(l) for l in open(Path(__file__).parent.parent / "data/work/candidates.jsonl")][IDX::CNT]
    blob = bucket.blob(f"{PREFIX}/results/task-{IDX:04d}.jsonl")
    lines = blob.download_as_text().splitlines() if blob.exists() else []
    done = {json.loads(l)["pageid"] for l in lines}
    todo = [r for r in todo if r["pageid"] not in done]
    print(f"task {IDX}/{CNT}: {len(todo)} to do, {len(done)} done", flush=True)
    workers = max(1, (os.cpu_count() or 2) // int(os.environ.get("SEG_THREADS", "3")))
    last = time.time()
    with Pool(workers, initializer=lp.init_worker) as pool:
        for r in pool.imap_unordered(lp.solve, todo):
            lines.append(json.dumps(r))
            img = lp.IMGDIR / f"{r['pageid']}.jpg"
            if r.get("status") == "ok" and img.exists():
                bucket.blob(f"{PREFIX}/photos/{r['pageid']}.jpg").upload_from_filename(str(img))
            if time.time() - last > 60:
                blob.upload_from_string("\n".join(lines) + "\n")
                last = time.time()
    blob.upload_from_string("\n".join(lines) + "\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()
