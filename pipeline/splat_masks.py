"""Per-photo supervision masks for splat training: 0 = transient (people, vehicles; not supervised), 1 = static scene,
2 = sky (supervised to stay transparent so no floaters form in front of the background).

usage: splat_masks.py <name>   -> data/work/splat/<name>/masks/<image stem>.png (from the undistorted images)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
import segment

ROOT = Path(__file__).resolve().parent.parent
# ADE20k: person, car, bus, truck, van, motorbike, bicycle, boat, animal, flag
TRANSIENT = [12, 20, 80, 83, 102, 116, 127, 76, 126, 149]


def main():
    d = ROOT / f"data/work/splat/{sys.argv[1]}"
    out = d / "masks"
    out.mkdir(exist_ok=True)
    seg = segment.Segmenter()
    for f in sorted(d.glob("undist/*/images/*.jpg")):
        dst = out / f"{f.stem}.png"
        if dst.exists():
            continue
        ade = seg.probs(Image.open(f), size=768).argmax(0)
        m = np.ones(ade.shape, np.uint8)
        m[np.isin(ade, segment._SKY)] = 2
        m[np.isin(ade, TRANSIENT)] = 0
        Image.fromarray(m).save(dst)
    print("masks done")


if __name__ == "__main__":
    main()
