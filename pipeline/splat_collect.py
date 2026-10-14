"""Collect every Wikimedia Commons photo of one scene for Gaussian splatting: geotagged photos near a point plus the
scene's category tree (most landmark photos carry no coordinates; SfM registers them anyway).

usage: splat_collect.py <name> <lon> <lat> <radius_m> [Category:Name ...]
writes data/work/splat/<name>/images/<pageid>.jpg and meta.jsonl (title, url, artist, license, date)
"""
import json, re, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_commons import S, api

ROOT = Path(__file__).resolve().parent.parent
PX = 1600
SKIP = re.compile(r"(interior|inside|lobby|map|plan|diagram|logo|drawing|painting|postcard|\.svg|\.tif|\.pdf|\.gif)", re.I)


def category_files(cat, depth=2, seen=None):
    seen = seen if seen is not None else set()
    if cat in seen:
        return set()
    seen.add(cat)
    files, cont = set(), {}
    while True:
        d = api(action="query", list="categorymembers", cmtitle=cat, cmlimit=500, cmtype="file|subcat", **cont)
        for m in d["query"]["categorymembers"]:
            if m["ns"] == 6:
                files.add(m["title"])
            elif m["ns"] == 14 and depth > 0 and not SKIP.search(m["title"]):
                files |= category_files(m["title"], depth - 1, seen)
        if "continue" not in d:
            return files
        cont = d["continue"]


def near(lon, lat, radius):
    d = api(action="query", list="geosearch", gscoord=f"{lat}|{lon}", gsradius=min(int(radius), 10000), gslimit=500,
            gsnamespace=6, gsprimary="all")
    return {h["title"] for h in d["query"]["geosearch"]}


def info(titles):
    d = api(action="query", prop="imageinfo", titles="|".join(titles), iiprop="url|size|mime|extmetadata",
            iiurlwidth=PX)
    out = []
    for p in d["query"]["pages"]:
        ii = (p.get("imageinfo") or [None])[0]
        if not ii or ii.get("mime") != "image/jpeg" or min(ii["width"], ii["height"]) < 900:
            continue
        em = ii.get("extmetadata", {})
        g = lambda k: re.sub(r"<[^>]+>", "", em.get(k, {}).get("value", ""))[:200]
        out.append(dict(pageid=p["pageid"], title=p["title"], thumb=ii.get("thumburl") or ii["url"], url=ii["descriptionurl"],
                        artist=g("Artist"), license=g("LicenseShortName"), date=g("DateTimeOriginal")))
    return out


def download(rec, out):
    dst = out / f"{rec['pageid']}.jpg"
    if dst.exists():
        return True
    for k in range(4):
        try:
            r = S.get(rec["thumb"], timeout=60)
            if r.status_code == 429:
                time.sleep(5 * (k + 1)); continue
            r.raise_for_status()
            dst.write_bytes(r.content)
            return True
        except Exception:
            time.sleep(2 * (k + 1))
    return False


def main():
    name, lon, lat, radius, *cats = sys.argv[1:]
    out = ROOT / f"data/work/splat/{name}"
    (out / "images").mkdir(parents=True, exist_ok=True)
    titles = near(float(lon), float(lat), float(radius))
    for c in cats:
        titles |= category_files(c)
    titles = sorted(t for t in titles if not SKIP.search(t))
    print(len(titles), "titles", flush=True)
    recs = []
    for i in range(0, len(titles), 50):
        recs += info(titles[i:i + 50])
    print(len(recs), "usable jpegs", flush=True)
    with ThreadPoolExecutor(4) as ex:
        ok = list(ex.map(lambda r: download(r, out / "images"), recs))
    recs = [r for r, g in zip(recs, ok) if g]
    with open(out / "meta.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(len(recs), "downloaded")


if __name__ == "__main__":
    main()
