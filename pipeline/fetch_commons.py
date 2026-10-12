"""Harvest geotagged Wikimedia Commons photos taken in Manhattan (camera location + heading + EXIF)."""
import json, re, sys, time, requests
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/raw/commons.jsonl"
API = "https://commons.wikimedia.org/w/api.php"
S = requests.Session()
S.headers["User-Agent"] = "cityviz/0.1 (https://github.com/kevmo314/cityviz; kevmo314@gmail.com)"

LAT0, LAT1, LON0, LON1 = 40.698, 40.882, -74.022, -73.906
STEP = 0.004

def api(**params):
    params.update(format="json", formatversion=2)
    for attempt in range(6):
        try:
            r = S.post(API, data=params, timeout=90)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status()
            d = r.json()
            if "error" in d:
                print("api error", d["error"].get("code"), file=sys.stderr)
                time.sleep(5 * (attempt + 1)); continue
            return d
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed {params}")

def geosearch(bbox):
    """Titles with camera coordinates in bbox (top|left|bottom|right), subdividing dense cells."""
    t, l, b, r = bbox
    d = api(action="query", list="geosearch", gsbbox=f"{t}|{l}|{b}|{r}", gslimit=500,
            gsnamespace=6, gsprimary="all", gsprop="type")
    hits = d["query"]["geosearch"]
    if len(hits) >= 500 and (t - b) > 0.0006:
        mlat, mlon = (t + b) / 2, (l + r) / 2
        out = []
        for q in [(t, l, mlat, mlon), (t, mlon, mlat, r), (mlat, l, b, mlon), (mlat, mlon, b, r)]:
            out += geosearch(q)
        return out
    return [h["title"] for h in hits if h.get("type") in (None, "camera")]

HEAD_RE = re.compile(r"\{\{\s*(?:Location|Location dec|Camera location|Camera location dec)\s*\|([^}]*)\}\}", re.I)
CARD = {"N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
        "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5}

def parse_heading(wt):
    m = HEAD_RE.search(wt)
    if not m:
        return None
    h = re.search(r"heading:\s*([A-Za-z]+|[-\d.]+)", m.group(1))
    if not h:
        return None
    v = h.group(1)
    if v.upper() in CARD:
        return CARD[v.upper()]
    try:
        return float(v) % 360
    except ValueError:
        return None

def frac(v):
    if v is None: return None
    if isinstance(v, (int, float)): return float(v)
    try:
        if "/" in v:
            a, b = v.split("/"); return float(a) / float(b) if float(b) else None
        return float(v)
    except ValueError:
        return None

def details(titles):
    try:
        return _details(titles)
    except RuntimeError as e:
        print("skip batch", titles[0], file=sys.stderr)
        return []

def _details(titles):
    d = api(action="query", titles="|".join(titles), prop="coordinates|revisions|imageinfo",
            coprop="type|dim", coprimary="all", colimit="max", rvprop="content", rvslots="main",
            iiprop="url|extmetadata|commonmetadata|size|mime", iiurlwidth=1280)
    out = []
    for p in d["query"]["pages"]:
        if "imageinfo" not in p: continue
        ii = p["imageinfo"][0]
        if ii.get("mime") not in ("image/jpeg", "image/png", "image/webp"): continue
        cam = [c for c in p.get("coordinates", []) if c.get("type") == "camera"]
        if not cam: continue
        wt = p["revisions"][0]["slots"]["main"]["content"] if p.get("revisions") else ""
        cm = {m["name"]: m["value"] for m in ii.get("commonmetadata", []) if not isinstance(m["value"], list)}
        em = {k: v.get("value") for k, v in ii.get("extmetadata", {}).items()}
        heading = parse_heading(wt)
        if heading is None and cm.get("GPSImgDirection") is not None:
            heading = frac(cm["GPSImgDirection"])
        out.append({
            "title": p["title"], "pageid": p["pageid"], "lat": cam[0]["lat"], "lon": cam[0]["lon"],
            "heading": heading, "width": ii["width"], "height": ii["height"],
            "thumb": ii.get("thumburl"), "url": ii["descriptionurl"],
            "focal": frac(cm.get("FocalLength")), "focal35": frac(cm.get("FocalLengthIn35mmFilm")),
            "model": cm.get("Model"), "make": cm.get("Make"), "orientation": cm.get("Orientation"),
            "date": em.get("DateTimeOriginal") or em.get("DateTime"),
            "artist": re.sub(r"<[^>]+>", "", em.get("Artist") or "").strip(),
            "license": em.get("LicenseShortName"), "license_url": em.get("LicenseUrl"),
            "categories": re.findall(r"\[\[Category:([^\]|]+)", wt)[:30],
        })
    return out

def main():
    cells = []
    lat = LAT0
    while lat < LAT1:
        lon = LON0
        while lon < LON1:
            cells.append((round(lat + STEP, 5), round(lon, 5), round(lat, 5), round(lon + STEP, 5)))
            lon += STEP
        lat += STEP
    cache = ROOT / "data/raw/commons_titles.txt"
    if cache.exists():
        titles = cache.read_text().splitlines()
    else:
        titles = set()
        with ThreadPoolExecutor(4) as ex:
            for i, ts in enumerate(ex.map(geosearch, cells)):
                titles.update(ts)
                if i % 50 == 0: print(f"cells {i}/{len(cells)} titles {len(titles)}", file=sys.stderr)
        titles = sorted(titles)
        cache.write_text("\n".join(titles))
    print(f"{len(titles)} titles", file=sys.stderr)
    done = set()
    if OUT.exists():
        done = {json.loads(l)["title"] for l in OUT.open()}
        last = max(done) if done else ""
        titles = [t for t in titles if t > last]
    batches = [titles[i:i + 50] for i in range(0, len(titles), 50)]
    n = len(done)
    with open(OUT, "a") as f, ThreadPoolExecutor(4) as ex:
        for i, recs in enumerate(ex.map(details, batches)):
            for r in recs:
                f.write(json.dumps(r) + "\n"); n += 1
            if i % 20 == 0: print(f"batch {i}/{len(batches)} records {n}", file=sys.stderr)

if __name__ == "__main__":
    main()
