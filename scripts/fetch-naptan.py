#!/usr/bin/env python3
"""Download NaPTAN-schema XML feeds (GB DfT and IE TFI) and emit per-prefix chunks.

For each StopPoint and StopArea in each feed, emit a record:
    {"id": ..., "name": ..., "lat": ..., "lon": ..., "type": ..., "xml": ...}
grouped by the first 3 characters of the ATCO / StopArea code into
public/data/<region>/<prefix>.json.gz.

GB and IE share the same NaPTAN XML schema; only the URL and the
grid-coord fallback CRS differ (OSGB36 vs Irish Grid). The frontend
decompresses chunks with the browser's native DecompressionStream API.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import requests
from lxml import etree
from pyproj import Transformer

NS = "http://www.naptan.org.uk/"
NS_MAP = {"n": NS}

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "public" / "data"


@dataclass(frozen=True)
class Source:
    name: str         # subdirectory under public/data/
    url: str
    cache: str        # local filename for the downloaded XML
    grid_crs: str     # EPSG code for Easting/Northing fallback


SOURCES = [
    Source(
        name="gb",
        url="https://naptan.api.dft.gov.uk/v1/access-nodes?dataFormat=xml",
        cache="naptan-gb.xml",
        grid_crs="EPSG:27700",  # OSGB36 British National Grid
    ),
    Source(
        name="ie",
        # Mostly Republic of Ireland with some cross-border NI stops.
        url="https://www.transportforireland.ie/transitData/Data/NaPTAN.xml",
        cache="naptan-ie.xml",
        grid_crs="EPSG:29903",  # TM75 Irish Grid (may need revising once we see real data)
    ),
]


def download(url: str, dest: Path, attempts: int = 5) -> None:
    """Stream-download to disk with exponential backoff.

    Reuses an existing >5 MB file at `dest` so local iteration doesn't
    re-hit the origin. CI runners always start with a clean checkout.
    """
    if dest.exists() and dest.stat().st_size > 5_000_000:
        print(f"reusing cached {dest.name} ({dest.stat().st_size // 1_000_000} MB)")
        return
    for attempt in range(1, attempts + 1):
        try:
            print(f"GET {url} (attempt {attempt}/{attempts})", flush=True)
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
            return
        except (requests.RequestException, OSError) as exc:
            wait = min(60, 2 ** attempt)
            print(f"  attempt {attempt} failed: {exc}; sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise SystemExit(f"failed to download {url} after {attempts} attempts")


def _coords(loc_el, transformer: Transformer) -> tuple[float | None, float | None]:
    """Return (lat, lon) — fall back to grid → WGS84 when lat/lon absent."""
    if loc_el is None:
        return None, None
    container = loc_el.find("n:Translation", NS_MAP)
    if container is None:
        container = loc_el

    lat_text = container.findtext("n:Latitude", namespaces=NS_MAP)
    lon_text = container.findtext("n:Longitude", namespaces=NS_MAP)
    if lat_text and lon_text:
        try:
            return float(lat_text), float(lon_text)
        except ValueError:
            pass

    east_text = container.findtext("n:Easting", namespaces=NS_MAP)
    north_text = container.findtext("n:Northing", namespaces=NS_MAP)
    if east_text and north_text:
        try:
            e, n = float(east_text), float(north_text)
        except ValueError:
            return None, None
        lon, lat = transformer.transform(e, n)
        return round(lat, 6), round(lon, 6)

    return None, None


def _stop_point_record(el, transformer: Transformer) -> dict | None:
    if el.get("Status") == "del":
        return None
    atco = el.findtext("n:AtcoCode", namespaces=NS_MAP)
    if not atco:
        return None
    return {
        "id": atco,
        "name": el.findtext("n:Descriptor/n:CommonName", namespaces=NS_MAP) or "",
        "lat": (coords := _coords(el.find("n:Place/n:Location", NS_MAP), transformer))[0],
        "lon": coords[1],
        "type": el.findtext("n:StopClassification/n:StopType", namespaces=NS_MAP) or "",
        "xml": etree.tostring(el, encoding="unicode"),
    }


def _stop_area_record(el, transformer: Transformer) -> dict | None:
    if el.get("Status") == "del":
        return None
    code = el.findtext("n:StopAreaCode", namespaces=NS_MAP)
    if not code:
        return None
    return {
        "id": code,
        "name": el.findtext("n:Name", namespaces=NS_MAP) or "",
        "lat": (coords := _coords(el.find("n:Location", NS_MAP), transformer))[0],
        "lon": coords[1],
        "type": el.findtext("n:StopAreaType", namespaces=NS_MAP) or "",
        "xml": etree.tostring(el, encoding="unicode"),
    }


def parse(source_xml: Path, transformer: Transformer) -> tuple[dict[str, list[dict]], list[list], list[list]]:
    """Stream-parse NaPTAN XML.

    Returns (chunks_by_prefix, slim_points, slim_names):
      - chunks_by_prefix: {3-char prefix: [full records with XML]} — used by stop pages
      - slim_points: [[id, lat, lon, type], ...] for StopPoints only — feeds the
        homepage MapLibre cluster source so the map doesn't pull in the heavy XML
      - slim_names: [[id, name, type], ...] for every record (points and areas) —
        loaded by the homepage search box for in-browser substring matching
    """
    sp_tag = f"{{{NS}}}StopPoint"
    sa_tag = f"{{{NS}}}StopArea"
    by_prefix: dict[str, list[dict]] = defaultdict(list)
    points: list[list] = []
    names: list[list] = []
    stops = areas = no_loc = 0

    context = etree.iterparse(str(source_xml), events=("end",), tag=(sp_tag, sa_tag))
    for _, el in context:
        if el.tag == sp_tag:
            stops += 1
            rec = _stop_point_record(el, transformer)
        else:
            areas += 1
            rec = _stop_area_record(el, transformer)

        if rec is not None:
            if rec["lat"] is None or rec["lon"] is None:
                no_loc += 1
            elif el.tag == sp_tag:
                # 5dp ≈ 1.1m — plenty for map display, smaller JSON than chunks
                points.append([rec["id"], round(rec["lat"], 5), round(rec["lon"], 5), rec["type"]])
            names.append([rec["id"], rec["name"], rec["type"]])
            by_prefix[rec["id"][:3]].append(rec)

        # iterparse memory hygiene: drop already-processed siblings
        el.clear(keep_tail=True)
        while el.getprevious() is not None:
            del el.getparent()[0]

        total = stops + areas
        if total and total % 50_000 == 0:
            print(f"  parsed {total:,} elements ({stops:,} stops, {areas:,} areas)", flush=True)

    print(f"parsed {stops:,} StopPoints + {areas:,} StopAreas "
          f"({no_loc:,} without resolvable coordinates)")
    return by_prefix, points, names


def write_chunks(by_prefix: dict[str, list[dict]], output_dir: Path) -> None:
    """Write each prefix bucket as gzipped JSON (.json.gz).

    The frontend fetches these and decompresses via the browser's native
    DecompressionStream API. Wire size is unchanged vs raw JSON (GitHub
    Pages would gzip on the wire anyway); deploy size shrinks ~16x.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for old in output_dir.glob("*.json*"):
        old.unlink()

    total_bytes = 0
    for prefix, records in sorted(by_prefix.items()):
        records.sort(key=lambda r: r["id"])
        path = output_dir / f"{prefix}.json.gz"
        payload = json.dumps(records, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        # mtime=0 -> deterministic output across runs with identical input
        with gzip.GzipFile(path, "wb", compresslevel=9, mtime=0) as fh:
            fh.write(payload)
        total_bytes += path.stat().st_size

    print(f"wrote {len(by_prefix)} chunk(s) to {output_dir.relative_to(ROOT)} "
          f"({total_bytes / (1024 * 1024):.1f} MB total, gzipped)")


def _write_gz_json(payload: list, path: Path) -> None:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    with gzip.GzipFile(path, "wb", compresslevel=9, mtime=0) as fh:
        fh.write(raw)


def write_points(points: list[list], output_dir: Path) -> None:
    """Slim StopPoint index for the homepage MapLibre source."""
    points.sort()
    path = output_dir / "points.json.gz"
    _write_gz_json(points, path)
    print(f"wrote points.json.gz: {len(points):,} StopPoints "
          f"({path.stat().st_size / (1024 * 1024):.1f} MB, gzipped)")


def write_names(names: list[list], output_dir: Path) -> None:
    """Slim name index for the homepage's in-browser substring search."""
    names.sort()
    path = output_dir / "names.json.gz"
    _write_gz_json(names, path)
    print(f"wrote names.json.gz: {len(names):,} records "
          f"({path.stat().st_size / (1024 * 1024):.1f} MB, gzipped)")


def process(source: Source) -> None:
    print(f"\n=== {source.name.upper()} === {source.url}")
    cache_path = ROOT / source.cache
    download(source.url, cache_path)
    transformer = Transformer.from_crs(source.grid_crs, "EPSG:4326", always_xy=True)
    by_prefix, points, names = parse(cache_path, transformer)
    output_dir = DATA_DIR / source.name
    write_chunks(by_prefix, output_dir)
    write_points(points, output_dir)
    write_names(names, output_dir)


def main(argv: list[str]) -> int:
    selected = set(argv[1:]) if len(argv) > 1 else None
    for source in SOURCES:
        if selected is None or source.name in selected:
            process(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
