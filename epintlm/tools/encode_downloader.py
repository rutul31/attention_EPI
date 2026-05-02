"""Dynamically resolve and download ENCODE BigWig tracks for the EPINTLM dataset.

Hardcoding accessions doesn't work — ENCODE periodically deprecates files. Instead, this tool
queries the ENCODE REST search API for the best matching released file per (cell, assay)
combination and downloads it.

Selection criteria per assay (in priority order):
  - Histone ChIP-seq (H3K27ac, H3K4me1, H3K4me3): assay_title="Histone ChIP-seq",
                                                  output_type="fold change over control"
  - TF ChIP-seq (CTCF):                            assay_title="TF ChIP-seq",
                                                  output_type="fold change over control"
  - DNase-seq:                                     assay_title="DNase-seq",
                                                  output_type prefers
                                                    "read-depth normalized signal" then
                                                    "signal of unique reads" then
                                                    any released BigWig
All requests fix assembly=hg19, status=released, file_format=bigWig.
Among matches, prefers the file covering both biological replicates ([1, 2]).

Usage:
  python -m epintlm.tools.encode_downloader \\
    --dest data/raw/encode \\
    --cells GM12878 HeLa-S3 HUVEC IMR90 K562 NHEK \\
    --assays CTCF DNase H3K27ac H3K4me1 H3K4me3 \\
    [--manifest data/raw/encode/manifest.json] \\
    [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Iterable, Optional

API_BASE = "https://www.encodeproject.org"

# ENCODE biosample term names sometimes differ from how the paper labels cell lines.
CELL_TO_BIOSAMPLE = {
    "GM12878": "GM12878",
    "HeLa-S3": "HeLa-S3",
    "HUVEC":   "HUVEC",
    "IMR90":   "IMR-90",   # ENCODE uses the dash
    "K562":    "K562",
    "NHEK":    "NHEK",
}

ASSAY_QUERIES = {
    # Each tuple: (assay_title, target_label_or_None, list of preferred output_types)
    "CTCF":    ("TF ChIP-seq",      "CTCF",    ["fold change over control"]),
    "DNase":   ("DNase-seq",        None,      ["read-depth normalized signal",
                                                "signal of unique reads"]),
    "H3K27ac": ("Histone ChIP-seq", "H3K27ac", ["fold change over control"]),
    "H3K4me1": ("Histone ChIP-seq", "H3K4me1", ["fold change over control"]),
    "H3K4me3": ("Histone ChIP-seq", "H3K4me3", ["fold change over control"]),
}


def _api_get(path: str, params: dict) -> dict:
    qs = urllib.parse.urlencode(params, doseq=True)
    url = f"{API_BASE}{path}?{qs}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def _resolve_accession(cell: str, assay: str) -> Optional[dict]:
    """Return {accession, output_type, biological_replicates} or None if not found."""
    if cell not in CELL_TO_BIOSAMPLE:
        return None
    if assay not in ASSAY_QUERIES:
        return None
    biosample = CELL_TO_BIOSAMPLE[cell]
    assay_title, target, preferred_outputs = ASSAY_QUERIES[assay]

    base_params = {
        "type": "File",
        "file_format": "bigWig",
        "assembly": "hg19",
        "status": "released",
        "biosample_ontology.term_name": biosample,
        "assay_title": assay_title,
        "limit": "all",
    }
    if target is not None:
        base_params["target.label"] = target

    for output_type in preferred_outputs + [None]:  # last fallback: any output_type
        params = dict(base_params)
        if output_type is not None:
            params["output_type"] = output_type

        try:
            payload = _api_get("/search/", params)
        except Exception as e:
            print(f"    API error for {cell}/{assay}: {e}", file=sys.stderr)
            continue

        candidates = payload.get("@graph", [])
        if not candidates:
            continue

        # Prefer files that cover both biological replicates (i.e. pooled).
        candidates.sort(key=lambda c: (
            -len(c.get("biological_replicates") or []),  # more reps first
            c.get("date_created", ""),                   # newer first by lexical date
        ), reverse=False)
        # We want "more reps first" → first sort key is negative count (smaller = better);
        # newer first → reverse_sort the date ascending then take last? Simpler: dedicated key.
        candidates.sort(key=lambda c: (
            -(len(c.get("biological_replicates") or [])),
            -ord(c.get("date_created", "0000")[0]) if c.get("date_created") else 0,
        ))

        chosen = candidates[0]
        return {
            "accession": chosen.get("accession"),
            "output_type": chosen.get("output_type"),
            "biological_replicates": chosen.get("biological_replicates"),
            "date_created": chosen.get("date_created"),
        }

    return None


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"    fetching {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, tmp)  # follows redirects
    tmp.rename(dest)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", required=True, type=Path,
                        help="Output dir (creates {dest}/{cell}/{assay}.bigWig)")
    parser.add_argument("--cells", nargs="+", default=list(CELL_TO_BIOSAMPLE.keys()))
    parser.add_argument("--assays", nargs="+", default=list(ASSAY_QUERIES.keys()))
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Write a JSON manifest of resolved accessions (default: {dest}/manifest.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Resolve accessions and print manifest only; skip downloads.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    args.dest.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or (args.dest / "manifest.json")
    manifest: dict = {}

    for cell in args.cells:
        manifest.setdefault(cell, {})
        for assay in args.assays:
            print(f"==> {cell} / {assay}")
            info = _resolve_accession(cell, assay)
            if info is None or not info.get("accession"):
                print(f"    !! no released BigWig found for {cell}/{assay}")
                manifest[cell][assay] = {"error": "not found"}
                continue

            accession = info["accession"]
            url = f"{API_BASE}/files/{accession}/@@download/{accession}.bigWig"
            out = args.dest / cell / f"{assay}.bigWig"

            print(f"    accession={accession} output_type={info['output_type']!r} "
                  f"reps={info['biological_replicates']}")

            manifest[cell][assay] = {
                "accession": accession,
                "url": url,
                "output_type": info["output_type"],
                "biological_replicates": info["biological_replicates"],
                "date_created": info["date_created"],
                "local_path": str(out),
            }

            if args.dry_run:
                continue
            if out.exists() and out.stat().st_size > 0:
                print(f"    [skip] {out} already present ({out.stat().st_size} bytes)")
                continue
            _download(url, out)

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest → {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
