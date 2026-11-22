"""Check syntax, private paths, document imports and source checksums."""
import ast
import csv
import hashlib
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]


def check():
    issues=[]
    with (ROOT/"docs/source_manifest.csv").open(newline="") as f:
        manifest=list(csv.DictReader(f))
    for row in manifest:
        path=ROOT/row["path"]
        if not path.is_file():
            issues.append(f"Missing source: {row['path']}")
            continue
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row["release_sha256"]:
            issues.append(f"Checksum mismatch: {row['path']}")
        text=raw.decode("utf-8")
        try:
            tree=ast.parse(text)
            compile(tree,str(path),"exec")
        except (SyntaxError,ValueError) as e:
            issues.append(f"Syntax: {row['path']}: {e}")
            continue
        for n in ast.walk(tree):
            if isinstance(n,ast.ImportFrom) and (n.module or "").split(".")[0] in ("docx","pptx"):
                issues.append(f"Document import: {row['path']}")
            if isinstance(n,ast.Import) and any(a.name.split(".")[0] in ("docx","pptx") for a in n.names):
                issues.append(f"Document import: {row['path']}")
            if isinstance(n,ast.Constant) and isinstance(n.value,str):
                if n.value.startswith("/Users/"):
                    issues.append(f"Private absolute input path: {row['path']}")
                if re.search(r"\.(?:docx|pptx)(?:$|[\s\"'])",n.value):
                    issues.append(f"Document dependency: {row['path']}")
        if re.search(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)",text):
            issues.append(f"Possible credential: {row['path']}")
    print(f"Checked {len(manifest)} archived calculation sources.")
    if issues:
        print("\n".join(sorted(set(issues))),file=sys.stderr)
        return 1
    print("Syntax, source hashes, path and document-dependency checks passed.")
    return 0


if __name__=="__main__":
    raise SystemExit(check())
