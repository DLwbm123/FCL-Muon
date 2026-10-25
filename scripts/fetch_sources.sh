#!/bin/sh
set -eu
mkdir -p sources
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7897}"
export HTTP_PROXY="$HTTPS_PROXY" ALL_PROXY="$HTTPS_PROXY" NO_PROXY=''
if [ ! -d sources/fed-mammoth/.git ]; then
  git -c http.proxy="$HTTPS_PROXY" clone https://github.com/aimagelab/fed-mammoth.git sources/fed-mammoth
fi
git -C sources/fed-mammoth checkout 6d57b5d361f743f3ff0cd1371409bd7a1158de64
curl --fail -L https://proceedings.iclr.cc/paper_files/paper/2026/file/427f20d90386fd27804f1831d6a3d48f-Supplemental-Conference.zip -o sources/fedmuon.zip
python3 - <<'PY'
import hashlib,zipfile
from pathlib import Path
p=Path('sources/fedmuon.zip')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='dfc176670be5a63a8256d521229819ef428ca0529555f0efd58f6194b066904b','Supplement version changed: audit before use'
with zipfile.ZipFile(p) as z:z.extractall('sources/fedmuon')
PY
