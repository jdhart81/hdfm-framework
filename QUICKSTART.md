# Quickstart: reproduce the worked examples

Three short paths, each with exact commands and the output you should see. Pick the one that matches what you want to check. Every command below was run from a fresh clone on Linux with Node 22.22.0 and Python 3.13.

```sh
git clone https://github.com/jdhart81/hdfm-framework.git
cd hdfm-framework
```

Read [research status](docs/RESEARCH_STATUS.md) before relying on any number here. These examples show that the software runs and gives the same answer every time. They are not field evidence of ecological benefit.

## 1. Landscape assessment from the command line (Node 22+)

Runs the same geometric analysis as the web workspace on the included fictional watershed, with a 50 m waterway buffer and a 6 m road surface.

```sh
cd web
npm ci
npm run analyze -- examples/fictional-watershed.geojson /tmp/dfm-assessment.json 50 6
```

Expected output:

```
Assessment written to /tmp/dfm-assessment.json. 7 data warnings. Geometric screening only.
```

The seven warnings ask for the source, date and reuse terms of each input layer; the fictional example records none. Read the headline metrics:

```sh
node -e "const r=require('/tmp/dfm-assessment.json');console.log(r.method);for(const[k,v]of Object.entries(r.metrics))console.log(k.padEnd(20),+v.toFixed(3))"
```

```
dfm-geometric-0.3.0
boundaryHa           772.772
bufferHa             47.374
forestHa             96.442
netForestHa          96.442
retainedForestHa     2.62
roadSurfaceHa        3.069
openWaterHa          0
managementOverlapHa  0
crossingCandidates   4
```

The analysis refuses to overwrite an existing output file, so delete `/tmp/dfm-assessment.json` before running it again. The same input and parameters give a byte-identical file. `npm test` includes a check (`tests/quickstart.test.mjs`) that fails if these numbers change.

To open the same project in the map workspace instead, run `npm run dev` and visit http://127.0.0.1:4173/map.

## 2. The dfm-core engine test suite (Node 22+)

`@viridis/dfm-core` holds the corridor connectivity checks and the old-growth spine. Its tests run woodlot, prairie and watershed fixtures and re-derive a stored VergeCommon result and checksum.

```sh
cd packages/dfm-core
npm ci
npm test
```

Expected last lines:

```
# tests 133
# pass 133
# fail 0
```

## 3. Research toolkit: synthetic topology comparison (Python 3.8+)

The experimental Python package in `hdfm-framework/` compares five corridor network topologies on random synthetic landscapes using the package's composite entropy score.

```sh
cd hdfm-framework
python -m venv .venv && . .venv/bin/activate
python -m pip install -e . pytest
python -m pytest -q
MPLBACKEND=Agg python examples/synthetic_landscape_validation.py --n-landscapes 100 --seed 42 --no-save
```

`pytest` should report `73 passed`. The comparison takes a few seconds and prints this table:

```
Topology             Mean H       Std H        vs. Dendritic        p-value
--------------------------------------------------------------------------------
Dendritic (MST)      1.321        0.129        (reference)          —
Gabriel              8.719        2.042        +559.8%              <0.0001
Delaunay             22.312       1.124        +1588.5%             <0.0001
Knn                  17.117       4.101        +1195.3%             <0.0001
Threshold            24.920       8.999        +1785.8%             <0.0001
```

Drop `--no-save` to also write figures and a text summary to `validation_results/`. The seed fixes the landscapes, so a second run prints the same table. These values were produced with numpy 2.5.4, scipy 1.18.1 and networkx 3.7; other library versions may shift the last digit.

How to read this: the dendritic network is a minimum spanning tree, which minimizes summed edge length on the supplied graph, not necessarily the composite score. A lower score on synthetic landscapes is a property of the score and the generator. It does not show that dendritic corridors work better in a real forest. These are also not the figures in the HDFM manuscript, which reports a 100 km region and the movement-entropy term alone; that table has not yet been reproduced from this code.

## Cite this work

Citation metadata is in [CITATION.cff](CITATION.cff). GitHub shows a "Cite this repository" button built from it, and `cffconvert -f bibtex` turns it into BibTeX. Please also name the component and version you used (web application, dfm-core or research toolkit), since they are versioned separately.

## Something did not match

Open an issue with the command you ran, your OS, Node or Python version, and the output you saw. A mismatch in these examples is a bug worth reporting.
