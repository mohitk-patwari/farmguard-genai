# Label formats

Put your hand labels here. Paths are relative to the repo root. Splits are by **clip**, never by
frame, so frames from the same clip never appear in two splits.

## Workflow
```bash
python scripts/extract_frames.py data/clips/*.mp4      # data/frames/<clip>/*.jpg + manifest.csv
python scripts/label.py split                          # writes splits.csv; then fill in lighting
python scripts/label.py vlm c01 --labeller a           # also: retreat, intrusions
python scripts/label.py vlm --labeller b --from data/labels/vlm_samples.a.csv
python scripts/label.py kappa vlm a b                  # report this number in the paper
python scripts/label.py merge vlm a b                  # agreed -> vlm_samples.csv, rest -> .conflicts.csv
```
Keys are listed at the top of `scripts/label.py`. `intrusions` is labelled once and checked by the
second person; two interval lists have no sample ids to compute kappa on.

## splits.csv
```
clip,split,lighting
c01,dev,day
c02,test,night
```
`split`: dev | test. `lighting`: day | night, filled in by hand. An assigned clip never moves. Tune
anything (fuse rules, prompts, thresholds) on dev only. `python -m farmguard.eval` reports on test by
default and stops with an error if a clip is missing from this file.

## vlm_samples.csv (E2: species + behaviour)
```
sample_id,frames,trigger_index,bbox,species,behaviour
c01_t012,data/frames/c01/0118.jpg;data/frames/c01/0120.jpg;data/frames/c01/0122.jpg,1,"412,300,610,455",wild_boar,eating
```
- `frames`: `;`-separated, time order. `trigger_index`: which of them holds the box.
- `species`: a key from config/species.yaml, or `none` / `other_animal`.
- `behaviour`: eating | moving_through | resting | leaving | absent

## retreat_samples.csv (E3: did it leave after the sound?)
```
sample_id,frames,species,status
c01_after1,data/frames/c01/0130.jpg;...;data/frames/c01/0250.jpg,wild_boar,still_present
```
`status`: gone | leaving | still_present (judged on the last frames). List every frame in the window:
the tracker baseline needs them, and the VLM picks its few last-heavy frames from the same list.

## intrusions.csv (E5: false alarms)
```
clip,start_s,end_s,species
data/clips/night_03.mp4,41.0,95.5,nilgai
```
Every true intrusion interval in every evaluated clip. Triggers outside all intervals are false alarms.
A clip passed to `e5` with no row here counts as clean video.

## Labelling rules (write these into the paper's method section)
- Two people label independently; report Cohen's kappa; resolve disagreements by discussion.
- Night/IR clips are marked in `splits.csv` so day vs night results can be reported separately
  (`--lighting night`).
- Check licences before using a public clip, and record the source of each.
- Never run a cloud provider on a frame that shows a person (PAPER_PLAN ethics statement): remove such
  samples from the CSV used for the cloud run.
