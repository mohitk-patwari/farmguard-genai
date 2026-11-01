# Label formats

Put your hand labels here. Frames are image files (extract them with ffmpeg or the helper in
`farmguard/eval`). Paths are relative to the repo root. Keep a train/dev/test split by **clip**, never
by frame, so frames from the same clip never appear in two splits.

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
`status`: gone | leaving | still_present (judged on the last frames).

## intrusions.csv (E5: false alarms)
```
clip,start_s,end_s,species
data/clips/night_03.mp4,41.0,95.5,nilgai
```
Every true intrusion interval in every evaluated clip. Triggers outside all intervals are false alarms.

## Labelling rules (write these into the paper's method section)
- Two people label independently; report Cohen's kappa; resolve disagreements by discussion.
- Night/IR clips are labelled separately so day vs night results can be reported.
