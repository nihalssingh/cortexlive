# Data

The model was trained on the Algonauts 2025 challenge release of the Courtois NeuroMod
(CNeuroMod) dataset. Running it on your own videos needs only the model bundle. The
dataset is needed to recognise the original stimuli and score runs against the recorded
fMRI, and, with the extracted backbone features (the README's Reproduce section), to
rebuild the model and the measurements in `runs/`.

## What it contains, and the terms

- **fMRI**: the brain responses of four participants (sub-01, 02, 03 and 05) watching
  Friends seasons 1–6 and four Movie10 films, already parcellated to the Schaefer
  1000-parcel atlas. The participants shared it under CC0 through the Canadian Open
  Neuroscience Platform.
- **Transcripts**: the timed words of each stimulus's dialogue, distributed with the release.
- **Videos**: the Friends episodes and the films (The Bourne Supremacy, Hidden Figures,
  Life, The Wolf of Wall Street). They are copyrighted and not covered by CC0: CNeuroMod
  shares them only for scientific research, under Canada's Fair Dealing policy, and asks
  users to check the terms that apply to them. No video is included in this repository,
  the model bundle or exported pages.

By downloading the data you confirm that you are 18 or older; some of the films may be
disturbing to some viewers.

## Credit

Work that uses the data should cite:

- Gifford AT, Bersch D, St-Laurent M, Pinsard B, Boyle J, Bellec L, Oliva A, Roig G,
  Cichy RM. 2025. The Algonauts Project 2025 Challenge: How the Human Brain Makes Sense of
  Multimodal Movies. arXiv:2501.00504. https://doi.org/10.48550/arXiv.2501.00504
- Boyle J, Pinsard B, Borghesani V, Paugam F, DuPre E, Bellec P. 2023. The Courtois
  NeuroMod project: quality assessment of the initial data release (2020). 2023 Conference
  on Cognitive Computational Neuroscience. https://doi.org/10.32470/CCN.2023.1602-0

and include CNeuroMod's acknowledgement:

> The Courtois project on neural modelling was made possible by a generous donation from
> the Courtois foundation, administered by the Fondation Institut Gériatrie Montréal at
> CIUSSS du Centre-Sud-de-l’île-de-Montréal and University of Montreal. The Courtois
> NeuroMod team is based at Centre de Recherche de l’Institut Universitaire de Gériatrie de
> Montréal, with several other institutions involved. See the CNeuroMod documentation for
> an up-to-date list of contributors (https://docs.cneuromod.ca).

## Download

The challenge lists a data form among its resources: https://forms.gle/kmgYdxR92H4nUBfH7.
From the repository folder, with [DataLad](https://www.datalad.org) and git-annex installed:

```sh
datalad install -r -s https://github.com/courtois-neuromod/algonauts_2025.competitors.git data/algonauts
cd data/algonauts
datalad get -r -J8 fmri/ stimuli/transcripts/
datalad get stimuli/movies/friends/s6/friends_s06e01a.mkv
```

The first `get` fetches the fMRI and transcripts (about 2.3 GB); the second fetches one
Friends episode to run live. The videos (Friends seasons 1–7 and the four films; season 7
has no released fMRI, so it runs like any other video and is not scored) are large, so
fetch only the ones you need. Rebuilding the features needs no local videos:
`scripts/modal_extract.py` fetches each one on Modal. `data/` is not tracked by git.
