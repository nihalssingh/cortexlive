# Llama-cortexlive model license

Llama-cortexlive is the trained model that `cortexlive setup --models` installs: the
fitted head, a standardization and PCA per backbone and a ridge regression per participant
(`models/final_head.npz`, `runs/final_head/meta.json`), its reference data
(`models/reference.npz`) and the speech-feature control (`models/control_head.npz`).

Copyright (c) 2026 Nihal Sanjay Singh (https://github.com/nihalssingh). Licensed under
the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0):
https://creativecommons.org/licenses/by-nc/4.0/

**Built with Llama.** The model's text features are hidden states of Llama 3.2 1B, so it
is also subject to the Llama 3.2 Community License (`LLAMA_3.2_COMMUNITY_LICENSE.txt`) and
the Llama 3.2 Acceptable Use Policy (https://dev.meta.ai/llama/llama3_2/use-policy; the
license text gives its earlier address, https://www.llama.com/llama3_2/use-policy). As that
license requires of a model trained on Llama outputs, its name begins with "Llama".

Llama 3.2 is licensed under the Llama 3.2 Community License, Copyright © Meta Platforms, Inc. All Rights Reserved.

The model was fit to fMRI recordings of four Courtois NeuroMod (CNeuroMod) participants,
released for the Algonauts 2025 challenge under CC0, and to features of the Friends and
Movie10 videos, which are copyrighted and were used for scientific research. Its
predictions are for those four participants, not for whoever is watching. It is a research
model, not a clinical or diagnostic tool.
