# Known issues

Things that were noticed during cleanup and deliberately left as they are.

## jellyfish version
`requirements.txt` pins `jellyfish==1.0.3`. Version 1.0.0 had no installable wheel on the machine used for cleanup. The library feeds one feature and one blocking pass, and the phonetic codes may differ slightly from the version the committed model was trained with. If scores look off, retrain with the pinned version.

## `&` to `and` rule never fires
`normalize.py` maps `&` to `and`, but punctuation is stripped earlier, so the rule has no effect. Fixing it changes the features, so the model would need retraining. It is left alone so the committed model keeps matching its features.

## Leaderboard score for the final model
Only the first model's public score (0.711) was recorded. The final model's validation macro F0.5 is 0.8512 and its blocking recall is 0.8470, taken from `model_meta.json`. Its leaderboard score was never written down. The challenge data is not in this repository, so these numbers cannot be reproduced from here.
