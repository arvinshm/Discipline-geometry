# Euclidean distances between academic disciplines

The goal of this experiment is to examine whether distances between a language
model's internal representations reflect familiar relationships between academic
disciplines. The model is Qwen3-4B-Base. We compare 24 disciplines using the same
set of short sentences and measure how far apart their average representations
are.

For each discipline, we create 12 prompts by inserting its name into 12 sentence
templates. For example, one template gives "A university department specializes
in mathematics." Another gives "A new journal publishes work in mathematics."
The same templates are used for physics, biology, and the other disciplines.
Each sentence is followed by a newline and `Topic:`. This gives 288 prompts in
total.

The model converts each prompt into tokens and processes them through its
transformer layers. We record the vector at the final colon
of `Topic:` immediately after layer 18. This residual stream is 2,560 dimensional. The colon provides a common token
at which to take the measurement.

We average the 12 vectors for each discipline, coordinate by coordinate. This
produces one vector of 2,560 numbers per discipline. We then calculate the
Euclidean distance between every pair of these mean vectors: subtract their
corresponding coordinates, square the differences, add them, and take the square
root. Averaging takes place before calculating distances. The calculation uses
all 2,560 coordinates in their original scale.

The results are shown in [distances.png](distances.png), and the numerical values
are in [distances.csv](distances.csv). The figure is a heatmap of the 24 by 24
distance matrix. Both axes list disciplines, and each cell's color represents
the distance between its row and column. The matrix is symmetric and has zeros
on its diagonal. The 276 distinct pairs provide the comparisons discussed here.

Some relationships seem natural. Sociology and anthropology are nearest neighbors,
with a distance of 2.95, as are biology and ecology, with a distance of 3.54.
Neuroscience and cognitive science are nearest neighbors at 3.87, and linguistics
and psycholinguistics are nearest neighbors at 3.64. Computer science and
electrical engineering are also nearest neighbors, at 4.86.

Other relationships are somewhat odd. Physics is closer to biology, at 5.36, than to mathematics, at 10.74. Biochemistry is closer to physics, at 5.44, than to chemistry, at 8.12. Electrical engineering is closer to ecology, at 5.98, than to physics, at 8.67. Philosophy is closer to sociology, at 4.45, than to logic, at 14.15. Literary studies is closer to political science, at 5.80, than to history, at 9.30.


---

below is AI-generated summary

The current run used a Runpod RTX A5000, float32, and batches of 16 prompts.
Inference for the 288 prompts took about 2.6 seconds, excluding model loading
and the padding check. The model revision and execution settings are stored
in `data.npz`.

The padding check compared a short prompt alone and beside a longer prompt;
the relative vector difference was 0.0000011. The new vectors also agree with
the previous run's corresponding unprefixed vectors to a relative difference
of 0.00000085. Comparing the current and previous distance matrices gives
15 of 24 unchanged closest neighbors and 57 of 72 shared top-three neighbors.

## The eight files

- `README.md`: Experiment description and reproduction guide.
- `experiment.py`: Build prompts and extract vectors.
- `analyze.py`: Average vectors, calculate distances, compare runs, and plot.
- `stimuli.json`: Discipline names and sentence templates.
- `requirements.txt`: Pinned Python dependencies.
- `data.npz`: Vectors, prompts, labels, run settings, and the previous distance matrix.
- `distances.csv`: The current 24 by 24 distance matrix.
- `distances.png`: The current distance heatmap.

`data.npz` contains `activations` (288 × 2560, float32), `discipline` (288 integer
labels), `names` (24 strings), `prompts` (288 strings), and `metadata` (a JSON
string with the model revision, settings, runtime, and extraction-source hashes).
Row `r` of `activations` corresponds to `prompts[r]` and
`names[discipline[r]]`. All project-specific code is in the two Python files.
The previous 24 × 24 matrix is embedded as `reference_distances`, in the same
discipline order, for reproducible comparison. `metadata.reference` describes
that run; `compare_distances` prints the comparison when this matrix is present.

## Run the reference implementation

Recreate the CSV and figure on this Mac, without loading the model:

```bash
cd /Users/arvinshm/Desktop/Work/ML/Projects/discipline-geometry
/Users/arvinshm/venvs/.discipline-geometry-venv/bin/python analyze.py
```

To reproduce extraction, connect to the existing Runpod machine:

```bash
ssh -tt faikzvjsp2j23g-64411540@ssh.runpod.io -i ~/.ssh/id_ed25519
cd /workspace/discipline-geometry
export HF_HOME=/workspace/.cache/huggingface
/workspace/venvs/discipline-geometry/bin/python experiment.py --out /tmp/reproduction.npz
/workspace/venvs/discipline-geometry/bin/python analyze.py --data /tmp/reproduction.npz --out /tmp/reproduction
```

Choose a new output filename for another extraction. The script protects existing
data from overwriting. Use a CUDA GPU with 24 GB memory for the default batch of
16. Model loading needs internet access or the existing model cache. On a fresh
machine, use Python 3.12 and install `requirements.txt` in a virtual environment;
install a CUDA build of PyTorch appropriate for that machine first.

## Rebuild it yourself, one step at a time

1. **Construct prompts.** Start with mathematics and statistics in
   `A university department specializes in {discipline}.\nTopic:`.
   Print both strings. Then write the two loops in `make_prompts`: template,
   discipline. Check for 288 strings and 12 occurrences of each label.
2. **Find the readout token.** Load the pinned tokenizer from `experiment.py`.
   Tokenize the two strings with `add_special_tokens=False`; inspect them with
   `convert_ids_to_tokens`. Both end at the colon. With right padding, its index
   is `attention_mask.sum(dim=1) - 1`, calculated separately for every prompt.
3. **Extract one vector.** Load the pinned model with float32 and eager attention,
   call `eval()`, and use `torch.inference_mode()`. Register a forward hook on
   `model.layers[17]`. Its output has shape `(batch, tokens, 2560)`; select each
   prompt's colon to obtain `(batch, 2560)`. Follow `capture` and `forward` in
   `experiment.py`. Compare a short prompt alone and beside a longer prompt:
   relative vector error should be below `1e-4`. Then collect all 288 vectors.
4. **Average by discipline.** For each integer label `d`, compute
   `activations[labels == d].mean(axis=0, dtype=np.float64)`.
   Check that the result has shape `(24, 2560)`. See `mean_vectors`.
5. **Calculate distances.** First calculate mathematics versus statistics with
   `np.linalg.norm(means[0] - means[13])`; expect about `7.084455`.
   Then generalize to all pairs as in `euclidean_distances`. Check the matrix is
   symmetric, nonnegative, and has a zero diagonal.
6. **Draw the heatmap.** Use `imshow(distances, vmin=0)`, label both axes with
   discipline names, and add a colorbar in raw activation units.
   Compare with `distances.png`; `plot_distances` is the reference.

Start now with steps 4–5 using the saved vectors; this requires only NumPy and
lets you verify the central calculation before tackling GPU extraction:

```python
import numpy as np

with np.load("data.npz", allow_pickle=False) as saved:
    x, labels, names = saved["activations"], saved["discipline"], saved["names"]
math = x[labels == np.flatnonzero(names == "mathematics")[0]].mean(axis=0, dtype=np.float64)
stats = x[labels == np.flatnonzero(names == "statistics")[0]].mean(axis=0, dtype=np.float64)
print(np.linalg.norm(math - stats))  # approximately 7.084455
```
