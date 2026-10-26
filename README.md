# CVDS Admission Debugging Exercises - Solutions

Solutions to the Computer Vision & Data Science (NHL Stenden) admission debugging exercises (`debugging.ipynb`).

All cells in [`debugging.ipynb`](debugging.ipynb) have been executed in order from top to bottom with their outputs saved.

## Summary of Bugs and Fixes

### Exercise 1 - `id_to_fruit`
- **Bug:** `fruits` was passed as a Python `set` (`Set[str]`). Because sets are unordered collections, iterating over `fruits` returns items in arbitrary hash order rather than the order written in the literal.
- **Fix:** Use an ordered sequence (`List[str]` / `Sequence[str]`) so that indices `1`, `3`, and `4` deterministically return `'orange'`, `'kiwi'`, and `'strawberry'`.

### Exercise 2 - `swap`
- **Bug 1 (obvious typo):** The right-hand side of the assignment repeated `coords[:, 1]` twice (`coords[:, 1], coords[:, 1], coords[:, 3], coords[:, 2]`) instead of `coords[:, 1], coords[:, 0], coords[:, 3], coords[:, 2]`.
- **Bug 2 (NumPy view aliasing):** NumPy slices (`coords[:, 0]`, `coords[:, 1]`) are views into the same array. Unpacking `coords[:, 0], coords[:, 1] = coords[:, 1], coords[:, 0]` first copies column 1 into column 0 in-place, and then copies the modified column 0 back into column 1.
- **Fix:** Copy the array first (`swapped = coords.copy()`) and swap columns using fancy indexing: `swapped[:, [0, 1, 2, 3]] = coords[:, [1, 0, 3, 2]]`.

### Exercise 3 - `plot_data`
- **Bug 1:** `csv.reader` returns each row as a list of strings, so `np.stack(results)` produced an array of strings rather than floating-point numbers (and on Windows, opening the CSV without `newline=""` inserts empty rows).
- **Bug 2:** The CSV columns are `["precision", "recall"]` (column 0 = precision, column 1 = recall), and precision belongs on the x-axis with recall on the y-axis. The original code plotted `plt.plot(results[:, 1], results[:, 0])` and swapped the axis labels.
- **Fix:** Parse each row as `float`, plot `plt.plot(results[:, 0], results[:, 1])`, and set `plt.xlabel("Precision")` and `plt.ylabel("Recall")`.

### Exercise 4 - `train_gan`
- **Bug 1 (structural):** The MNIST training set has 60,000 samples. With `batch_size=32`, `60000 % 32 == 0`, so every batch has 32 samples. With `batch_size=64`, `60000 % 64 == 32`, so the final batch of each epoch has 32 real samples while `real_samples_labels`, `latent_space_samples`, and `generated_samples_labels` were hardcoded to `batch_size` (64). Concatenating 32 real samples and 64 generated samples produced 96 inputs against 128 target labels (`torch.Size([96, 1])` vs `torch.Size([128, 1])`).
- **Fix 1:** Use the actual batch size `current_batch_size = real_samples.size(0)` for label and latent-space tensors in each step.
- **Bug 2 (cosmetic):** The display check `if n == batch_size - 1:` only triggered on batch index `batch_size - 1` rather than at the end of each epoch (`if n == len(train_loader) - 1:`), and figures were not closed with `plt.close(fig)` after `display(fig)`.
- **Fix 2:** Check `if n == len(train_loader) - 1:` and call `plt.close(fig)` after displaying each figure.
