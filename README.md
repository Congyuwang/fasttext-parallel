# Multithreading For FastText

An easy to use tool for multithreading fasttext batch inference.

## Features
- Fast inference using multithreading utilizing all cpu cores.
- Predicted labels are encoded as i16 numpy array for small memory footprint and easy serialization.

## Performance

On macbook Air M2 chip, the performance is about 4 times faster,
which is reasonable since it has 4 performance cores.

```
❯ python -m unittest discover test
fasttext-parallel 1.7130832079565153s
fasttext          6.7440170829650015s
```

## Usage

```
pip install fasttext-parallel
```

```python
import fasttext_parallel as ft
model = ft.load_model("./model/lid.176.bin")

# this uses multiple threads
labels, probabilities = model.batch(["你好", "how are you"])

# labels are in a format of numpy.ndarray (i16) format
# to get actual label, call get_label_by_id
assert model.get_label_by_id(labels[0][0]) == "__label__zh"
assert model.get_label_by_id(labels[1][0]) == "__label__en"

# to view all labels (a dict from label_id to label)
print(model.get_labels())
```
