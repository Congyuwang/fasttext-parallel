# Multithreading For FastText

```
pip install fasttext-parallel
```

```python
import fasttext_parallel as ft
model = ft.load_model("./model/lid.176.bin")
result = model.batch(["你好", "how are you"])
```
