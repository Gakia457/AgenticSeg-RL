### 输入
cd base_segmentation_train
ls
ls train

### 输出
base_segmentation_train$ ls train
data-00000-of-00014.arrow  data-00003-of-00014.arrow  data-00006-of-00014.arrow  data-00009-of-00014.arrow  data-00012-of-00014.arrow  state.json
data-00001-of-00014.arrow  data-00004-of-00014.arrow  data-00007-of-00014.arrow  data-00010-of-00014.arrow  data-00013-of-00014.arrow
data-00002-of-00014.arrow  data-00005-of-00014.arrow  data-00008-of-00014.arrow  data-00011-of-00014.arrow  dataset_info.json


### 输入
python -c "from datasets import load_from_disk; d=load_from_disk(r'.'); print(d); print(d['train'].features); print(d['train'].column_names); print(len(d['train']))"

### 输出
DatasetDict({
    train: Dataset({
        features: ['id', 'problem', 'solution', 'solution_mask', 'image', 'img_height', 'img_width'],
        num_rows: 5166
    })
})
{'id': Value('string'), 'problem': Value('string'), 'solution': Value('string'), 'solution_mask': LargeList(Array2D(shape=(840, 840), dtype='bool')), 'image': Image(mode=None, decode=True), 'img_height': Value('int64'), 'img_width': Value('int64')}
['id', 'problem', 'solution', 'solution_mask', 'image', 'img_height', 'img_width']
5166


### 输入
python -c "from datasets import load_from_disk; import pandas as pd; ds=load_from_disk(r'.')['train']; ds=ds.remove_columns(['image','solution_mask']); pd.set_option('display.max_colwidth', 200); print(ds.select(range(5)).to_pandas())"

### 输出
                         id                                                                                              problem  ... img_height  img_width
0            refcocog_28421  'this is a display screen for presentations, showing exactly what the computer screen on the table'  ...        640        478
1      refcocog_16521_10608                 'a black and white dog laying down, looking away from the camera' and 'standing dog'  ...        426        640
2           refcocog_428468                                                         'a man with a mustache and a checked jacket'  ...        446        640
3  refcocog_1534147_1039564                                'a pink plastic box which is rectangle' and 'container holding fruit'  ...        480        640
4           refcocog_583192                        'a large adult elephant that is being followed by two small infant elephants'  ...        427        640

[5 rows x 5 columns]


### 输入
python -c "from datasets import load_from_disk; import numpy as np; ds=load_from_disk(r'.')['train']; row=ds[0]; masks=row['solution_mask']; print('num_masks:', len(masks)); m=np.asarray(masks[0]); print('first_mask_shape:', m.shape); print('dtype:', m.dtype); print('foreground_pixels:', int(m.sum()))"

### 输出
num_masks: 1
first_mask_shape: (840, 840)
dtype: bool
foreground_pixels: 212680

### 输入
python - <<'PY'
from datasets import load_from_disk
from PIL import Image
import numpy as np
import os

ds = load_from_disk(r'.')['train']
row = ds[0]

os.makedirs('preview_sample_0', exist_ok=True)

img = row['image']
img.save('preview_sample_0/image.png')

masks = row['solution_mask']
print('num_masks:', len(masks))

for i, mask in enumerate(masks[:5]):
    arr = np.asarray(mask, dtype=np.uint8) * 255
    Image.fromarray(arr).save(f'preview_sample_0/mask_{i}.png')

print('saved to preview_sample_0/')
PY
### 输出
num_masks: 1
saved to preview_sample_0/
（图片掩码都有了）

### 输入
python - <<'PY'
import pyarrow as pa
import pyarrow.ipc as ipc

<!-- path = "train/data-00000-of-00014.arrow" -->
path = "train/data-00000-of-00003.arrow"

with pa.memory_map(path, "r") as source:
    reader = ipc.open_stream(source)
    print(reader.schema)
PY
### 输出
id: string
problem: string
solution: string
solution_mask: large_list<item: list<item: list<item: bool>>>
  child 0, item: list<item: list<item: bool>>
      child 0, item: list<item: bool>
          child 0, item: bool
    -- field metadata --
    ARROW:extension:metadata: '[[840, 840], "bool"]'
    ARROW:extension:name: 'datasets.features.features.Array2DExtensionTyp' + 1
image: struct<bytes: binary, path: string>
  child 0, bytes: binary
  child 1, path: string
img_height: int64
img_width: int64
-- schema metadata --
huggingface: '{"info": {"features": {"id": {"dtype": "string", "_type": "' + 355


### 输入
python - <<'PY'
from datasets import load_from_disk
import json

ds = load_from_disk(r'.')['train']

light_ds = ds.remove_columns(['image', 'solution_mask'])

with open('preview.jsonl', 'w', encoding='utf-8') as f:
    for row in light_ds.select(range(min(20, len(light_ds)))):
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

print('saved preview.jsonl')
PY

### 输出
saved preview.jsonl
（文件正常输出）

### 输入
grep -RniE "load_from_disk|load_dataset|read_parquet|from_parquet|to_parquet|pyarrow|arrow|parquet|Dataset" ./verl | head -n 100

grep -RniE "class .*Dataset|RLHFDataset|data_path|train_files|val_files|prompt_key|image_key|multi_modal" ./verl | head -n 200
### 输出




### 采样读取5行

python - <<'PY'
from datasets import Dataset
import json

path = "./data/hf_agentic/base_dataset/task3/train/data-00000-of-00015.arrow"

ds = Dataset.from_file(path)

print("dataset:", ds)
print("features:", ds.features)
print("columns:", ds.column_names)
print("num_rows:", len(ds))

omit_cols = {"image", "solution_mask"}

for i in range(min(5, len(ds))):
    row = ds[i]
    row = {k: v for k, v in row.items() if k not in omit_cols}

    print("\n" + "=" * 100)
    print("row_index:", i)
    print(json.dumps(row, ensure_ascii=False, indent=2))
PY