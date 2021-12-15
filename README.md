# Dynamic Neural Representational Decoders for High-Resolution Semantic Segmentation

## Requirements

This repository needs mmsegmentation

## Training

To train the model(s) in the paper, run this command:

```train
python tools/train.py ./configs/NRD/ade20k/NRD_r101_512x512_164k_ade20k.py
```
The batch size is 16 in this work. Please change the 'samples_per_gpu' in configs/_base_/datasets/.. accordingly

## Evaluation

To evaluate my model at single-scale inference, run:

```eval
python tools/eval.py ./configs/NRD/ade20k/NRD_r101_512x512_164k_ade20k.py  {path-to-checkpoint-file}   --eval mIoU
```

## Pre-trained Models


## Results

Our model achieves the following performance on :

### [Semantic segmentation results]

| Model name         |datasets| mIoU  | mIoU (ms) |
| ------------------ |--------------|---------------- | -------------- |
| NRD-r101   | ade20k (val) |   44.01         |      45.62       |
| NRD-x101   |ade20k (val) |  44.34         |      46.35       |
| NRD-r101   | pascal-context(val) |     52.31 (59 classes)       |      54.1 (59 classes)       |
| NRD-r101   | pascal-context(val) |     47.5  (60 classes)      |      49.0 (60 classes)       |
| NRD-r50   | Cityscapes (val) |   79.8         |      80.8       |
| NRD-r101   | Cityscapes (val) |   80.7         |      82.0      |


## Contributing

The code is mostly taken from mmsegmentation
mmsegmentation is released under the [Apache 2.0 license](LICENSE).
