# GEARS: Local geometry-aware Hand-Object Interaction Synthesis

Repo for **"GEARS: Local geometry-aware Hand-Object Interaction Synthesis, CVPR'24"** \
[[Paper]](http://virtualhumans.mpi-inf.mpg.de/papers/zhou24gears/gears.pdf) [[Project Page]](http://virtualhumans.mpi-inf.mpg.de/gears)

## Environment
This code is written and tested with Python 3.8. To install the required dependencies, run:  

```shell
pip install -r requirements.txt
```

We additionally require the following libraries：
- [MPI-IS Mesh Processing Library](https://github.com/MPI-IS/mesh)
- [Manopth layer for PyTorch](https://github.com/hassony2/manopth)

Please check the respective instructions for downloading and installation.

## Data
1. Download the raw GRAB dataset and SMPL-X models by following instructions [here](https://github.com/otaheri/GRAB).
2. Run our pre-processing code:
```shell
python data/create_dataset.py --grab_path $RAW_GRAB_FOLDER \
                                  --model_path $SMPLX_MODEL_FOLDER \
                                  --out_path $PROCESSED_GRAB_FOLDER
```
## Training
Train the model with
```shell
python train.py --data_path $PROCESSED_GRAB_FOLDER
```

## Citation
```bibtex
@inproceedings{zhou2024gears,
  title = {GEARS: Local Geometry-aware Hand-object Interaction Synthesis},
  author = {Zhou, Keyang and Bhatnagar, Bharat Lal and Lenssen, Jan Eric and Pons-Moll, Gerard},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2024},
}
