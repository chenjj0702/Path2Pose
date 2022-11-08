# Path2Pose: Path guided motion synthesis for drosophila larvae
## Representation
The Path2Pose model is an end-to-end deep generative model that synthesize motions of drosophila larvae given the initial poses and the guiding path. The synthesized larva crawls strictly along  the guiding path with various motion patterns. With the introduction of recursive generation and concatenation, the Path2Pose model could synthesize long motion with little error accumulation.
## Dataset
We constructed a large-scale dynamic pose dataset for drosophila larvae, DLPose, for the Path2Pose mode training. The dataset collected vidoes of 51 drosophila larvae with the total length of 5.1 hours. The pose data is estimated by the deep neural network and subsequently refined artifically.

The dataset is availiable at https://drive.google.com/drive/folders/1-oOVhZz1lYgQ-19T_ANKbhEx8Z8ELQB1?usp=sharing.

<p align="left">
    <img src="https://github.com/chenjj0702/Path2Pose/blob/main/images/dataset1.gif" width="200"\>  <img src="https://github.com/chenjj0702/Path2Pose/blob/main/images/dataset2.gif" width="279"\>
</p>


## Dependecies
To install all the dependencies quickly and easily you should use pip
```
pip install -r requirements.txt
```

## Execution
cd ./code/path2pose

### 1. train
Train the Path2Pose model to synthesize fixed-length pose sequnces

python main.py --mode train --save_dir ../results/train/ --train_npz ../database/public_larva_refine_pose_head_enh4_win40_step10_test1000.npz
### 2. test 
python main.py --mode test --load_model ../results/train/AttnCnNet/model/epoch_20000.pt --train_npz ../database/public_larva_refine_pose_head_enh4_win40_step10_test1000.npz
### 3. synthesize long pose sequence
python main.py --mode recursive --load_model ../results/train/AttnCnNet/model/epoch_20000.pt --recursive_npz ../database/recursive_public.npz
## Examples
We train the Path-to-Pose model with the training set of DLPose dataset and generate the long pose sequecne with the test set. The example pose sequences are concatenated with 4 short sequences.
### 1. short pose sequence
<p align="left">
    <img src="https://github.com/chenjj0702/Path2Pose/blob/main/images/pose1.gif" width="200"\>        <img src="https://github.com/chenjj0702/Path2Pose/blob/main/images/pose2.gif" width="275"\>
</p>

### 2. long pose sequence
<p align="left">
    <img src="https://github.com/chenjj0702/Path2Pose/blob/main/images/long1.gif" width="200"\>        <img src="https://github.com/chenjj0702/Path2Pose/blob/main/images/long2.gif" width="200"\>
</p>
