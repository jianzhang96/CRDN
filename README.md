## Cascade RDN: Towards Accurate Localization in Industrial Visual Anomaly Detection with Structural Anomaly Generation - IEEE RA-L 2023
paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/10187674)

## Datasets
To train on the MVtec AD dataset, [download](https://www.mvtec.com/company/research/datasets/mvtec-ad)
the data and extract it. The [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) dataset was used as the anomaly source image set in most of the experiments in the paper. The BTAD dataset are available at [here](http://avires.dimi.uniud.it/papers/btad/btad.zip).
You can run the **download_dataset.sh** script from the project directory
to download the MVTec and the DTD datasets to the **datasets** folder in the project directory:
```
./download_dataset.sh
```


## Train
Pass the folder containing the training dataset to the **train_mvtec.py** script as the --data_path argument and the
folder locating the anomaly source images as the --anomaly_source_path argument.
The training script also requires the batch size (--bs), learning rate (--lr), epochs (--epochs), path to store checkpoints
(--checkpoint_path) and path to store logs (--log_path).
Example:

```
python train_mvtec.py --gpu_id 0 --obj_id -1 --lr 0.0001 --bs 8 --epochs 700 --data_path ./datasets/mvtec/ --anomaly_source_path ./datasets/dtd/images/ --checkpoint_path ./checkpoints/ --log_path ./logs/
```

The conda environement used in the project is decsribed in **requirements.txt**.


## Test
The test script requires the --gpu_id arguments, the name of the checkpoint files (--base_model_name) for trained models, the
location of the MVTec AD dataset (--data_path) and the folder where the checkpoint files are located (--checkpoint_path)
with pretrained models can be run with:

```
python test_mvtec.py --gpu_id 0 --base_model_name "CRDN_test_0.0001_700_bs8" --data_path ./datasets/mvtec/ --checkpoint_path ./checkpoints/
```

## Pretrained Model
The pretrained CRDN-Base models on MVTec AD dataset are available at [OneDrive](https://mailhfuteducn-my.sharepoint.com/:f:/g/personal/2015216892_mail_hfut_edu_cn/EoXq2bIzbDpJkTbcPXyIOQMBCj2a9XWw4SYim7f1fA8Nag?e=wSyupa) or [Baidu Disk](https://pan.baidu.com/s/1WV6r9_KGVXSgEVOUYxY4Vg?pwd=mwln).</br>
98.6 AUROC for anomaly classification and 93.9 PRO, 75.8 AP for anomaly segmentation.

[//]: (提取码：mwln)

<!-- ## Reference
````
@inproceedings{zhang2022fdsnet,
  title={FDSNeT: An Accurate Real-Time Surface Defect Segmentation Network},
  author={Zhang, Jian and Ding, Runwei and Ban, Miaoju and Guo, Tianyu},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={3803--3807},
  year={2022},
  organization={IEEE}
}
```` -->

## Acknowledgement
[DRAEM](https://github.com/VitjanZ/DRAEM)
