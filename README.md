# SMDD-Synthetic-Face-Morphing-Attack-Detection-Development-dataset
Official repository of the paper Privacy-friendly Synthetic Data for the Development of Face Morphing Attack Detectors

[*** Accepted at CVPR workshops 2022 ***](https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/html/Damer_Privacy-Friendly_Synthetic_Data_for_the_Development_of_Face_Morphing_Attack_CVPRW_2022_paper.html)

![grafik](https://user-images.githubusercontent.com/85616215/158406086-b413c5b9-e4da-4e0e-be01-4de71d279979.png)


Paper available under this [LINK](https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/html/Damer_Privacy-Friendly_Synthetic_Data_for_the_Development_of_Face_Morphing_Attack_CVPRW_2022_paper.html)


![grafik](https://user-images.githubusercontent.com/85616215/158062680-a4275e0f-6e8a-4728-97f9-1b44e03ac53d.png)



The training data split of the SMDD data can be downloaded from this [LINK](https://drive.google.com/file/d/1I6x_gWtu3WxOloK8k-tecXjV3XsRPYYO/view?usp=sharing) (please share your name, affiliation, and official email in the request form).

The testing data split of the SMDD data can be downloaded from: [LINK](https://drive.google.com/file/d/1V8libhAEMOL77gtUDtcruHIKBUy8ks6y/view?usp=share_link) (please share your name, affiliation, and official email in the request form)

The pretrained weight of MixFaceNet-MAD model on SMDD training data can be downloaded from this [LINK](https://drive.google.com/file/d/1qw6YZ3cpaa9UK2-hRfzKWx5rPvRo0h63/view?usp=sharing) (please share your name, affiliation, and official email in the request form).

## Data preparation
Our face data is preprocessed by the face detection and cropping. The implementation can be found in image_preprocess.py file.
Moreover, for further training and test, the corresponding CSV files should be generated. The format of the dataset CSV file in our case is:
```
image_path,label
/image_dir/image_file_1.png, bonafide
/image_dir/image_file_2.png, bonafide
/image_dir/image_file_3.png, attack
/image_dir/image_file_4.png, attack
```
## Experiment
The main.py file can be used for training and test:
1. When training and test:
    ```
    python main.py \
      --train_csv_path 'train.csv' \
      --test_csv_path 'test.csv' \
      --model_path 'mixfacenet_SMDD.pth' \
      --is_train True \
      --is_test True \
      --output_dir 'output' \
    ```
2. When test by using pretrained weight, first download the model and give the model path:
    ```
    python main.py \
      --test_csv_path 'test.csv' \
      --model_path 'mixfacenet_SMDD.pth' \
      --is_train False \
      --is_test True \
      --output_dir 'output' \
    ```
More detailed information can be found in main.py.

##

**Citation:**

If you use SMDD dataset, please cite the following [paper](https://openaccess.thecvf.com/content/CVPR2022W/Biometrics/html/Damer_Privacy-Friendly_Synthetic_Data_for_the_Development_of_Face_Morphing_Attack_CVPRW_2022_paper.html):

```
@article{SMDD,
  author    = {Naser Damer and
               C{\'{e}}sar Augusto Fontanillo L{\'{o}}pez and
               Meiling Fang and
               No{\'{e}}mie Spiller and
               Minh Vu Pham and
               Fadi Boutros},
  title     = {Privacy-friendly Synthetic Data for the Development of Face Morphing
               Attack Detectors},
  booktitle = {{IEEE/CVF} Conference on Computer Vision and Pattern Recognition Workshops,
               {CVPR} Workshops 2022, New Orleans, LA, USA, June 19-20, 2022},
  pages     = {1605--1616},
  publisher = {{IEEE}},
  year      = {2022},
  url       = {https://doi.org/10.1109/CVPRW56347.2022.00167},
  doi       = {10.1109/CVPRW56347.2022.00167},
}
```


If you use the MixFaceNet-MAD, please cite the paper above and the original MixFaceNet paper ([repo](https://github.com/fdbtrs/mixfacenets), [paper](https://ieeexplore.ieee.org/document/9484374)):

```
@inproceedings{mixfacenet,
  author    = {Fadi Boutros and
               Naser Damer and
               Meiling Fang and
               Florian Kirchbuchner and
               Arjan Kuijper},
  title     = {MixFaceNets: Extremely Efficient Face Recognition Networks},
  booktitle = {International {IEEE} Joint Conference on Biometrics, {IJCB} 2021,
               Shenzhen, China, August 4-7, 2021},
  pages     = {1--8},
  publisher = {{IEEE}},
  year      = {2021},
  url       = {https://doi.org/10.1109/IJCB52358.2021.9484374},
  doi       = {10.1109/IJCB52358.2021.9484374},
}
```

The "IJCB-SYN-MAD-2022: Competition on Face Morphing Attack Detection Based on Privacy-aware Synthetic Training Data" ([PAPER](https://ieeexplore.ieee.org/document/10007950)) is based on the SMDD development data. The IJCB-SYN-MAD-2022 also released a real data evaluation bemchmark that can be found on its [REPO](https://github.com/marcohuber/SYN-MAD-2022).

##

**License:**

The dataset, the implementation, or trained models, use is restricted to research purpuses. The use of the dataset or the implementation/trained models for product development or product competetions (incl. NIST FRVT MORPH) is not allowed.
This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. Copyright (c) 2020 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt.
