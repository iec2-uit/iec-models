# Leaf Disease Dataset Framework
This repository is the official instruction of Leaf Disease Dataset Framework, a cloud collection datasets published from article: "**Towards Sustainable Agriculture: A Lightweight Hybrid Model and Cloud-based Collection of Datasets for Efficient Leaf Disease Detection**".

This framework provides 38 public leaf disease datasets, collected from research platforms (e.g., Google Scholar, IEEE Xplore, Kaggle, Data Mendeley).
### How to use this repository.
1. Clone this repository: !git clone https://github.com/iec2-uit/iec-models
2. Install prerequisites libraries: 
* **cd** into `train-iec-models` and install dependencies package: 
  * `pip install -r requirements.txt`   
* or run this: 
  * !pip3 install timm
  * !pip3 install pydicom
  * !pip3 install git+https://github.com/albumentations-team/albumentations
  * !pip3 install catalyst
  * !pip install -U albumentations==1.3.0
  * !pip install utils
3. Run `python3 main.py` or colab notebook (https://colab.research.google.com/drive/1isede0YB5hRCXHJ4nC-tVWi_wOVx4qQW?usp=sharing
) with your parameters and models via the supported pipeline from our IEC library with 6 functions:

* IEC.download('name datasets')

  * This function supports download available leaf disease datasets from our cloud.
These are currently 31 datasets you can download by name as described [here](https://github.com/iec2-uit/iec-models/releases/tag/List_of_Dataset_names_v1.0).

* IEC.seed_everything('seed value')

  * This function sets a seed value for pseudo-random number generators, which guarantees the reproductivity of deep learning algorithm implemented by Pytorch.

* IEC.folds()

  * This function splits dataset into k consecutive folds, which keeps the ratio between classes in each fold constant as in the original dataset.

* IEC.preprare_dataloader()

  * This function prepares the training and evaluation datasets according to the training/evaluation ratio.

* IEC.train_one_epoch()
  * This function finds the best combination of weights and bias for minimizing the loss value.
* IEC.valid_one_epoch()
  * This function evaluates the model performance after training with data.

### If you use this github for a paper please cite:
@article{thai2023towards,
  title={Towards sustainable agriculture: A lightweight hybrid model and cloud-based collection of datasets for efficient leaf disease detection},
  author={Thai, Huy-Tan and Le, Kim-Hung and Nguyen, Ngan Luu-Thuy},
  journal={Future Generation Computer Systems},
  year={2023},
  publisher={Elsevier}
}


