# train-iec-models
### How to run this repository.
1. Clone this repository: !git clone https://github.com/iec2-uit/iec-models
2. **cd** into `train-iec-models` and install dependencies package: `pip install -r requirements.txt` 
or run this: 

* !pip3 install timm
* !pip3 install pydicom
* !pip3 install git+https://github.com/albumentations-team/albumentations
* !pip3 install catalyst
* !pip install -U albumentations

3. run `python3 main.py` for deafault parameters or you can customize by call class IEC like: 

* IEC.download('name datasets')
=> These are 25 datasets you can download by type(still update more):
1. A-Citrus-Fruits-and-Leaves-Dataset => IEC.download("A-Citrus-Fruits-and-Leaves-Dataset")
2. Citrus-Leaves-Prepared-Dataset => IEC.download("Citrus-Leaves-Prepared-Dataset")
3. Corn-Disease-Dataset => IEC.download("Corn-Disease-Dataset")
4. Corn-Leaf-Diseases-Dataset => IEC.download("Corn-Leaf-Diseases-Dataset")
5. Corn-Leaf-Infection-Dataset => IEC.download("Corn-Leaf-Infection-Dataset")
6. DiaMOS-Dataset => IEC.download("DiaMOS-Dataset")
7. LeLePhid-Dataset => IEC.download("LeLePhid-Dataset")
8. Red-Rot-Sugarcane-Disease-Leaf-Dataset => IEC.download("Red-Rot-Sugarcane-Disease-Leaf-Dataset")
9. Rice-Disease-Dataset => IEC.download("Rice-Disease-Dataset")
10. Rice-Diseases-Image-Dataset => IEC.download("Rice-Diseases-Image-Dataset")
11. Rice-Leaf-Disease-Image-Samples-Dataset => IEC.download("Rice-Leaf-Disease-Image-Samples-Dataset")
12. Rice-Leaf-Diseases-Dataset => IEC.download("Rice-Leaf-Diseases-Dataset")
13. RoCoLe-Dataset => IEC.download("RoCoLe-Dataset")
14. Sugarcane-Disease-Dataset => IEC.download("Sugarcane-Disease-Dataset")
15. The-Cotton-Leaf-Dataset => IEC.download("The-Cotton-Leaf-Dataset")
16. The-Cotton-Leaf-Disease-Dataset => IEC.download("The-Cotton-Leaf-Disease-Dataset")
17. The-Dhan-Shomadhan-Dataset => IEC.download("The-Dhan-Shomadhan-Dataset")
18. The-Potato-Leaf-Dataset => IEC.download("The-Potato-Leaf-Dataset")
19. The-Soybean-Leaf-Dataset => IEC.download("The-Soybean-Leaf-Dataset")
20. The-Tomato-Leaf-Image-Dataset => IEC.download("The-Tomato-Leaf-Image-Dataset")
21. Wheat-Disease-Detection-Dataset => IEC.download("Wheat-Disease-Detection-Dataset")
22. Wheat-Fungi-Diseases-Dataset => IEC.download("Wheat-Fungi-Diseases-Dataset")
23. Wheat-Leaf-Dataset => IEC.download("Wheat-Leaf-Dataset")
24. Yellow-Rush-19-Dataset => IEC.download("Yellow-Rush-19-Dataset")
25. iCassava-2019-Dataset => IEC.download("iCassava-2019-Dataset")


* IEC.seed_everything(IEC.CFG['seed'])
=> Function that sets seed for pseudo-random number generators
* IEC.folds
=> Split dataset into k consecutive folds
* IEC.preprare_dataloader()
* IEC.train_one_epoch()
* IEC.valid_one_epoch()

https://colab.research.google.com/drive/13WWKR97NsVPYmJOa_ojecC6aVdaO1WXk?usp=sharing
