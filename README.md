# train-iec-models
### How to run this repository.
1. Clone this repository: !git clone https://github.com/nvhieu-04/train-iec-models
2. **cd** into `train-iec-models` and install dependencies package: `pip install -r requirements.txt` 
or run this: 
* !pip3 install timm
* !pip3 install pydicom
* !pip3 install git+https://github.com/albumentations-team/albumentations
* !pip3 install catalyst
* !pip install -U albumentations
3. run `py main.py` for deafault parameters or you can customize by call class IEC like: 
* IEC.download()
* IEC.seed_everything()
* IEC.prepare_dataloader()
* IEC.train_one_epoch()
* IEC.valid_one_epoch()

https://colab.research.google.com/drive/13WWKR97NsVPYmJOa_ojecC6aVdaO1WXk?usp=sharing
