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

3. run `py main.py` for deafault parameters or you can customize by call class IEC like: 

* IEC.download('name datasets')
=> These are 5 datasets you can download by type(still update more):
1. Corn Disease Dataset => IEC.download('Corn Dataset')
2. Wheat Disease Detection Dataset => 'Wheat Dataset'
3. Rice Leaf Disease Image Samples Dataset => 'Wheat Dataset'
4. The Potato Leaf Dataset => 'Potato Dataset'
5. iCassava 2019 Dataset => 'iCassava Dataset'

https://colab.research.google.com/drive/13WWKR97NsVPYmJOa_ojecC6aVdaO1WXk?usp=sharing
