# Vietnamese Food Recognition Project

## Description
This project implements a deep learning model to recognize and classify Vietnamese dishes from images. The system can identify various traditional Vietnamese foods, helping users learn about Vietnamese cuisine and its diverse dishes. Besides, this system is also capable of extracting ingredients from the food image 
## Features
- Image classification of Vietnamese dishes such as Pho, Banh mi and Mi Quang. You can add more food into your dataset for more diverse food recognition
- Real-time food recognition
- User-friendly interface for food recognition and ingredients extraction

## Technologies Used
- Python 3.x
- PyTorch
- NumPy
- Streamlit

## Dataset
This project utilizes the following datasets:
- [Vietnamese Food Image Dataset](https://www.kaggle.com/datasets/quandang/vietnamese-foods/data) from Kaggle

## Setup and Installation
1. Clone the repository
```bash
git clone https://github.com/DangCongKhai/VN_Food_Recognizer.git
cd VN_Food_Recognizer
```

2. Create and activate virtual environment
```bash
python -m venv .venv
source venv/bin/activate  # For Mac/Linux
```

3. Install required packages
```bash
pip install -r requirements.txt
```


## Model Architecture and Performance
I have only trained a simple CNN model. You can use **Transfer learning** from models such as `MobileV2Net`, `VGG16`, etc for better performance

```
        Layer (type)               Output Shape         Param #
            Conv2d-1         [-1, 32, 222, 222]             896
              ReLU-2         [-1, 32, 222, 222]               0
            Conv2d-3         [-1, 32, 220, 220]           9,248
              ReLU-4         [-1, 32, 220, 220]               0
         MaxPool2d-5         [-1, 32, 110, 110]               0
            Conv2d-6         [-1, 64, 108, 108]          18,496
              ReLU-7         [-1, 64, 108, 108]               0
            Conv2d-8        [-1, 128, 106, 106]          73,856
              ReLU-9        [-1, 128, 106, 106]               0
        MaxPool2d-10          [-1, 128, 53, 53]               0
AdaptiveAvgPool2d-11            [-1, 128, 1, 1]               0
          Flatten-12                  [-1, 128]               0
           Linear-13                   [-1, 64]           8,256
             ReLU-14                   [-1, 64]               0
          Dropout-15                   [-1, 64]               0
           Linear-16                   [-1, 32]           2,080
             ReLU-17                   [-1, 32]               0
           Linear-18                    [-1, 3]              99

Total params: 112,931
Trainable params: 112,931
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 86.73
Params size (MB): 0.43
Estimated Total Size (MB): 87.74
```

During model development, I split the dataset into 80% for training and 20% for validation. In general, my current model achieves around an accuracy of **73%** on the validation set.


## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset provided by [30VNFoods](https://www.kaggle.com/datasets/quandang/vietnamese-foods/data)
- Special thanks to the contributors of the Vietnamese food image datasets