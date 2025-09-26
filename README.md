<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/7bb294fb-8259-4a2e-b2c1-5b35b65976c5" /># CIFAR-100 Image Classification with CNN + Residual Blocks

This project implements a **Convolutional Neural Network (CNN)** enhanced with **Residual Blocks (ResNet-style)** for image classification on the **CIFAR-100** dataset using PyTorch.  
The model leverages skip connections to ease the training of deeper networks and improve accuracy.

---

## 📂 Project Structure

├── dataset.py # Code for loading CIFAR-100 dataset
├── model.py # CNN architecture with Residual Blocks
├── train.py # Training loop
├── evaluate.py # Evaluation functions
├── main.py # Entry point: train + evaluate
├── requirements.txt # Dependencies
└── README.md # Project description

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/thetrungpham/cifar100-classification.git
cd cifar100-classification
```
### 2.Install dependencies
pip install -r requirements.txt

### 3.Train model
python main.py


### 4. Evaluate on test set
python -c "from evaluate import evaluate; from dataset import cifar100_test; import torch; from model import Net; net=torch.load('best_model.pth'); print('Test acc:', evaluate(net, cifar100_test))"

## Results
- Test Accuracy: 0.7017%

- Plot train loss:
```bash
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()
```
<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/3469ec69-4129-4b85-9215-730c7852f19d" />

- Plot train accuracy:
```bash
plt.plot(train_accuracies)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
```
<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/9f0bb4b6-7ca3-476b-b199-cf1a2d592209" />

   

