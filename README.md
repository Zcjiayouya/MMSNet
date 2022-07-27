# MMS-Net
Thank the author of u2net(https://github.com/xuebinqin/U-2-Net) for giving me inspiration.  

Structure of MMS-Net  
https://raw.githubusercontent.com/Zcjiayouya/MMSNet/main/MMS-Net/mmsnet.png  

How to use this project:  
1.Operating system environment:This project supports both windows and Linux environments.  
  
2.Pycharm environment(Please install the rest by yourself):  
    albumentations:1.2.1  
    cudnn:6.0  
    numpy:1.19.5  
    opencv-contrib-python:4.5.5.64  
    opencv-python:4.5.1.48  
    opencv-python-headless:4.2.0.34  
    pillow:6.2.2  
    python:3.8.8  
    pytorch:1.11.0  
    scikit-image:0.18.1  
    scikit-learn:0.24.1  
    scipy:1.7.3  
    torch:1.11.0  
  
3.Storage location of data:  
    MMS-Net/data/CZ/Train  

4.File structure of data  
    MMS-Net
    --data
    ----ZC
    ------Train
    --------Image
    --------Mask
    --------temp
    --------predict_result
    --------test_image
    --------test_mask
  
5.How to train the model:  
    Our model is modified based on u2net, the model has been modified to MMS-Net.  
    Please click on MMSNet_train, and then it will train. 
  
6.How to evaluate the performance of the model:  
   You need to modify six places of u2net_metric file.  
      Line 171 represent the pictures you need to predict.  
      Line 172 represent the corresponding labels of the pictures.  
      Line 173 represents an empty folder and holds the intermediate process.  
      Line 174 represents an empty folder and stores the results.  
      The path of the folder in line 51 is the same as that in line 174.  
      Finally, modify the model name in line 198, and its path is located in MMS-Net/saved_ models/MMS-Net  
