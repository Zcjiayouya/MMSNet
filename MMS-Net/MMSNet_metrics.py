import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
import cv2 as cv

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model.MMSNet import MMSNet # full size version 173.6 MB
import sklearn.metrics as metrics111
# normalize the predicted SOD probability map


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)
    # dn=torch.clamp(d,0,1)
    return dn

def setThershold(image_path):
    global right_masks
    for root, dirs, files in os.walk(image_path, topdown=False):
        right_masks = [file for file in files]

    for i in range(len(right_masks)):
        print(str(i) + "start")
        right_img = cv.imread(image_path + right_masks[i], 0)
        # left_img=cv.imread(left_mask_path+left_masks[i])
        for k in range(len(right_img)):
            for m in range(len(right_img[k])):
                if (right_img[k][m] < 127):
                    right_img[k][m] = 0
                else:
                    right_img[k][m] = 255
        for root, dirs, files in os.walk(image_path, topdown=False):
            cv.imwrite(r"E:\study\MMS-Net\data\CZ\Train\predict_result/" + right_masks[i], right_img)
            break


def save_output(image_name,pred,d_dir):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    im = Image.fromarray(predict_np*255).convert('RGB')   #yuanlai
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)

    # imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR) yuanlai
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    imo.save(d_dir+imidx+'.png')

__all__ = ['SegmentationMetric']

def get_contours(img):
    """获取连通域

    :param img: 输入图片
    :return: 最大连通域
    """
    # 灰度化, 二值化, 连通域分析
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, img_bin = cv.threshold(img_gray, 200, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(img_bin, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    return contours[0]

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵（空）

    def pixelAccuracy(self):
        # return all class overall pixel accuracy 正确的像素占总像素的比例
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        """
        Mean Pixel Accuracy(MPA，均像素精度)：是PA的一种简单提升，计算每个类内被正确分类像素数的比例，之后求所有类的平均。
        :return:
        """
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        return IoU

    def meanIntersectionOverUnion(self):
        mIoU = np.nanmean(self.IntersectionOverUnion())  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  #
        """
        同FCN中score.py的fast_hist()函数,计算混淆矩阵
        :param imgPredict:
        :param imgLabel:
        :return: 混淆矩阵
        """
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        # print(confusionMatrix)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        """
        FWIoU，频权交并比:为MIoU的一种提升，这种方法根据每个类出现的频率为其设置权重。
        FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        """
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)  # 得到混淆矩阵
        return self.confusionMatrix

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

def main():

    # --------- 1. get image path and name ---------
    model_name='MMS-Net'

    image_dir = r'E:\study\MMS-Net\data\CZ\Train\test_image/'
    test_mask_dir = r"E:\study\MMS-Net\data\CZ\Train\test_mask/"
    temp_dir = r'E:\study\MMS-Net\data\CZ\Train\temp/'
    result_dir=r"E:\study\MMS-Net\data\CZ\Train\predict_result/"


    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    # print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=0)

    # --------- 3. model define ---------
    net = MMSNet(3, 1)

    # net.load_state_dict(torch.load(model_dir))
    model_name="MMS-Net_left_3520_comp_itr_10_train_5.112929_tar_0.633511"
    net.load_state_dict(torch.load('./saved_models/MMS-Net/'+model_name+'.pth')) #加载自己的模型
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
        save_output(img_name_list[i_test], pred, temp_dir)

        del d1,d2,d3,d4,d5,d6,d7

    setThershold(temp_dir)

    smooth_num = 0.000001

    for root, dirs, files in os.walk(test_mask_dir, topdown=False):
        true_labels = [file for file in files]

    for root, dirs, files in os.walk(temp_dir, topdown=False):
        predict_labels = [file for file in files]

    result_confusion = [[0 for i in range(2)] for i in range(2)]
    mae = 0
    hd95 = 0
    sapmle_num = 0

    for i in range(len(predict_labels)):
        print(str(i) + " start")
        sapmle_num += 1
        imgPredict = cv.imread( result_dir+ predict_labels[i])
        imgLabel = cv.imread(test_mask_dir + true_labels[i])

        imgPredict = np.array(cv.cvtColor(imgPredict, cv.COLOR_BGR2GRAY) / 255., dtype=np.uint8)
        imgLabel = np.array(cv.cvtColor(imgLabel, cv.COLOR_BGR2GRAY) / 255., dtype=np.uint8)

        if(imgLabel.shape!=imgPredict.shape):
            imgPredict=imgPredict.transpose()

        metric = SegmentationMetric(2)  # 2表示有2个分类，有几个分类就填几
        hist = metric.addBatch(imgPredict, imgLabel)
        temp = np.array(hist)
        result_confusion[0][0] += temp[0][0]
        result_confusion[0][1] += temp[0][1]
        result_confusion[1][0] += temp[1][0]
        result_confusion[1][1] += temp[1][1]

        mae += metrics111.median_absolute_error(imgLabel, imgPredict)

    TN = result_confusion[0][0]
    FN = result_confusion[0][1]
    FP = result_confusion[1][0]
    TP = result_confusion[1][1]
    mae = (mae ) / (sapmle_num+smooth_num)
    ACC = (TP + TN) / (TP + FP + FN + TN+smooth_num)
    Sen = (TP) / (TP + FN+smooth_num)
    Spe = (TN) / (TN + FP+smooth_num)
    Pre = (TP) / (TP + FP+smooth_num)
    F1 = (2 * Pre * Sen) / (Pre + Sen+smooth_num)
    print("confusion matrix:", result_confusion)
    print("Dice Similarity Coefficient:", (TP * 2) / (FP + TP * 2 + FN+smooth_num))
    print("mae:", mae)
    print("ACC:", ACC)
    print("Sen:", Sen)
    print("Spe:", Spe)
    print("Pre:", Pre)
    print("F1:", F1)


if __name__ == "__main__":
    main()
