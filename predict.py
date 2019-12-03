# test a single image
from torchvision import models, transforms
from torch.autograd import Variable
import torch
import torch.nn as nn
from resnet import *

class Predictor:

    def __init__(self, ckpt_path='/home/hw3/ckpts/46000.pth'):
        self.model = wide_resnet50_2().to('cpu')
        self.model.eval()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        # load from the final checkpoint
        load_checkpoint(ckpt_path, self.model, self.optimizer)

    def predict(self, image):
        """
        image: (400,) numpy array
        """
        def preprocess(image):
            image = image.reshape(1, 20, 20)
            image = torch.from_numpy(image).float()
            return image
        
        img_tensor = preprocess(image)
        img_tensor.unsqueeze_(0) # turn image into a batch
        img_tensor = img_tensor.float()
        # print(img_tensor.shape)
        img_variable = Variable(img_tensor)
        prediction = self.model(img_variable)
        return prediction