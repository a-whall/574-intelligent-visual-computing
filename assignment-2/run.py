from trainMVShapeClassifier import trainMVShapeClassifier
from testMVImageClassifier import testMVImageClassifier
import pickle as p
import torch

train_path = './dataset/train'
test_path = './dataset/test'

use_cuda = torch.cuda.is_available()

# TRAIN
model, info = trainMVShapeClassifier(train_path, cuda=use_cuda, verbose=False)

# TO SAVE TIME
# -------------------------------------------------------------------------------------------------------
# For testing a pre-trained model, replace the training function above with the following 2 lines of code
# -------------------------------------------------------------------------------------------------------
# model = torch.load('model/model_epoch_19.pth', map_location=lambda storage, location: storage)["model"]
# info = p.load( open( "info.p", "rb" ) )

# TEST
testMVImageClassifier(test_path, model, info, pooling='mean', cuda=use_cuda, verbose=False)
testMVImageClassifier(test_path, model, info, pooling='max', cuda=use_cuda, verbose=False)