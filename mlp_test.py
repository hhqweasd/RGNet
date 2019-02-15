import torch
import torchvision
import numpy as np
import pickle

class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(64, 8192)
        self.fc2 = torch.nn.Linear(8192, 8192)
        self.fc3 = torch.nn.Linear(8192, 10)

    def forward(self, din):
        din = din.view(-1, 64)
        dout = torch.nn.functional.relu(self.fc1(din))
        dout = torch.nn.functional.relu(self.fc2(dout))
        # dout = torch.nn.functional.relu(self.fc3(dout))
        # dout = torch.nn.functional.relu(self.fc4(dout))
        # print(np.shape(dout))
        return torch.nn.functional.softmax(self.fc3(dout))

model = MLP().cuda()
model.load_state_dict(torch.load('mlp_g_mnist_epoch_300.pkl'))
# print(model)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lossfunc = torch.nn.CrossEntropyLoss().cuda()

def AccuarcyCompute(pred,label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
#     print(pred.shape(),label.shape())
    test_np = (np.argmax(pred,1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)

load_testfile = open("test_g_mnist_epoch300.bin", "rb")
testlv = pickle.load(load_testfile)
testlabel = pickle.load(load_testfile)
testset = torch.utils.data.TensorDataset(testlv, testlabel)

accuarcy_list = []
for i, (inputs, labels) in enumerate(testset):
    inputs = torch.autograd.Variable(inputs).cuda()
    labels = torch.autograd.Variable(labels).cuda()
    labels = labels.long()
    outputs = model(inputs)
    accuarcy_list.append(AccuarcyCompute(outputs, labels))
print('Test Accuarcy: ', sum(accuarcy_list) / len(accuarcy_list))
