import torch
import torchvision
import numpy as np
import pickle

load_file = open("train_ali_mnist_epoch_200.bin", "rb")
trainlv = pickle.load(load_file)
trainlabel = pickle.load(load_file)
print(np.shape(trainlv))
print(np.shape(trainlabel))
trainset = torch.utils.data.TensorDataset(trainlv, trainlabel)
loader = torch.utils.data.DataLoader(
    dataset = trainset,
    batch_size = 100,
    shuffle = False,
    num_workers = 14,
)

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

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
lossfunc = torch.nn.CrossEntropyLoss().cuda()

def AccuarcyCompute(pred,label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
#     print(pred.shape(),label.shape())
    test_np = (np.argmax(pred,1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)


for epoch in range(500):
    scheduler.step()
    for step, (trainlv, trainlabel) in enumerate(loader):
        optimizer.zero_grad()

        inputs = torch.autograd.Variable(trainlv).cuda()
        labels = torch.autograd.Variable(trainlabel).cuda()

        outputs = model(inputs)

        labels = labels.long()
        loss = lossfunc(outputs, labels)
        loss.backward()

        optimizer.step()
        
    # for param_group in optimizer.param_groups:
    #     print(param_group['lr'])

    print('Epoch: ', epoch, '| Loss: ', loss)
    # if epoch%10 == 0:
    #     print('Epoch: ', epoch, '| Step: ', step,'| Accuarcy: ', AccuarcyCompute(outputs, labels))

torch.save(model.state_dict(), 'mlp_ali_mnist_epoch_200.pkl')

load_testfile = open("test_ali_mnist_epoch_200.bin", "rb")
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
