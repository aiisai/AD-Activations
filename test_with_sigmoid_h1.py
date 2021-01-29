# test Sigmoids with hyper-parameter set1
import torch
import torch.nn as nn
import torch.optim as optim
import AD
import Additional
import sys
import logging


n_epochs = 300
#torch.backends.cudnn.enabled = False

#'Logistic', 'Improved_Logistic', 'AD_Logistic'

class LogisticLayers(nn.Module):
    def __init__(self, width=2, n_layers=10, type='ReLU'):
        super(LogisticLayers, self).__init__()

        self.n = n_layers

        if type == 'Logistic':
            self.activation = nn.Sigmoid()
        elif type == 'Improved_Logistic':
            self.activation = Additional.I_Sigmoid()
        elif type == 'AD_Logistic1':
            self.activation = AD.AD_Sigmoid1()
        elif type == 'AD_Logistic2':
            self.activation = AD.AD_Sigmoid2()
        else:
            self.activation = nn.Sigmoid()
            print('Error type!')

        layers = [nn.Linear(width, width, bias=True)]
        layers.append(self.activation)

        for i in range(n_layers-2):
            layers.append(nn.Linear(width, width, bias=True))
            layers.append(self.activation)

        layers.append(nn.Linear(width, 1, bias=True))
        layers.append(self.activation)

        self.model = nn.Sequential(*layers)
        self.init_weights()
        self.transformed = False

    def init_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                pass
                #nn.init.xavier_normal_(m.weight, gain=1.0)
                #nn.init.constant_(m.bias, 0)

    def forward(self, x, epoch, loss):
        x = self.model(x)
        return x


class DRNet(nn.Module):
    def __init__(self, width=2, n_layers=6, type=type):
        super(DRNet, self).__init__()
        self.layers = LogisticLayers(width=width, n_layers=n_layers, type=type)

    def init_weights(self):
        self.layers.init_weights()

    def forward(self, x, epoch, loss):
        x = self.layers(x, epoch, loss)
        return x


def train(epoch, optimizer, network, last_loss):
    #logger.info('\nTraining:')
    network.train()
    train_loss = 0
    inputs1 = (torch.sqrt(torch.tensor([12.])) * torch.rand(3000) - torch.sqrt(torch.tensor([3.]))).cuda()#.half()
    inputs2 = (torch.sqrt(torch.tensor([12.])) * torch.rand(3000) - torch.sqrt(torch.tensor([3.]))).cuda()
    inputs1 = inputs1.reshape([60, 50, 1])
    inputs2 = inputs2.reshape([60, 50, 1])
    inputs = torch.cat([inputs1,inputs2],dim=-1)
    targets = (inputs1<inputs2).float()
    for i, input in enumerate(inputs):
        target = targets[i]
        optimizer.zero_grad()
        output = network(input, epoch, last_loss)
        loss = criterion(output, target)
        train_loss += loss
        loss.backward()
        optimizer.step()
    train_loss /= 60
    return train_loss.item()


AD_times = -1
def test(epoch, network, target_loss):
    network.eval()
    test_loss = 0
    inputs1 = (torch.sqrt(torch.tensor([12.])) * torch.rand(500) - torch.sqrt(torch.tensor([3.]))).cuda()  # .half()
    inputs2 = (torch.sqrt(torch.tensor([12.])) * torch.rand(500) - torch.sqrt(torch.tensor([3.]))).cuda()
    inputs1 = inputs1.reshape([5, 100, 1])
    inputs2 = inputs2.reshape([5, 100, 1])
    inputs = torch.cat([inputs1, inputs2], dim=-1)
    targets = (inputs1<inputs2).float()
    with torch.no_grad():
        for i, input in enumerate(inputs):
            target = targets[i]
            output = network(input, epoch, -1)
            test_loss += criterion(output, target)
    test_loss /= 5
    print(str(epoch) + " test loss: {:.3e}".format(test_loss.item()))
    return test_loss.item()


def one_train(t, optimizer, network, logger):
    print("Within ID:"+str(t)+" task!")
    global AD_times
    AD_times = -1
    final_loss = test(0,network,1000)
    best_loss = final_loss
    final_epoch = 0
    for epoch in range(1, n_epochs + 1):
        train(epoch,optimizer,network,final_loss)
        loss = test(epoch,network,best_loss)
        if loss < best_loss:
            best_loss = loss
        #if epoch == n_epochs:
        final_loss = loss
        final_epoch = epoch
        if loss < .01:
            break
    success = 0
    if final_loss < .1 and best_loss < .1:
        success = 1
    logger.info('{:.3e},{:.3e},{:s},{:d},{:d}'.format(final_loss, best_loss, str(success>0), AD_times, final_epoch))
    return success

criterion = nn.MSELoss()
n_layers = 0
all_type = ['ReLU', 'LReLU', 'AD_ReLU',
            'Logistic', 'Improved_Logistic', 'AD_Logistic1', 'AD_Logistic2']

def main(argv):
    global n_layers
    n_layers = int(argv[1])
    type = argv[2]
    if type not in all_type:
        exit('Error activation type!')

    output_file = "experiments/HyperPara1/" + type + "_L" + str(n_layers) + "_Batch50_IDefault_B0_Epoch300-.csv"
    logging.basicConfig(filemode='a')
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(output_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    logger.addHandler(handler)
    logger.addHandler(console)

    success = 0
    for i in range(1):
        network = DRNet(n_layers=n_layers, type=type)
        total_params = sum(p.numel() for p in network.parameters())
        print("Total parameters: " + str(total_params))
        network = network.cuda()
        optimizer = optim.Adam(network.parameters())
        success += one_train(0, optimizer, network, logger)
    file = open(output_file)
    text = file.read()
    n_fails = text.count("False")
    n_success = text.count("True")
    print("Fail/Total: " + str(n_fails) + "/" + str(n_fails+n_success))

if __name__ == "__main__":
    main(sys.argv)