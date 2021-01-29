# test ReLUs with hyper-parameter set2
import torch
import torch.nn as nn
import torch.optim as optim
import AD
import sys
import logging


n_epochs = 300
d_epoch = [100, 200]
learning_rate = 1e-2
gamma = 0.5
momentum = 0.0


class ReLULayers(nn.Module):
    def __init__(self, width=2, n_layers=6, type='ReLU'):
        super(ReLULayers, self).__init__()
        #self.d = 3/(n_layers-2)
        self.a = torch.FloatTensor([3.5]).cuda()#.half()
        #self.a = torch.nn.Parameter(torch.FloatTensor([1]).cuda())#.half()
        self.n = n_layers

        if type == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif type == 'LReLU':
            self.activation = nn.LeakyReLU(inplace=True)
        elif type == 'AD_ReLU1':
            self.activation = AD.AD_ReLU1(self.a)
        elif type == 'AD_ReLU2':
            self.activation = AD.AD_ReLU2(self.a)
        else:
            self.activation = nn.ReLU(inplace=True)
            print('Error type!')

        layers = [nn.Linear(1, width, bias=True)]
        layers.append(self.activation)
        for i in range(n_layers-2):
            layers.append(nn.Linear(width, width, bias=True))
            layers.append(self.activation)
        layers.append(nn.Linear(width, 1, bias=True))
        self.model = nn.Sequential(*layers)
        self.init_weights()
        self.transformed = False

    def transform(self):
        if not self.transformed:
            for i,m in enumerate(self.model):
                if isinstance(m, AD.AD_ReLU1) or isinstance(m, AD.AD_ReLU2):
                    self.model[i] = nn.ReLU(inplace=True)
            self.transformed = True
            print("Model transformed to ReLU!")  # print(self.model)
        else:
            for i,m in enumerate(self.model):
                if isinstance(m, nn.ReLU):
                    self.model[i] = self.activation
            self.transformed = False
            print("Model transformed to AD!")   #print(self.model)

    def init_weights(self):
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                #nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, epoch, loss):
        x = self.model(x)
        return x


class DRNet(nn.Module):
    def __init__(self, width=2, n_layers=6, type='ReLU'):
        super(DRNet, self).__init__()
        self.layers = ReLULayers(width=width, n_layers=n_layers, type=type)

    def init_weights(self):
        self.layers.init_weights()

    def forward(self, x, epoch, loss):
        x = self.layers(x, epoch, loss)
        return x


def train(epoch, optimizer, network, last_loss):
    #logger.info('\nTraining:')
    network.train()
    train_loss = 0
    inputs = (torch.sqrt(torch.tensor([12.])) * torch.rand(3000) - torch.sqrt(torch.tensor([3.]))).cuda()#.half()
    inputs = inputs.reshape([30,100,1])
    for i, input in enumerate(inputs):
        target = torch.abs(input)
        optimizer.zero_grad()
        output = network(input, epoch, last_loss)
        loss = criterion(output, target)
        train_loss += loss
        loss.backward()
        optimizer.step()
    train_loss /= 30
    return train_loss.item()
    #print(str(epoch) + " train loss: {:.3e}".format(test_loss.item()))
    #print(network.layers.a.item())


AD_times = -1
def test(epoch, network, type, target_loss):
    network.eval()
    test_loss = 0
    inputs = (torch.sqrt(torch.tensor([12.])) * torch.rand(1000) - torch.sqrt(torch.tensor([3.]))).cuda()#.half()
    inputs = inputs.reshape([10,100,1])
    with torch.no_grad():
        for i, input in enumerate(inputs):
            target = torch.abs(input)
            output = network(input, epoch, -1)
            test_loss += criterion(output, target)
    test_loss /= 10
    print(str(epoch) + " test loss: {:.3e}".format(test_loss.item()))

    global AD_times
    # r = np.random.randint(1, 101)
    if type == 'AD_ReLU1' or type == 'AD_ReLU2':
        if not network.layers.transformed:
            AD_times += 1
            if test_loss > 1.06 * target_loss:  # and r > 66:
                network.layers.transform()
        else:
            if test_loss > 1.03 * target_loss:  # and r > 66:
                network.layers.transform()
    return test_loss.item()

def adjust_learning_rate(optimizer, epoch):
    global learning_rate
    if epoch in d_epoch:
        learning_rate *= gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate

def one_train(t, optimizer, network, type, logger):
    print("Within ID:"+str(t)+" task!")
    global AD_times
    AD_times = -1
    final_loss = test(0, network, type, 1000)
    best_loss = final_loss
    final_epoch = 0
    for epoch in range(1, n_epochs + 1):
        train(epoch,optimizer,network,final_loss)
        loss = test(epoch, network, type, best_loss)
        if loss < best_loss:
            best_loss = loss
        #if epoch == n_epochs:
        final_loss = loss
        final_epoch = epoch
        if loss < 0.001:
            break
        adjust_learning_rate(optimizer, epoch)
        #logger.info('{:d}      {:.3e}'.format(epoch + 1, loss))
    success = 0
    if final_loss < .01 and best_loss < .01:
        success = 1
    logger.info('{:.3e},{:.3e},{:s},{:d},{:d}'.format(final_loss, best_loss, str(success>0), AD_times, final_epoch))
    return success

criterion = nn.MSELoss()
n_layers = 0
all_type = ['ReLU', 'LReLU', 'AD_ReLU1', 'AD_ReLU2',
            'Logistic', 'Improved_Logistic', 'AD_Logistic1', 'AD_Logistic2']

def main(argv):
    global n_layers
    n_layers = int(argv[1])
    type = argv[2]
    if type not in all_type:
        exit('Error activation type!')

    output_file = "experiments/HyperPara2/" + type + "_L" + str(n_layers) + "_Batch100_IHe_B0_Epoch300-.csv"
    logging.basicConfig(filemode='a')
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    # handler = logging.FileHandler("log/"+logfile+".txt")
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
        network = DRNet(n_layers=n_layers, width=2, type=type)
        total_params = sum(p.numel() for p in network.parameters())
        print("Total parameters: " + str(total_params))
        network = network.cuda()
        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
        success += one_train(0, optimizer, network, type, logger)
    file = open(output_file)
    text = file.read()
    n_fails = text.count("False")
    n_success = text.count("True")
    print("Fail/Total: " + str(n_fails) + "/" + str(n_fails+n_success))

if __name__ == "__main__":
    main(sys.argv)