import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.nn import functional as F

class ConvNet(nn.Module):
    def __init__(self,hyper_params):
        super(ConvNet, self).__init__()
        self.name = "conv"
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5) #out put has (W-F+2P)/S+1 dim (W input dim,
                                                     #F receptive or kernel size,)
                                                     #P padding
                                                     #S stride | read more in pytorch docs
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(256, hyper_params['hidden_n'])
        self.fc2 = nn.Linear(hyper_params['hidden_n'], 10)
        self.num_epochs = hyper_params['num_epochs']
        self.criterion = nn.MSELoss()  
        self.optimizer = torch.optim.Adam(self.parameters(), lr=hyper_params['learning_rate'])  
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=3, stride=3)) # 8x8x32
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)) # 2x2x64
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x

    def train(self, train_loader):
        total_step = len(train_loader)
        #train_loader =  train_loader.cuda()
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(train_loader):
                # Reshape images to (batch_size, input_size)
                images = images.view(-1, 1, 28, 28)

                # Forward pass
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                outputs = self(images)
                target = self.convert_to_one_hot_labels(images, labels)
                loss = self.criterion(outputs, target)

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if (i+1) % 100 == 0:
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                           .format(epoch+1, self.num_epochs, i+1, total_step, loss.item()))
                    
    def train_a_batch(self, images, labels):
        # Reshape images to (batch_size, input_size)
        images = images.view(-1, 1, 28, 28)

        # Forward pass
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def test(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(-1, 1, 28, 28)
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
                classes_prob = self(images)
                classes_prob = classes_prob.max(1)[1]
                test_target = self.convert_to_one_hot_labels(images, labels)
                total += labels.size(0)
                nb_test_errors = 0
                for n in range(0, classes_prob.size()[0]):
                    if test_target[n,classes_prob[n]] < 0:
                        nb_test_errors = nb_test_errors + 1
                correct += classes_prob.size()[0] - nb_test_errors
            print('Accuracy of the model \'{}\' on the 10000 test images: {} %'.format(self.name,100 * correct / total))
            return 100 * correct / total
            
    def convert_to_one_hot_labels(self, input, target):
        tmp = input.new(target.size(0), 10).fill_(-1)
        tmp.scatter_(1, target.view(-1, 1), 1.0)
        return tmp
        
