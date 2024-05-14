import torch

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
    
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding = 1)
        self.conv2 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding = 1)
        self.batchnorm1 = torch.nn.BatchNorm2d(256)
        
        self.conv3 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding = 1)
        self.batchnorm2 = torch.nn.BatchNorm2d(512)
        
        self.conv4 = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=4, stride=2, padding = 1)
        self.batchnorm3 = torch.nn.BatchNorm2d(1024)
        
        self.conv5 = torch.nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=4, stride=1, padding = 1)
        
        self.leakyrelu = torch.nn.LeakyReLU(0.2)

        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        x = self.leakyrelu(self.conv1(x))
        x = self.leakyrelu(self.batchnorm1(self.conv2(x)))
        x = self.leakyrelu(self.batchnorm2(self.conv3(x)))
        x = self.leakyrelu(self.batchnorm3(self.conv4(x)))
        x = self.conv5(x)
        # x = x.view(-1, 1).squeeze(1)
        ##########       END      ##########
        
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
    
        self.conv1 = torch.nn.ConvTranspose2d(noise_dim, 1024, kernel_size=4, stride=1)
        self.batchnorm1 = torch.nn.BatchNorm2d(1024)
        self.conv2 = torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.batchnorm2 = torch.nn.BatchNorm2d(512)
        self.conv3 = torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.batchnorm3 = torch.nn.BatchNorm2d(256)
        self.conv4 = torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.batchnorm4 = torch.nn.BatchNorm2d(128)
        self.conv5 = torch.nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        
        # x = x.view(-1, self.noise_dim, 1, 1)
        
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.relu(self.batchnorm4(self.conv4(x)))
        x = self.tanh(self.conv5(x))
        
        ##########       END      ##########
        
        return x
    

