import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=(1, 1), padding=2),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=5, stride=(1, 1), padding=2),
                                   nn.ReLU(), nn.MaxPool2d(kernel_size=2))

        # Fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # Flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)

        return output, x  # Return x for visualization


def train(num_epochs, cnn, loaders):
    cnn.train()

    # Train the model
    total_step = len(loaders['train'])

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(loaders['train']):
            # Gives batch data, normalize x when iterate train_loader
            b_x = Variable(images.cuda())  # Batch x
            b_y = Variable(labels.cuda())  # Batch y

            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)

            # Clear gradients for this training step
            optimizer.zero_grad()

            # Backpropagation, compute gradients
            loss.backward()

            # Apply gradients
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))


def test(cnn, loaders):
    # Test the model
    cnn.eval()
    with torch.no_grad():
        for images, labels in loaders['test']:
            images, labels = images.cuda(), labels.cuda()
            test_output, last_layer = cnn(images)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (pred_y == labels).sum().item() / float(labels.size(0))
        print('Test Accuracy of the model on the 10000 test images: %.2f' % accuracy)


if __name__ == '__main__':
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Use Hardware: ", "GPU" if torch.cuda.is_available() else "CPU")

    train_data = datasets.MNIST(root='data', train=True, transform=ToTensor(), download=True)
    test_data = datasets.MNIST(root='data', train=False, transform=ToTensor())

    # Preparing data for training
    loaders = {
        'train': torch.utils.data.DataLoader(train_data, batch_size=100, shuffle=True, num_workers=1),
        'test': torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=1),
    }

    # Setting CNN
    cnn = CNN().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.01)
    num_epochs = 10

    # Train
    train(num_epochs, cnn, loaders)

    # Validation
    test(cnn, loaders)
