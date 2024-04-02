import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=36,
                                               shuffle=True, num_workers=0)

    # 10000张验证图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    val_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=False, transform=transform)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=5000,
                                             shuffle=False, num_workers=0)
    val_data_iter = iter(val_loader)
    val_image, val_label = next(val_data_iter)
    
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    net = LeNet()
    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    for epoch in range(5):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data  # inputs(b,3,32,32), labels(b,).
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)  # outputs(b,10)
            loss = loss_function(outputs, labels)  # outputs(b,10), labels(b,).
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        net.train()
        with torch.no_grad():  # 进入eval模型，节省资源
            val_image = val_image.to(device)
            val_label = val_label.to(device)
            outputs = net(val_image)  # [batch, 10]
            # outputs(b,10). predict_y(b,). predict_y[3]=8
            predict_y = torch.max(outputs, dim=1)[1]
            accuracy = torch.eq(predict_y, val_label).sum().item() / val_label.size(0)

            print('[%d] train_loss: %.3f  test_accuracy: %.3f' %
                  (epoch + 1, running_loss / len(train_loader), accuracy))

    print('Finished Training')

    save_path = './Lenet.pth'
    torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    main()
    """
    Files already downloaded and verified
    [1,   500] train_loss: 1.706  test_accuracy: 0.461
    [1,  1000] train_loss: 1.421  test_accuracy: 0.534
    [2,   500] train_loss: 1.200  test_accuracy: 0.587
    [2,  1000] train_loss: 1.132  test_accuracy: 0.596
    [3,   500] train_loss: 1.015  test_accuracy: 0.640
    [3,  1000] train_loss: 1.000  test_accuracy: 0.653
    [4,   500] train_loss: 0.900  test_accuracy: 0.659
    [4,  1000] train_loss: 0.904  test_accuracy: 0.657
    [5,   500] train_loss: 0.822  test_accuracy: 0.660
    [5,  1000] train_loss: 0.843  test_accuracy: 0.642
    Finished Training

    """