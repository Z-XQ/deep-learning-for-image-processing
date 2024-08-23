import torch
import torchvision
import torch.nn as nn
from model import LeNet
import torch.optim as optim
import torchvision.transforms as transforms


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ]
    )

    # 50000张训练图片
    # 第一次使用时要将download设置为True才会自动去下载数据集
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
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

    for epoch in range(200):  # loop over the dataset multiple times
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

            print('[%d] train_loss: %.3f' %(epoch + 1, loss.item()))

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
    [1] train_loss: 1.652  test_accuracy: 0.493
    [2] train_loss: 1.349  test_accuracy: 0.551
    [3] train_loss: 1.226  test_accuracy: 0.569
    [4] train_loss: 1.137  test_accuracy: 0.589
    [5] train_loss: 1.069  test_accuracy: 0.608
    [6] train_loss: 1.009  test_accuracy: 0.633
    [7] train_loss: 0.961  test_accuracy: 0.642
    [8] train_loss: 0.911  test_accuracy: 0.648
    [9] train_loss: 0.873  test_accuracy: 0.652
    [10] train_loss: 0.837  test_accuracy: 0.661
    [11] train_loss: 0.810  test_accuracy: 0.662
    [12] train_loss: 0.783  test_accuracy: 0.670
    [13] train_loss: 0.762  test_accuracy: 0.663
    [14] train_loss: 0.742  test_accuracy: 0.669
    Finished Training

    """