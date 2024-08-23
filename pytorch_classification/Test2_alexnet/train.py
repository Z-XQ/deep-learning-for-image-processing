import os
import sys
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16
    # nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 0  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    test_data_iter = iter(validate_loader)
    test_image, test_label = next(test_data_iter)

    # def imshow(img):
    #     img = img / 2 + 0.5  # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #     plt.show()
    #
    # print(' '.join('%5s' % cla_dict[test_label[j].item()] for j in range(4)))
    # imshow(utils.make_grid(test_image))

    net = AlexNet(num_classes=5, init_weights=True)

    net.to(device)
    loss_function = nn.CrossEntropyLoss()
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)

    epochs = 10
    save_path = './AlexNet.pth'
    best_acc = 0.0
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = net(images)  # images(b,3,224,224). outputs(b,5)
            loss = loss_function(outputs, labels)  # labels(b,5). labels[6] = 4
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data  # val_labels(b,)
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                outputs = net(val_images.to(device))  # outputs(b,5)
                predict_y = torch.max(outputs, dim=1)[1]  # predict_y(b,)
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

        val_accurate = acc / val_num  # 整个验证集合的准确率
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    """ libtorch with no rand init weight
    Epoch: 0, Training Loss: 0.74375838041305542
    Epoch: 1, Training Loss: 0.90064239501953125
    Epoch: 2, Training Loss: 1.34159445762634284
    Epoch: 3, Training Loss: 0.75602966547012329
    Epoch: 4, Training Loss: 0.70253342390060425
    Epoch: 5, Training Loss: 0.90282332897186279
    Epoch: 6, Training Loss: 0.47993755340576172
    Epoch: 7, Training Loss: 0.86767882108688354
    Epoch: 8, Training Loss: 0.78316569328308105
    Epoch: 9, Training Loss: 0.52741509675979614
    Epoch: 10, Training Loss: 0.58984518051147461
    
    libtorch with kaiming_normal_
    Epoch: 0, Training Loss: 1.57387423515319829
    Epoch: 1, Training Loss: 0.88000547885894775
    Epoch: 2, Training Loss: 0.91211783885955811
    Epoch: 3, Training Loss: 0.54084092378616333
    Epoch: 4, Training Loss: 0.98211300373077393
    Epoch: 5, Training Loss: 0.71742963790893555
    Epoch: 6, Training Loss: 0.14369988441467285
    Epoch: 7, Training Loss: 0.36351940035820007
    Epoch: 8, Training Loss: 0.45700043439865112
    Epoch: 9, Training Loss: 0.53762894868850708
    Epoch: 10, Training Loss: 0.46801453828811646
    """

    """pytorch
    [epoch 1] train_loss: 1.371  val_accuracy: 0.456
    [epoch 2] train_loss: 1.178  val_accuracy: 0.530
    [epoch 3] train_loss: 1.099  val_accuracy: 0.615
    [epoch 4] train_loss: 1.016  val_accuracy: 0.624
    [epoch 5] train_loss: 0.977  val_accuracy: 0.673
    [epoch 6] train_loss: 0.946  val_accuracy: 0.668
    [epoch 7] train_loss: 0.910  val_accuracy: 0.643
    [epoch 8] train_loss: 0.866  val_accuracy: 0.596
    [epoch 9] train_loss: 0.879  val_accuracy: 0.654
    [epoch 10] train_loss: 0.821  val_accuracy: 0.692
    Finished Training

    """

    main()
