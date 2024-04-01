import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F

from model import LeNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    net = LeNet()
    net.to(device)

    net.load_state_dict(torch.load('Lenet.pth'))

    im = Image.open('plane.png')
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]

    with torch.no_grad():
        im = im.to(device)
        outputs = net(im)  # (1,10)
        outputs = F.softmax(outputs, dim=1)  # 可以省略
        predict = torch.max(outputs, dim=1)[1].cpu().numpy()  # (1,)
    print(classes[int(predict)])


if __name__ == '__main__':
    main()
