from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from os.path import join
from torch.nn.functional import one_hot
import numpy as np

from utils.dataset import CustomDataset
from utils.model import Net
from utils.utils import ic

def eval_model(model, device, eval_loader):
    """
    评估模型
    Args:
        model: 模型
        device: 训练device
        eval_loader: 测试数据加载器

    Returns: 模型评估指标

    """

    model.eval()
    eval_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in eval_loader:
            data, target = data.to(device), target.squeeze(1).to(device)
            output = model(data)
            eval_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    eval_loss /= len(eval_loader.dataset)

    print('\nEval set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        eval_loss, correct, len(eval_loader.dataset),
        100. * correct / len(eval_loader.dataset)))

    return eval_loss


def train_model(args, model, device, train_loader, optimizer, epoch):

    """
    训练函数
    :param args: 训练参数
    :param model: 模型
    :param device: 训练device
    :param train_loader: 训练数据加载器
    :param optimizer: 优化器
    :param epoch: 训练轮次
    :return:
    """

    model.train()

    # 遍历数据加载器
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.squeeze(1).to(device)  # [bs, 3] / [bs]

        optimizer.zero_grad()

        output = model(data)
        # 使用负对数似然损失函数(negative log likelihood loss)计算分类损失
        loss = F.nll_loss(output, target)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 打印训练日志
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def main():
    """
    主函数
    :return:
    """

    # 训练参数
    parser = argparse.ArgumentParser(description='Convolutional neural network example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-5, metavar='LR',
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--save_path', type=str, default='./ckpts')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    # 设置随机种子
    torch.manual_seed(args.seed)

    # 选择模型训练device
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    # 定义数据集
    dataset1 = CustomDataset(num_samples=10000, image_shape=(200, 200, 3))
    dataset2 = CustomDataset(num_samples=1000, image_shape=(200, 200, 3))

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    eval_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    
    print('Device:', device)

    # 模型构建
    model = Net().to(device)
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    max_loss = -np.inf
    max_epoch = 0
    epoch_iter = 5  # 早停轮次, 5轮没有提升就停止训练
    for epoch in range(1, args.epochs + 1):
        # 训练模型
        train_model(args, model, device, train_loader, optimizer, epoch)
        # 评估模型
        eval_loss = eval_model(model, device, eval_loader)
        if eval_loss > max_loss:
            max_loss = eval_loss
            max_epoch = epoch
            torch.save(model, join(args.save_path, str(max_epoch)) + '.pt')
        else:
            # 早停
            if epoch - max_epoch >= epoch_iter:
                final_model = torch.load(join(args.save_path, str(max_epoch)) + '.pt')
                torch.save(final_model.state_dict(), join(args.save_path, 'final.pt'))
                print('Early stop at epoch {}. Model saved !'.format(max_epoch))
                break
        scheduler.step()

    print('Training finished!')


if __name__ == '__main__':

    main()    
    