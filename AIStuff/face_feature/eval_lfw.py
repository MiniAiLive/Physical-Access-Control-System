import os
import argparse
import numpy as np
import scipy.io
import onnxruntime as ort
import torch.utils.data
import torchvision.transforms as transforms
from torch.nn import DataParallel
from model import mobilefacenet, resnet, cbam
from dataloader.lfw import LFW, LFWDataset


def getAccuracy(scores, flags, threshold):
    p = np.sum(scores[flags == 1] > threshold)
    n = np.sum(scores[flags == -1] < threshold)
    return 1.0 * (p + n) / len(scores)

def getThreshold(scores, flags, thrNum):
    accuracys = np.zeros((2 * thrNum + 1, 1))
    thresholds = np.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])
    max_index = np.squeeze(accuracys == np.max(accuracys))
    bestThreshold = np.mean(thresholds[max_index])
    return bestThreshold

def evaluation_10_fold(feature_path='./result/cur_epoch_result.mat'):
    ACCs = np.zeros(10)
    result = scipy.io.loadmat(feature_path)
    for i in range(10):
        fold = result['fold']
        flags = result['flag']
        featureLs = result['fl']
        featureRs = result['fr']

        valFold = fold != i
        testFold = fold == i
        flags = np.squeeze(flags)

        mu = np.mean(np.concatenate((featureLs[valFold[0], :], featureRs[valFold[0], :]), 0), 0)
        mu = np.expand_dims(mu, 0)
        featureLs = featureLs - mu
        featureRs = featureRs - mu
        featureLs = featureLs / np.expand_dims(np.sqrt(np.sum(np.power(featureLs, 2), 1)), 1)
        featureRs = featureRs / np.expand_dims(np.sqrt(np.sum(np.power(featureRs, 2), 1)), 1)

        scores = np.sum(np.multiply(featureLs, featureRs), 1)
        threshold = getThreshold(scores[valFold[0]], flags[valFold[0]], 10000)
        ACCs[i] = getAccuracy(scores[testFold[0]], flags[testFold[0]], threshold)

    return ACCs


def load_model(data_root, file_list, backbone_net, gpus='0', resume=None):

    if backbone_net == 'MobileFace':
        net = mobilefacenet.MobileFaceNet()
    elif backbone_net == 'Res50':
        net = resnet.ResNet50()
    elif backbone_net == 'CBAM_50':
        net = cbam.CBAMResNet(50, feature_dim=args.feature_dim, mode='ir')
    elif backbone_net == 'CBAM_50_SE':
        net = cbam.CBAMResNet(50, feature_dim=args.feature_dim, mode='ir_se')
    elif backbone_net == 'CBAM_100':
        net = cbam.CBAMResNet(100, feature_dim=args.feature_dim, mode='ir')
    elif backbone_net == 'CBAM_100_SE':
        net = cbam.CBAMResNet(100, feature_dim=args.feature_dim, mode='ir_se')
    else:
        print(backbone_net, ' is not available!')

    # gpu init
    multi_gpus = False
    if len(gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net.load_state_dict(torch.load(resume)['net_state_dict'])

    if multi_gpus:
        net = DataParallel(net).to(device)
    else:
        net = net.to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    lfw_dataset = LFW(data_root, file_list, transform=transform)
    lfw_loader = torch.utils.data.DataLoader(lfw_dataset, batch_size=128,
                                             shuffle=False, num_workers=2, drop_last=False)

    return net.eval(), device, lfw_dataset, lfw_loader


def load_onnx_model(data_root, file_list):
    ort_session = ort.InferenceSession('checkpoints/resnet50_Quant.onnx')

    lfw_dataset = LFWDataset(data_root, file_list)

    return ort_session, lfw_dataset


def getFeatureFromTorch(feature_save_dir, net, device, data_set, data_loader):
    featureLs = None
    featureRs = None
    count = 0
    for data in data_loader:
        for i, _ in enumerate(data):
            data[i] = data[i].to(device)
        count += data[0].size(0)
        #print('extracing deep features from the face pair {}...'.format(count))
        with torch.no_grad():
            res = [net(d).data.cpu().numpy() for d in data]
        featureL = np.concatenate((res[0], res[1]), 1)
        featureR = np.concatenate((res[2], res[3]), 1)
        # print(featureL.shape, featureR.shape)
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)
        # print(featureLs.shape, featureRs.shape)

    result = {'fl': featureLs, 'fr': featureRs, 'fold': data_set.folds, 'flag': data_set.flags}
    scipy.io.savemat(feature_save_dir, result)


def getFeatureFromOnnx(feature_save_dir, net, data_set):
    featureLs = None
    featureRs = None
    count = 0

    for data in data_set:
        res = []
        for _, i in enumerate(data):
            feat = net.run(None, {"input": data[i]})
            res.append(feat)
        count += data[0].size(0)

        featureL = np.concatenate((res[0], res[1]), 1)
        featureR = np.concatenate((res[2], res[3]), 1)
        # print(featureL.shape, featureR.shape)
        if featureLs is None:
            featureLs = featureL
        else:
            featureLs = np.concatenate((featureLs, featureL), 0)
        if featureRs is None:
            featureRs = featureR
        else:
            featureRs = np.concatenate((featureRs, featureR), 0)
        # print(featureLs.shape, featureRs.shape)

    result = {'fl': featureLs, 'fr': featureRs, 'fold': data_set.folds, 'flag': data_set.flags}
    scipy.io.savemat(feature_save_dir, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--root', type=str, default='/datasets/public1/upload/datasets/lfw', help='The path of lfw data')
    parser.add_argument('--file_list', type=str, default='/datasets/public1/upload/datasets/lfw_pair.txt', help='The path of lfw data')
    parser.add_argument('--backbone_net', type=str, default='Res50', help='MobileFace, Res50, CBAM_50, CBAM_50_SE, CBAM_100, CBAM_100_SE')
    parser.add_argument('--feature_dim', type=int, default=512, help='feature dimension')
    parser.add_argument('--resume', type=str, default='./checkpoints/Res50_RES50_20210711_091848/Iter_066000_net.ckpt',
                        help='The path pf save checkpoints')
    parser.add_argument('--feature_save_path', type=str, default='./result/cur_epoch_lfw_result.mat',
                        help='The path of the extract features save, must be .mat file')
    parser.add_argument('--gpus', type=str, default='0', help='gpu list')
    args = parser.parse_args()

    # inference by torch
    # net, device, lfw_dataset, lfw_loader = load_model(args.root, args.file_list, args.backbone_net, args.gpus, args.resume)
    # getFeatureFromTorch(args.feature_save_path, net, device, lfw_dataset, lfw_loader)
    # ACCs = evaluation_10_fold(args.feature_save_path)

    # inference by onnx
    net, lfw_dataset = load_onnx_model(args.root, args.file_list)
    getFeatureFromOnnx(args.feature_save_path, net, lfw_dataset)
    ACCs = evaluation_10_fold(args.feature_save_path)

    for _, i in enumerate(ACCs):
        print(f'{i + 1}    {ACCs[i] * 100:.2f}')
    print('--------')
    print(f'AVE    {np.mean(ACCs) * 100:.4f}')
