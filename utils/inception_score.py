import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import numpy as np
from scipy.stats import entropy
import torchvision.datasets as dset
import torchvision.transforms as transforms

# we should use same mean and std for inception v3 model in training and testing process
# reference web page: https://pytorch.org/hub/pytorch_vision_inception_v3/
mean_inception = [0.485, 0.456, 0.406]
std_inception = [0.229, 0.224, 0.225]


def inception_score(imgs, batch_size=64, resize=True, splits=10):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    batch_size -- batch size for feeding into Inception v3
    resize -- if image size is smaller than 229, then resize it to 229
    splits -- number of splits, if splits are different, the inception score could be changing even using same data
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dataloader
    # print('Creating data loader')
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=False).cuda()
    inception_model.eval()

    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions using pre-trained inception_v3 model
    # print('Computing predictions using inception v3 model')
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.cuda()
        inception_model.eval()
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batch)

    # Now compute the mean KL Divergence
    # print('Computing KL Divergence')
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :] # split the whole data into several parts
        py = np.mean(part, axis=0)  # marginal probability
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]  # conditional probability
            scores.append(entropy(pyx, py))  # compute divergence
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)




if __name__ == '__main__':

    #------------------- main function -------------------#
    # example of torch dataset, you can produce your own dataset
    cifar = dset.CIFAR10(root='../dataSet/cifar10', download=True,
                        transform=transforms.Compose([transforms.Resize(32),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean_inception, std_inception)
                                                    ])
                        )
    mean, std = inception_score(cifar, 5, splits=10)
    print('IS is %.4f' % mean)
    print('The std is %.4f' % std)