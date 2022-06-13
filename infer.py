import torch
from utils.singleBatch_model import AttenNet
from torchvision.utils import save_image
import argparse
import torch.nn as nn
from datasets import *
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', type=str, default=r'D:\PycharmProjects\low-light enhancement\oa_code\test_images')
parser.add_argument('--trained_model_path', type=str, default=r'D:\PycharmProjects\low-light enhancement\oa_code\saved_models\netG_105.pth')
parser.add_argument('--resultsPath', type=str, default=r'./test_results')

parser.add_argument('--imgHeight', type=int, default=240, help='size of the data crop (squared assumed)')
parser.add_argument('--imgWidth', type=int, default=480, help='size of the data crop (squared assumed)')
parser.add_argument('--device_ids', default=[0])
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')

opt = parser.parse_args()
print(opt)

os.makedirs(opt.resultsPath, exist_ok=True)

netG = AttenNet().cuda()
netG = nn.DataParallel(netG, device_ids=opt.device_ids)


if __name__ == '__main__':
    netG.load_state_dict(torch.load(opt.trained_model_path))
    imgs = os.listdir(opt.dataPath)
    for i in tqdm(range(len(os.listdir(opt.dataPath)))):
        imgPath = os.path.join(opt.dataPath, imgs[i])
        img = Image.open(imgPath)
        img_transforms = transforms.Compose([transforms.Resize((240, 240)),
                                             transforms.ToTensor()])

        proc_img = img_transforms(img).unsqueeze(0).cuda()
        enhanced_img = netG(proc_img)

        save_image(enhanced_img, os.path.join(opt.resultsPath, os.path.split(imgPath)[-1].split('.')[-2] + '.PNG'),
                   normalize=True, value_range=(0, 1))
