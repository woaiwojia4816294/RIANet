import torch
from singleBatch_model import AttenNet
from torchvision.utils import save_image
import argparse
import torch.nn as nn
from tqdm import tqdm
import os
from PIL import Image
from torchvision import transforms


parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', type=str, default=r'.\test_images\BDD10K_test')
parser.add_argument('--trained_model_path', type=str, default=r'.\checkpoints\netG.pth')
parser.add_argument('--resultsPath', type=str, default=r'.\test_results')
parser.add_argument('--device_ids', default=[0])


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
        img_transforms = transforms.Compose([
            transforms.Resize((240, 480)),
            transforms.ToTensor()])
        
        proc_img = img_transforms(img).unsqueeze(0).cuda()
        enhanced_img = netG(proc_img)

        save_image(enhanced_img, os.path.join(opt.resultsPath, os.path.split(imgPath)[-1].split('.')[-2] + '.jpg'),
                   normalize=True, value_range=(0, 1))
