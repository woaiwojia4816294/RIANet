import torch
import time
from singleBatch_model import AttenNet
from ptflops import get_model_complexity_info


H = 720
W = 1280
netG = AttenNet()
macs, params = get_model_complexity_info(netG, (3, H, W), as_strings=True,
                                         print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8}'.format('Computational complexity of model for (720,1280) image: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters for (720,1280) image: ', params))

tensor = torch.randn(1, 3, H, W)
with torch.no_grad():
    netG.eval()
    print("Beginning Warmup...")
    netG(tensor)

    beg = time.time()
    for i in range(5):
        netG(tensor)
    print('Time taken by our model on CPU for (720,1280) image : {} seconds'.format((time.time() - beg) / 5))
