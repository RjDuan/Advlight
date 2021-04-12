import numpy as np
import math
import os
import torch
from light_simulation import tube_light_generation_by_func, simple_add
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
import torchvision.transforms as transforms
import argparse
import random
import shutil
import itertools

from tqdm import tqdm

from PIL import Image
import torchvision.transforms.functional as transf


parser = argparse.ArgumentParser()

parser.add_argument('--data', type=str, default='./query_imagenet', help='location of the data corpus')
parser.add_argument('--portion_for_search', type=float, default=1.0, help='portion of training data')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--trial_nums', type=int, default=300, help='number of the trials')
parser.add_argument('--model', type=str, default='resnet50', help='number of the trials')
parser.add_argument('--output_csv', type=str, default='random_search.csv', help='number of the trials')

args = parser.parse_args()

# transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
test_transform =  transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

# dataset
# imagenet_dataset = Query_Data('/root/project/data/query_imagenet', transform=test_transform)
# imagenet_dataset = ImageFolder('/disk1/imagenet_val/val', transform=test_transform)
imagenet_path_list = os.listdir(args.data)
imagenet_dataset = []
for img_path in imagenet_path_list:
    imagenet_dataset.append((img_path, int(img_path.split('.')[0])))

total_num = len(imagenet_dataset)
current_num = 0

# valid_queue = torch.utils.data.DataLoader(
#     imagenet_dataset, batch_size=args.batch_size,
#     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:total_num]),
#     pin_memory=True, num_workers=16)

# model
if args.model == 'resnet50':
    print("Loading model...")
    model = resnet50(pretrained=True)
elif args.model == 'df_resnet50':
    print("Loading adv trained model...")
    model = resnet50(pretrained=False)
    model.load_state_dict(torch.load('/root/project/data/checkpoint-89.pth.tar')['state_dict'])


model.cuda()
model.eval()

# for cln_image, adv_image, target, index in search_queue:
#     print(target, index)
#     break
# for cln_image, adv_image, target, index in search_queue:
#     print(target, index)
#     break
# break
acc_adv = 0
acc_cln = 0
total_q = 0
delay_threhold = 20
Q = np.asarray([[1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [1,1,0,0],
        [1,0,1,0],
        [1,0,0,1],
        [0,1,1,0],
        [0,1,0,1],
        [0,0,1,1]
        ])

for image_path, target in tqdm(imagenet_dataset):
    current_num += 1
    img = Image.open(os.path.join('./query_imagenet', image_path).encode("utf-8")).convert('RGB')

    # clean
    clean_image = img.resize((256, 256), Image.BILINEAR)
    clean_image = test_transform(clean_image).unsqueeze(0)
    clean_image = clean_image.cuda()
    with torch.no_grad():
        org_pred_label = model(clean_image)
        org_pred_label = org_pred_label.cpu().detach()
    
    min_confidence = org_pred_label[0, target].item()
    org_pred_label = org_pred_label.max(1)[1].item()

    adv_image = np.asarray(img)

    cur_pred_label = org_pred_label

    correct_adv = org_pred_label == target
    correct_cln = cur_pred_label == target

    cur_search = 0
    # V = np.asarray([[580, 31, 74, 400], [580, 17, 131, 200], [580, 144, 316, 600]])
    # init_v = V[np.random.randint(len(V))]
    params_list = []
    for i in range(200):
        init_v_it = [np.random.randint(380, 750), np.random.randint(0,180), np.random.randint(0,400), np.random.randint(10, 1600)]
        params_list.append(init_v_it)
    # params_list = list(itertools.product(list(np.linspace(380, 750, num=4)),
    # list(np.linspace(0, 180, num=4)),
    # list(np.linspace(0, 400, num=4)),
    # list(np.linspace(10, 1600, num=4))))
    for init_v in params_list:

        for search_i in range(delay_threhold):
            q_id = np.random.randint(len(Q))
            q = Q[q_id]
            step_size = np.random.randint(1, 20)
            q = q*step_size
            for a in [-1, 1]:
                cur_search += 1
                #print(a*q)
                temp_q = init_v + a*q
                temp_q = np.clip(temp_q, [380, 0, 0, 10], [750, 180, 400, 1600])
                
                radians = math.radians(temp_q[1])
                k = round(math.tan(radians), 2)
                
                tube_light = tube_light_generation_by_func(k, temp_q[2], alpha = 1.0, beta=temp_q[3], wavelength=temp_q[0]) 
                tube_light =  tube_light * 255.0
                img_with_light = simple_add(adv_image, tube_light, 1.0)
                img_with_light = np.clip(img_with_light, 0.0, 255.0).astype('uint8')
                img_with_light = Image.fromarray(img_with_light)

                img_with_light = img_with_light.resize((224, 224), Image.BILINEAR)
                img_with_light = test_transform(img_with_light).unsqueeze(0)
                img_with_light = img_with_light.cuda()
                with torch.no_grad():
                    cur_pred_label = model(img_with_light)
                    cur_pred_label = cur_pred_label.cpu().detach()

                cur_confidence = cur_pred_label[0, target].item()
                cur_pred_label = cur_pred_label.max(1)[1].item()

                if cur_confidence < min_confidence:
                    min_confidence = cur_confidence
                    init_v = temp_q
                    break
            
            if cur_pred_label != org_pred_label:
                correct_adv = False
                break
            
        if cur_pred_label != org_pred_label:
            correct_adv = False
            break
    
    total_q += cur_search
    if correct_cln:
        acc_cln += 1

    if correct_adv:
        acc_adv += 1
        print('{} attack failed\tqueries: {}\tmean queries: {}\tclean acc: {}\tadv acc:{}'.format(image_path, cur_search, total_q/current_num, acc_cln/current_num, acc_adv/current_num))
    else:
        print('{} attack success\tqueries: {}\tmean queries: {}\tclean acc: {}\tadv acc:{}'.format(image_path, cur_search, total_q/current_num, acc_cln/current_num, acc_adv/current_num))




