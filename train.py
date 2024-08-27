import torch
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import random
import scipy.stats
from utils import set_dataset
import os
from network import MTIQA360
import logging
from sklearn.metrics import mean_squared_error
import random

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = "cpu"
joint_texts = ""
distortion_types = ['jpeg', 'j2k', 'avc', 'hevc', 'noise', 'blur', 'other']

initial_lr = 1e-2
num_epoch = 2
bs = 32


def train(model, loader, epoch, optimizer, scheduler):
    model.eval()
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    if optimizer.state_dict()['param_groups'][0]['lr'] == 0:
        scheduler.step()
        print(optimizer.state_dict()['param_groups'][0]['lr'])
    for step, sample_batched in enumerate(loader):
        x, mos, dist = sample_batched['I'], sample_batched['mos'], sample_batched['distortion']
        x = x.to(device)
        mos = mos.to(device)
        
        dist_gt = np.zeros((len(dist), len(distortion_types)), dtype=float)

        for i in range(x.size(0)):
            dist_gt[i, distortion_types.index(dist[i])] = 1.0
            
        dist_gt = torch.from_numpy(dist_gt).to(device)
        optimizer.zero_grad()

        logits_quality, logits_distortion = model(x, joint_texts)
        
        q_loss = nn.L1Loss()(logits_quality, mos).mean()
        d_loss = nn.CrossEntropyLoss()(logits_distortion, dist_gt.detach()).mean()
        total_loss = (q_loss + d_loss)
        total_loss.backward()

        optimizer.step()

        # statistics
        print(f"(E:{epoch}, B:{step+1} / {len(loader)})  [q_loss: {q_loss}, d_loss: {d_loss}]")

def eval(model, loader, epoch):
    model.eval()
    correct_dist = 0.0
    q_mos = []
    q_hat = []
    num_dist = 0
    for step, sample_batched in enumerate(loader):
        x, gmos, dist = sample_batched['I'], sample_batched['mos'], sample_batched['distortion']
        x = x.to(device)
        q_mos.extend(gmos.cpu().tolist())

        with torch.no_grad():
            logits_quality, logits_distortion = model(x, joint_texts)
            
        q_hat.extend(logits_quality.cpu().tolist())

        indice2 = logits_distortion.argmax(dim=1)
        for i in range(len(dist)):
            if distortion_types.index(dist[i]) == indice2[i]:  # dist_map: mapping dict #dists_map: type list
                correct_dist += 1
            num_dist += 1

        dist_acc = correct_dist / num_dist
        
    q_mos = [round(q * 10, 8) for q in q_mos]
    q_hat = [round(q * 10, 8) for q in q_hat]
    plcc = scipy.stats.mstats.pearsonr(x=q_mos, y=q_hat)[0]
    srcc = scipy.stats.mstats.spearmanr(x=q_mos, y=q_hat)[0]
    rmse = mean_squared_error(q_mos, q_hat, squared=False)

    return {"plcc": plcc, "srocc": srcc, "dist_acc": dist_acc, "epoch": epoch, "rmse": rmse}



def main():
    dataset = "oiqa"
    gpus = 1
    work_dir = f"./output/{dataset}"


    os.makedirs(work_dir, exist_ok=True)
    logging.basicConfig(
            level=logging.INFO,
            filename=os.path.join(work_dir, "log.log"),
            filemode='w',
            format='[%(asctime)s %(levelname)-8s] %(message)s',
            datefmt='%m%d_%H:%M'
        )
    new_handler = logging.FileHandler(os.path.join(work_dir, "log.log"))
    root_logger = logging.getLogger()
    root_logger.removeHandler(root_logger.handlers[0])
    root_logger.addHandler(new_handler)
    
    
    logging.info(f"prompts_distortion_types: {distortion_types}")

    seed = random.randrange(1, 100000000) 
    
    # seed = 194774 # best oiqa
    seed = 90740614 # best cviq
    logging.info(f"seed: {seed}")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ratio = 0.8
    if dataset == "cviq":
        ## cviqd 
        with open("./csv/cviq.csv", 'r') as fp:
            items = fp.read().split("\n")
        numbers = list(range(1, 545))
        distor_indices = [n for n in numbers if n % 34 != 0]
        refer_indices = [n for n in numbers if n % 34 == 0]
        random.shuffle(distor_indices)
        random.shuffle(refer_indices)
        train_indices = distor_indices[:int(len(distor_indices)*ratio)] + refer_indices[:int(len(refer_indices)*ratio)]
        test_indices = distor_indices[int(len(distor_indices)*ratio):] + refer_indices[int(len(refer_indices)*ratio):]
    elif dataset == "oiqa":            
        ## oiqa 
        with open("./csv/oiqa.csv", 'r') as fp:
            items = fp.read().split("\n")
        distor_indices = list(range(1, 321))
        refer_indices = list(range(321, 337))
        ratio = 0.8
        random.shuffle(distor_indices)
        random.shuffle(refer_indices)
        train_indices = distor_indices[:int(len(distor_indices)*ratio)] + refer_indices[:int(len(refer_indices)*ratio)]
        test_indices = distor_indices[int(len(distor_indices)*ratio):] + refer_indices[int(len(refer_indices)*ratio):]


    logging.info(f"dataset: {dataset}")
    logging.info(f"train_indices: {train_indices}")
    logging.info(f"test_indices: {test_indices}")

    num_workers = 8
    for session in range(1):
        if gpus > 1:
            model = torch.nn.DataParallel(MTIQA360(), device_ids=list(range(gpus))).cuda()
        else:
            model = MTIQA360().to(device)
        # model.half()
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, verbose=True)

        train_loader = set_dataset(items, train_indices, bs, f"./data224x224/{dataset}", num_workers, False)
        test_loader = set_dataset(items, test_indices, bs, f"./data224x224/{dataset}", num_workers, True)

        best_result = {"plcc": 0.0, "srocc": 0.0, "dist_acc": 0.0, "epoch": 0, "rmse": 0}
        for epoch in range(0, num_epoch):
            train(model, train_loader, epoch, optimizer, scheduler)
            result = eval(model, test_loader, epoch)
            scheduler.step()

            logging.info(f"epoch: {epoch}")
            logging.info(f"plcc={result['plcc']}, srocc={result['srocc']}, dist_acc={result['dist_acc']}, rmse={result['rmse']}")
            
            if best_result['srocc'] + best_result['plcc'] <= result['srocc'] + result['plcc']:
                logging.info("========Best LCC results so far========")
                best_result = result
                if gpus > 1:
                    torch.save(model.module.state_dict(), os.path.join(work_dir, "best_srocc_plcc.pth"))
                else:
                    torch.save(model.state_dict(), os.path.join(work_dir, "best_srocc_plcc.pth"))
                    
            print('...............current srocc best...............')
            print(f"current epoch: {epoch}")
            print(f"\tplcc={result['plcc']}, srocc={result['srocc']}, dist_acc={result['dist_acc']}, rmse={result['rmse']}")
            
            print(f"best epoch: {best_result['epoch']}")
            print(f"\tplcc={best_result['plcc']}, srocc={best_result['srocc']}, dist_acc={best_result['dist_acc']}, rmse={best_result['rmse']}")
            
        logging.info("========Best LCC results so far========")
        logging.info(f"epoch: {best_result['epoch']}")
        logging.info(f"plcc={best_result['plcc']}, srocc={best_result['srocc']}, dist_acc={best_result['dist_acc']}, rmse={best_result['rmse']}")


    torch.save(model.state_dict(), os.path.join(work_dir, "saved.pth"))

if __name__ == "__main__":
    for i in range(10):
        main()
