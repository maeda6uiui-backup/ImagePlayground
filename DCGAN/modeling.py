"""
参考記事: https://qiita.com/oki_uta_aiota/items/3154616e36009122efe1
"""

import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from typing import Dict

import sys
sys.path.append(".")

from models import Generator,Discriminator

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

class DCGANDataset(Dataset):
    def __init__(
        self,
        image_dir:str,
        transform=transforms.ToTensor(),
        latent_dim:int=100):
        self.image_dir=image_dir
        self.image_filepaths=[os.path.join(self.image_dir,name) for name in os.listdir(self.image_dir)]
        self.data_length=len(self.image_filepaths)

        self.transform=transform
        self.latent_dim=latent_dim

    def __len__(self):
        return self.data_length

    def __getitem__(self,index:int):
        latent=torch.randn(self.latent_dim,1,1)

        image_filepath=self.image_filepaths[index]
        image=Image.open(image_filepath)
        if self.transform is not None:
            image=self.transform(image)

        ret={
            "latent":latent,
            "image":image
        }
        return ret

def train(
    gen_model:Generator,
    dis_model:Discriminator,
    gen_optimizer:optim.Optimizer,
    dis_optimizer:optim.Optimizer,
    train_dataloader:DataLoader,
    logging_steps:int)->Dict[float,float]:
    gen_model.train()
    dis_model.train()

    criterion=nn.BCEWithLogitsLoss()

    count_steps=0
    total_gen_loss=0
    total_dis_loss=0
    total_dis_fake_loss=0
    total_dis_real_loss=0

    for step,batch in enumerate(train_dataloader):
        latents=batch["latent"]
        real_images=batch["image"]

        latents=latents.to(device)
        real_images=real_images.to(device)
        batch_len=len(real_images)

        ones=torch.ones(batch_len,device=device)
        zeros=torch.zeros(batch_len,device=device)

        fake_images=gen_model(latents)
        pred_fake=dis_model(fake_images)

        gen_loss=criterion(pred_fake,ones)

        gen_model.zero_grad()
        dis_model.zero_grad()
        gen_loss.backward()
        gen_optimizer.step()

        pred_real=dis_model(real_images)
        dis_real_loss=criterion(pred_real,ones)

        fake_images=gen_model(latents)
        pred_fake=dis_model(fake_images)
        dis_fake_loss=criterion(pred_fake,zeros)
        
        dis_loss=dis_real_loss+dis_fake_loss

        gen_model.zero_grad()
        dis_model.zero_grad()
        dis_loss.backward()
        dis_optimizer.step()

        count_steps+=1
        total_gen_loss+=gen_loss.item()
        total_dis_loss+=dis_loss.item()
        total_dis_fake_loss+=dis_fake_loss.item()
        total_dis_real_loss+=dis_real_loss.item()

        if step%logging_steps==0:
            logger.info("[Step {}]".format(step))
            logger.info("Generator Loss: {}".format(gen_loss.item()))
            logger.info("Discriminator Loss: Total: {} Fake: {} Real: {}".format(
                dis_loss.item(),dis_fake_loss.item(),dis_real_loss.item()))

    ret={
        "gen_loss":total_gen_loss/count_steps,
        "dis_loss":total_dis_loss/count_steps,
        "dis_fake_loss":total_dis_fake_loss/count_steps,
        "dis_real_loss":total_dis_real_loss/count_steps
    }
    return ret

def test(
    gen_model:Generator,
    dis_model:Discriminator,
    test_dataloader:DataLoader,
    latent_dim:int,
    epoch:int,
    results_save_dir:str):
    gen_model.eval()
    dis_model.eval()

    criterion=nn.BCEWithLogitsLoss()

    gen_losses=[]
    dis_losses=[]
    dis_fake_losses=[]
    dis_real_losses=[]

    for batch in test_dataloader:
        latents=batch["latent"]
        real_images=batch["image"]

        latents=latents.to(device)
        real_images=real_images.to(device)
        batch_len=len(real_images)

        ones=torch.ones(batch_len,device=device)
        zeros=torch.zeros(batch_len,device=device)

        pred_images=gen_model(latents)
        pred_fake=dis_model(pred_images)
        gen_loss=criterion(pred_fake,ones)

        gen_losses.append(gen_loss.item())

        pred_fake=dis_model(pred_images)
        pred_real=dis_model(real_images)
        dis_fake_loss=criterion(pred_fake,zeros)
        dis_real_loss=criterion(pred_real,ones)
        dis_loss=dis_real_loss+dis_fake_loss

        dis_fake_losses.append(dis_fake_loss.item())
        dis_real_losses.append(dis_real_loss.item())
        dis_losses.append(dis_loss.item())

    mean_gen_loss=sum(gen_losses)/len(gen_losses)
    mean_dis_loss=sum(dis_losses)/len(dis_losses)
    mean_dis_fake_loss=sum(dis_fake_losses)/len(dis_fake_losses)
    mean_dis_real_loss=sum(dis_real_losses)/len(dis_real_losses)

    test_latents=torch.randn(9,latent_dim,1,1,device=device)
    pred_images=gen_model(test_latents)

    test_image_save_filepath=os.path.join(results_save_dir,"test_{}.jpg".format(epoch))
    save_image(pred_images,test_image_save_filepath,nrow=3)

    ret={
        "gen_loss":mean_gen_loss,
        "dis_loss":mean_dis_loss,
        "dis_fake_loss":mean_dis_fake_loss,
        "dis_real_loss":mean_dis_real_loss
    }
    return ret

def main(args):
    data_dir:str=args.data_dir
    train_batch_size:int=args.train_batch_size
    test_batch_size:int=args.test_batch_size
    latent_dim:int=args.latent_dim
    gen_learning_rate:float=args.gen_learning_rate
    dis_learning_rate:float=args.dis_learning_rate
    num_epochs:int=args.num_epochs
    resume_epoch:int=args.resume_epoch
    results_save_dir:str=args.results_save_dir
    logging_steps:int=args.logging_steps
    checkpoint_save_epochs:int=args.checkpoint_save_epochs
    test_epochs:int=args.test_epochs

    logger.info(args)

    os.makedirs(results_save_dir,exist_ok=True)

    logger.info("Create dataloaders")

    train_images_dir=os.path.join(data_dir,"Train")
    train_dataset=DCGANDataset(
        train_images_dir,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        latent_dim=latent_dim
    )

    test_images_dir=os.path.join(data_dir,"Test")
    test_dataset=DCGANDataset(test_images_dir,latent_dim=latent_dim)

    train_dataloader=DataLoader(train_dataset,batch_size=train_batch_size,shuffle=True)
    test_dataloader=DataLoader(test_dataset,batch_size=test_batch_size,shuffle=False)

    logger.info("Create models")

    gen_model=Generator(latent_dim=latent_dim)
    dis_model=Discriminator()

    if resume_epoch is not None:
        logger.info("Load model parameters from checkpoints")
        logger.info("Resume epoch: {}".format(resume_epoch))

        generator_checkpoint_filepath=os.path.join(results_save_dir,"gen_checkpoint_{}.pt".format(resume_epoch))
        discriminator_checkpoint_filepath=os.path.join(results_save_dir,"dis_checkpoint_{}.pt".format(resume_epoch))

        generator_state_dict=torch.load(generator_checkpoint_filepath,map_location=torch.device("cpu"))
        discriminator_state_dict=torch.load(discriminator_checkpoint_filepath,map_location=torch.device("cpu"))

        gen_model.load_state_dict(generator_state_dict)
        dis_model.load_state_dict(discriminator_state_dict)

    gen_model.to(device)
    dis_model.to(device)

    logger.info("Create optimizers")

    gen_optimizer=optim.Adam(gen_model.parameters(),lr=gen_learning_rate)
    dis_optimizer=optim.Adam(dis_model.parameters(),lr=dis_learning_rate)

    logger.info("Start training")

    start_epoch=resume_epoch+1 if resume_epoch is not None else 0

    for epoch in range(start_epoch,num_epochs):
        logger.info("===== Epoch {}/{} =====".format(epoch,num_epochs-1))

        mean_losses=train(
            gen_model,
            dis_model,
            gen_optimizer,
            dis_optimizer,
            train_dataloader,
            logging_steps
        )

        logger.info("Finished epoch {} training".format(epoch))
        logger.info("Mean Loss (Generator): {}".format(mean_losses["gen_loss"]))
        logger.info("Mean Loss (Discriminator): Total: {} Fake: {} Real: {}".format(
            mean_losses["dis_loss"],mean_losses["dis_fake_loss"],mean_losses["dis_real_loss"]))

        if epoch%checkpoint_save_epochs==0:
            gen_checkpoint_filepath=os.path.join(results_save_dir,"gen_checkpoint_{}.pt".format(epoch))
            dis_checkpoint_filepath=os.path.join(results_save_dir,"dis_checkpoint_{}.pt".format(epoch))

            torch.save(gen_model.state_dict(),gen_checkpoint_filepath)
            torch.save(dis_model.state_dict(),dis_checkpoint_filepath)

        if epoch%test_epochs==0:
            mean_losses=test(
                gen_model,
                dis_model,
                test_dataloader,
                latent_dim,
                epoch,
                results_save_dir)

            logger.info("Finished epoch {} test".format(epoch))
            logger.info("Mean Loss (Generator): {}".format(mean_losses["gen_loss"]))
            logger.info("Mean Loss (Discriminator): Total: {} Fake: {} Real: {}".format(
                mean_losses["dis_loss"],mean_losses["dis_fake_loss"],mean_losses["dis_real_loss"]))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_dir",type=str,default="./Data")
    parser.add_argument("--train_batch_size",type=int,default=100)
    parser.add_argument("--test_batch_size",type=int,default=100)
    parser.add_argument("--latent_dim",type=int,default=100)
    parser.add_argument("--gen_learning_rate",type=float,default=0.0002)
    parser.add_argument("--dis_learning_rate",type=float,default=0.0002)
    parser.add_argument("--num_epochs",type=int,default=500)
    parser.add_argument("--resume_epoch",type=int)
    parser.add_argument("--results_save_dir",type=str,default="./Result")
    parser.add_argument("--logging_steps",type=int,default=1)
    parser.add_argument("--checkpoint_save_epochs",type=int,default=10)
    parser.add_argument("--test_epochs",type=int,default=10)
    args=parser.parse_args()

    main(args)
