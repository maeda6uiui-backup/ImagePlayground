"""
参考記事: https://qiita.com/fukuit/items/215ef75113d97560e599
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import logging
import os
from torch.utils.data import DataLoader

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging_fmt="%(asctime)s %(levelname)s: %(message)s"
logging.basicConfig(format=logging_fmt)
logger=logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1=nn.Conv2d(1,32,3)
        self.conv2=nn.Conv2d(32,64,3)
        self.pool=nn.MaxPool2d(2,2)
        self.dropout1=nn.Dropout2d()
        self.fc1=nn.Linear(12*12*64,128)
        self.dropout2=nn.Dropout2d()
        self.fc2=nn.Linear(128,10)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(F.relu(self.conv2(x)))
        x=self.dropout1(x)
        x=x.view(-1,12*12*64)
        x=F.relu(self.fc1(x))
        x=self.dropout2(x)
        x=self.fc2(x)

        return x

def create_dataloaders():
    transform=transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))])

    train_dataset=torchvision.datasets.MNIST(
        root="./Data",
        train=True,
        download=True,
        transform=transform)
    train_dataloader=DataLoader(train_dataset,batch_size=100,shuffle=True)

    test_dataset=torchvision.datasets.MNIST(
        root="./Data",
        train=False,
        download=True,
        transform=transform)
    test_dataloader=DataLoader(test_dataset,batch_size=100,shuffle=False)

    dataloaders={
        "train":train_dataloader,
        "test":test_dataloader
    }
    return dataloaders

def train(model:MNISTModel,train_dataloader:DataLoader,num_epochs:int,results_save_dir:str):
    os.makedirs(results_save_dir,exist_ok=True)

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=0.0005)

    model.train()

    for epoch in range(num_epochs):
        logger.info("===== Epoch {}/{} =====".format(epoch,num_epochs-1))

        for i,(inputs,labels) in enumerate(train_dataloader):
            inputs=inputs.to(device)
            labels=labels.to(device)

            optimizer.zero_grad()

            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            if i%100==0:
                logger.info("Step: {} Loss: {}".format(i,loss.item()))

        parameters=model.state_dict()
        checkpoint_save_filepath=os.path.join(results_save_dir,"checkpoint_{}.pt".format(epoch))
        torch.save(parameters,checkpoint_save_filepath)

def test(model:MNISTModel,test_dataloader:DataLoader,results_save_dir:str):
    count_correct=0
    count_total=0

    model.eval()

    with torch.no_grad():
        for(images,labels) in test_dataloader:
            images=images.to(device)
            labels=labels.to(device)

            outputs=model(images)
            _,predicted=torch.max(outputs,1)
            
            count_total+=labels.size(0)
            count_correct+=(predicted==labels).sum().item()

    result_msg="Accuracy: {}".format(count_correct/count_total)
    logger.info(result_msg)

    result_output_filepath=os.path.join(results_save_dir,"test_result.txt")
    with open(result_output_filepath,"w") as w:
        w.write(result_msg)

def main():
    logger.info("Create dataloaders")
    dataloaders=create_dataloaders()

    logger.info("Create a model")
    model=MNISTModel()
    model.to(device)

    logger.info("Start training")
    train(model,dataloaders["train"],20,"./Result")

    logger.info("Start test")
    test(model,dataloaders["test"],"./Result")

if __name__=="__main__":
    main()
