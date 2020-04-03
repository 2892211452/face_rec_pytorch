import torch as torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt #作图用到的库
import time
from torch.autograd import Variable
from load_data  import *


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(          #(3,64,64)
                in_channels=3,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2   #padding=(kernelsize-stride)/2
            ),#(16,64,64)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)#(16,32,32)
 
        )
        self.conv2=nn.Sequential(#(16,32,32)
            nn.Conv2d(16,32,5,1,2),#(32,32,32)
            nn.ReLU(),#(32,32,32)
            nn.MaxPool2d(2)#(32,16,16)
        )

        #分类器
        self.classifier = nn.Sequential(
            nn.Linear(32*16*16,1000),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(1000,100),
            nn.ReLU(True),
            nn.Dropout(),

            nn.Linear(100,num_classes)
        )

    #定义前向传播过程，过程名字不可更改，因为这是重写父类的方法
    def forward(self,x):
        x = self.conv1( x )
        x = self.conv2( x ) #(batch,32,16,16)
        x=x.view(x.size(0),-1) #(batch,32*7*7)
        output=self.classifier(x)
        return output

def pre(model, img):
        image = resize_image(img, IMAGE_SIZE, IMAGE_SIZE)
        image = np.array(image)
        image = image/256
        image= torch.tensor(image).view(-1, 3,64,64).float()
        ans = model(image)
        index = ans.argmax()
        
        index = index.numpy()
        ans = ans.view(ans.size(1))
        pro = ans.detach().numpy()[index]
        #print(ans)

        return index, pro




if  __name__ == "__main__":
    vgg = CNN(2)
    print(vgg)

    fig = plt.figure(1, figsize=(5,5))
    ax=fig.add_subplot()
    #plt.ion()

    loss_F = nn.MSELoss()
    optimizer = torch.optim.Adam(vgg.parameters(), lr=0.0003)



    images = []
    labels = [] #之所以这么写是因为python变量的生命周期的问题
    images , labels =  load_dataset("/home/lwl/code/python/opencv/face_rec/data", images, labels)
    images = images /256
    images = torch.tensor(images).float()
    labels = torch.tensor(labels.astype(float))

    images = images.view(images.size(0), 3, 64, 64)
    train = False
    if train:    
        images= Variable(images)
        labels= Variable(labels).float()

        print(images)
        print(labels)
        for step in range(100):

            pre = vgg(images.float())

            loss  = loss_F(pre, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 5 ==0 :
                print("the step " +str(step) +" the loss is"+ str(loss))
            
            vgg.eval()

        torch.save(vgg, './model.h5') 
    else:
        model = torch.load('./model.h5')
        y1= model(images[5].view(-1, 3,64,64))
        print(y1)
        print(labels[5])
        image = cv2.imread('/home/lwl/code/python/opencv/face_rec/data/other/50.jpg')

        print(pre(model,image))

        # y2 = model(images[110].view(-1,3,64,64))
        # print(y2)
        # print(labels[110])
        # print(pre(model, images[110].view(-1, 3,64,64)))
