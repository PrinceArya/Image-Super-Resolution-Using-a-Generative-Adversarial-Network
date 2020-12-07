from tqdm import tqdm_notebook as tqdm
from torch.utils.data import Dataset, DataLoader
from  Vgg_space import TruncatedVGG19
from generator import Generator
from discriminator import Discriminator
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = Data(Dir,x_train,transform=True,train=True)
valid_dataset = Data(Dir,x_val,transform=True,train=True)

batch_size=16
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
val_loader = DataLoader(dataset=valid_dataset,batch_size=batch_size,shuffle=True)

vgg19_i = 5 
vgg19_j = 4 
truncated_vgg19 = TruncatedVGG19(i=vgg19_i, j=vgg19_j) #(For understanding i and j refer to read me file)
truncated_vgg19.eval()

# Loss functions
vgg_loss = nn.MSELoss()

pth = "/content/drive/MyDrive/model_SRGAN.pth" #Path to weight file.
checkpoint = torch.load(pth)
gen = checkpoint['generator']
disc = checkpoint['discriminator']
gen_optimizer = checkpoint['optimizer_g']
disc_optimizer = checkpoint['optimizer_d']

gen_loss = nn.BCELoss()
mse_loss = nn.MSELoss()
disc_loss = nn.BCELoss()


#Note if you are training for the first time then uncomment the below line:
'''
gen = Generator()
disc = Discriminator()
truncated_vgg19 = truncated_vgg19

gen_optimizer = optim.Adam(gen.parameters(),lr=0.0001)
disc_optimizer = optim.Adam(disc.parameters(),lr=0.0001)
'''

gen = gen.to(device)
disc = disc.to(device)
truncated_vgg19 = truncated_vgg19.to(device)

def train_model(n_epochs):
  d1loss_list=[]
  d2loss_list=[]
  gloss_list=[]
  vloss_list=[]
  mloss_list=[]
  for epoch in tqdm(range(n_epochs)):
    for lr,hr in train_loader:
      lr = lr.to(device).float() #Low Resolution image
      hr = hr.to(device).float() #High Resolution image
      disc.zero_grad()
      gen_out = gen(lr)   #Output of Generator

      false_label = disc(gen_out) #Run Discriminator on Generated image
      true_label = disc(hr) #Run Discriminator on Hight-Resolution image
      d1_loss = (disc_loss(false_label,torch.zeros_like(false_label,dtype=torch.float)))
      d2_loss = (disc_loss(true_label,torch.ones_like(true_label,dtype=torch.float)))
      d2_loss.backward()
      d1_loss.backward(retain_graph=True)
      disc_optimizer.step()

      gen.zero_grad()

      g_loss = gen_loss(false_label.data,torch.ones_like(false_label,dtype=torch.float))#Generator loss
      sr_imgs_in_vgg_space = truncated_vgg19(gen_out)
      hr_imgs_in_vgg_space = truncated_vgg19(hr).detach()

      v_loss = vgg_loss(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space) #Find the loss between o/p of vgg network for generated image and high resolution image
      m_loss = mse_loss(gen_out,hr) #Pixel wise mean square error.
      generator_loss = g_loss + v_loss + m_loss #Net generator loss is sum of all above three
      generator_loss.backward()
      gen_optimizer.step()

      d1loss_list.append(d1_loss.item())
      d2loss_list.append(d2_loss.item())
        
      gloss_list.append(g_loss.item())
      vloss_list.append(v_loss.item())
      mloss_list.append(m_loss.item())

    #Now print the metric
    print("Epoch ::::  "+str(epoch+1)+"  d1_loss ::: "+str(np.mean(d1loss_list))+"  d2_loss :::"+str(np.mean(d2loss_list)))
    print("genLoss ::: "+str(np.mean(gloss_list))+"  vggLoss ::: "+str(np.mean(vloss_list))+"  MeanLoss  ::: "+str(np.mean(mloss_list)))
    
    #Save the model at path "pth"
    torch.save({'epoch': epoch,
               'generator': gen,
               'discriminator': disc,
               'optimizer_g': gen_optimizer,
               'optimizer_d': disc_optimizer},
                pth)
                
train_model(1000) #Train for 1000 epochs.



