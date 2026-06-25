import gymnasium as gym
import torch
import torch.nn as nn

class actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(3,16)
        self.layer1=nn.Linear(16,32)
        self.layer2=nn.Linear(32,32)
        self.layer3=nn.Linear(32,16)
        self.output=nn.Linear(16,2)
    def forward(self,x):
        x=nn.functional.relu(self.input(x))
        x=nn.functional.relu(self.layer1(x))
        x=nn.functional.relu(self.layer2(x))
        x=nn.functional.relu(self.layer3(x))
        x=self.output(x)
        std=x[...,0]
        mean = 2 * torch.tanh(x[..., 1])
        std = torch.clamp(std,-20,2)
        std=torch.exp(std)
        return std , mean

class critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.input=nn.Linear(3,16)
        self.layer1=nn.Linear(16,32)
        self.layer2=nn.Linear(32,32)
        self.layer3=nn.Linear(32,16)
        self.output=nn.Linear(16,1)
    def forward(self,x):
        x=nn.functional.relu(self.input(x))
        x=nn.functional.relu(self.layer1(x))
        x=nn.functional.relu(self.layer2(x))
        x=nn.functional.relu(self.layer3(x))
        x=self.output(x)
        return x

env = gym.make('Pendulum-v1', render_mode='human')
act=actor()
cri=critic()
optimizer1=torch.optim.Adam(cri.parameters(),lr=3e-4)
optimizer2=torch.optim.Adam(act.parameters(),lr=3e-4)
loss=nn.MSELoss()
gamma=0.99
for episode in range (10000):
    obs,_=env.reset()
    steps=0
    truncated=False
    next_obss=[]
    rewords=[]
    dones=[]
    log_probs=[]
    returnes=[]
    current_obss=[]
    actions=[]
    done=False
    while done==False and steps <500:
        steps+=1
        std,mean=act(torch.FloatTensor(obs))
        dist=torch.distributions.Normal(mean,std)
        action=dist.sample().detach()
        log_prob=dist.log_prob(action)
        next_obs,reword,terminated,truncated,_=env.step(action.detach().numpy().reshape(1))
        done = terminated or truncated
        next_obss.append(next_obs)
        rewords.append(reword)
        dones.append(done)
        log_probs.append(log_prob)
        current_obss.append(obs)
        actions.append(action)
        obs=next_obs
    G=0
    for r,d in zip(reversed(rewords),reversed(dones)):
        G=r+gamma*G*(1-int(d))
        returnes.insert(0,G)
    next_obss_B=torch.FloatTensor(next_obss)
    rewords_B=torch.FloatTensor(rewords)
    dones_B=torch.LongTensor(dones)
    log_probs_B=torch.stack(log_probs)
    current_obss_B=torch.FloatTensor(current_obss)
    action_B=torch.stack(actions)
    returnes_B=torch.FloatTensor(returnes)
    with torch.no_grad():
        values=cri(current_obss_B).view(-1)
        adv = returnes_B - values   
        adv=(adv-adv.mean())/(adv.std()+1e-8)
    
    for k in range(5):
        new_std, new_mean = act(current_obss_B)
        dist_new = torch.distributions.Normal(new_mean, new_std)
        new_log_probs = dist_new.log_prob(action_B)
        ratio = torch.exp(new_log_probs - log_probs_B.detach())
        clip = torch.clamp(ratio, 0.8, 1.2)
        losss = loss(cri(current_obss_B).view(-1), returnes_B)
        losss2 = -torch.min(ratio * adv, clip * adv).mean() - 0.001 * dist_new.entropy().mean()
        optimizer1.zero_grad()
        losss.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        losss2.backward()
        optimizer2.step()
    print(f"episode = {episode} || steps = {steps} || std = {std.item()} || total_reward = {sum(rewords):.2f}")
    torch.save(act.state_dict(), 'actMUJOCO.pth')
    torch.save(cri.state_dict(), 'criMUJOCO.pth')