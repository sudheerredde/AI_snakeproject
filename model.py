import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size) #this is for the input 
        self.linear2 = nn.Linear(hidden_size, output_size) #this is for the hiden layer

    #forward function 
    def forward(self, x):
        #here we apply activation function and apply linear layer that directly comes from torch.nn.functional module
        x = F.relu(self.linear1(x)) # we apply relu, activation function for tensor x as an input 
        #and apply the second layer linear2(x)
        x = self.linear2(x)
        return x

    #this is the helper function to save the model for the future
    def save(self, file_name='model.pth'):
        #creates a new folder in the current dir
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


#from we do the traning and optimization
class QTrainer:
    #here it gets the model and learning rate and gamma parameter
    def __init__(self, model, lr, gamma):
        #then this is the storing part.
        self.lr = lr
        self.gamma = gamma
        self.model = model
        #this is the pytorch optimization
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        #here we use the loss function
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        #we do this for state, next state, action and reward
        #we do the torch.tensor and specify the variable and datatype
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)
        #tensor for the multiple values 
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            #this is a tuple with one value

        # 1: predicted Q values with current state
        predic = self.model(state)

        target = predic.clone()
        for temp in range(len(done)):
            Q_new = reward[temp]
            if not done[temp]:
                Q_new = reward[temp] + self.gamma * torch.max(self.model(next_state[temp]))

            target[temp][torch.argmax(action[temp]).item()] = Q_new
    
        self.optimizer.zero_grad() #this from pytorch to zero the gradient
        loss = self.criterion(target, predic) #here the target is 'Q new' and predic is the 'Q'
        loss.backward() #this is for back propagation and apply the gradient

        self.optimizer.step()

        
