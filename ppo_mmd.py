import torch
import numpy as np
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from process_traj import distance_to_buffer
from valbuf import ValBuffer


################################## set device ##################################

print("===============================================================================")


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
print("===============================================================================")




################################## PPO Policy ##################################


def exp_kernel(x, y, h=1.0, vec=True):
    if x.ndim == 1:
        x = x.reshape((1, -1))
    if y.ndim == 1:
        y = y.reshape((1, -1))
    print("np.sum((x - y)**2, 1): ", np.sum((x - y)**2, 1))
    return np.exp(-1.0*np.sum((x - y)**2, 1) / h**2 / 600.0)


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        has_continuous_action_space, 
        action_std_init, 
        ):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh())
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1))
        
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1))
        
    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError
    

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO_mmd:
    def __init__(
        self, 
        use_mmd,
        known_trajs,
        state_dim, 
        action_dim, 
        lr_actor, 
        lr_critic, 
        gamma, 
        K_epochs, 
        eps_clip, 
        has_continuous_action_space, 
        action_std_init=0.65,
        batch_size=256,
        mmd_alpha=0.42,
        max_num=500,
        ):
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.use_mmd = use_mmd
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.batch_size = batch_size
        self.buffer = RolloutBuffer()
        self.terminated = False
        self.mmd_alpha = mmd_alpha
        
        # Store mmd distances
        self.valbuf = ValBuffer(max_num)
        
        if self.use_mmd: 
            self.lr_actor = lr_actor / 2
        else:
            self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        self.policies = []
        self.trajs = []
        self.known_trajectories = known_trajs
        
    def clear_trajs(self):
        del self.trajs[:]   
    
    def append_traj(self, traj):
        self.trajs.append(traj)
    
    def return_trajs(self):
        return self.trajs

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        
        else:
            print("-------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("-------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("-----------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("-----------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()
    
    def distance_to_demos(self, path, key='observations', h=3.):
        expert_features = []
        for expert_traj in self.known_trajectories:
            expert_features.append(expert_traj[key]) 
        
        current_trajectory = path[key]
        mmd_distance, expert_traj, idx = distance_to_buffer(current_trajectory, expert_features, h=h)
        return mmd_distance, idx

    def update(self, episode):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        
        ################ calculate the MMD distance ################
        mmd_distances = []
        if self.use_mmd:
            meet_same_goal = False

            for idx, path in enumerate(self.trajs):
                mmd_distance, expert_id = self.distance_to_demos(path, key='observations', h=9.)
                
                mmd_distances.append(mmd_distance)
                print("!!!!!!!!mmd_distance: ", mmd_distance)
                self.valbuf.add_values([mmd_distance])
                # Distance normalization method
                # mmd_distance = self.valbuf.normal_value(mmd_distance)
                
                # print("!!!!!!!!normalized mmd_distance: ", mmd_distance)
                
                if idx == 0:
                    discrepancy = mmd_distance * np.ones((len(path["observations"])))
                else:
                    mmd_path = mmd_distance * np.ones((len(path["observations"])))
                    discrepancy = np.concatenate([discrepancy, mmd_path], axis=0)

                for traj in self.known_trajectories:
                    if traj["goal"] == path["goal"]:
                        meet_same_goal = True
                        
            # DIPG settings            
            discrepancy = torch.tensor(discrepancy, dtype=torch.float32).to(device)
            discrepancy = (discrepancy - discrepancy.mean()) / (discrepancy.std() + 1e-7)
            
            # Constraint field.
            discrepancy[discrepancy > 0.6] = 0.6
            
            # TCPPO gradient
            discounted_discrepancies = []
            discounted_discrepancy = 0.
            for reward, is_terminal in zip(reversed(discrepancy), reversed(self.buffer.is_terminals)):
                if is_terminal:
                    discounted_discrepancy = 0.
                discounted_discrepancy = reward + (self.gamma * discounted_discrepancy)
                discounted_discrepancies.insert(0, discounted_discrepancy)
                
            discrepancy = torch.tensor(discounted_discrepancies, dtype=torch.float32).to(device)
            discrepancy = (discrepancy - discrepancy.mean()) / (discrepancy.std() + 1e-7)
            
            # Adaptive sigma scaling method
            if meet_same_goal:
                self.mmd_alpha = self.mmd_alpha * 1.5
            elif not meet_same_goal and self.mmd_alpha > 0.20:
                self.mmd_alpha = self.mmd_alpha * 0.96
            elif not meet_same_goal and self.mmd_alpha > 0.15 and episode > 240:
                self.mmd_alpha = self.mmd_alpha * 0.99
            print("!!!!!!!!!!!mmd_alpha: ", self.mmd_alpha)
        
        # Optimize policy for K epochs
        for k in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
                
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()  
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                
            ################ calculate the MMD loss ################
            if self.use_mmd:
                logprobs_concat = torch.tensor([])
                    
                for path in self.trajs:
                    # Evaluating old aligned actions and values
                    mmd_states = torch.squeeze(torch.tensor(path["observations"], dtype=torch.float32))
                    mmd_actions = torch.squeeze(torch.tensor(path["actions"], dtype=torch.float32))
                    mmd_log_probs, mmd_state_values, mmd_dist_entropy = self.policy.evaluate(
                        mmd_states, mmd_actions) 
                    logprobs_concat = torch.cat((logprobs_concat, mmd_log_probs), axis=0)
                
                # TCPPO mmd loss computing
                mmd_ratios = torch.exp(logprobs_concat - old_logprobs.detach())
                mmd_surr1 = mmd_ratios * discrepancy 
                mmd_surr2 = torch.clamp(mmd_ratios, 1-self.eps_clip, 1+self.eps_clip) * discrepancy
                
                # mmd_loss = discrepancy * logprobs_concat
                mmd_loss = torch.min(mmd_surr1, mmd_surr2)
            
            # the original PPO objective
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # final loss of clipped objective PPO
            if not self.use_mmd:
                loss = loss.mean()
            elif self.use_mmd:
                loss = loss.mean() - self.mmd_alpha * mmd_loss.mean()
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
        
        return loss, mmd_distances
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
