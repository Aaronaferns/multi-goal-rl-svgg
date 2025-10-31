import numpy as np
# from Networks import PolicyNetwork, QValueNetwork
from torch import from_numpy
from torch.optim.adam import Adam
from torch.nn import functional as F
import torch as th
from algorithms.goalBasedRL.goal_networks import Policy as PolicyNetwork, QValue as QValueNetwork


class SAC:
    def __init__(self, nS,nA, lr, gamma, batch_size,fixed_network_update_freq=100, self.debug = False):
        self.state_shape = nS
        self.n_actions = nA
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.fixed_network_update_freq=fixed_network_update_freq

        self.device = "cuda" if th.cuda.is_available() else "cpu"

        self.policy_network = PolicyNetwork(self.state_shape, self.n_actions).to(self.device)
        self.q_value_network1 = QValueNetwork(self.state_shape, self.n_actions).to(self.device)
        self.q_value_network2 = QValueNetwork(self.state_shape, self.n_actions).to(self.device)
        self.q_value_target_network1 = QValueNetwork(self.state_shape,
                                                     self.n_actions).to(self.device)
        self.q_value_target_network2 = QValueNetwork(self.state_shape,
                                                     self.n_actions).to(self.device)

        self.q_value_target_network1.load_state_dict(self.q_value_network1.state_dict())
        self.q_value_target_network1.eval()

        self.q_value_target_network2.load_state_dict(self.q_value_network2.state_dict())
        self.q_value_target_network2.eval()

        self.entropy_target = 0.98 * (-np.log(1 / self.n_actions))
        self.log_alpha = th.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()

        self.q_value1_opt = Adam(self.q_value_network1.parameters(), lr=self.lr)
        self.q_value2_opt = Adam(self.q_value_network2.parameters(), lr=self.lr)
        self.policy_opt = Adam(self.policy_network.parameters(), lr=self.lr)
        self.alpha_opt = Adam([self.log_alpha], lr=self.lr)

        self.update_counter = 0

    
    def train(self,replay_buffer,agent_batch_size):
            print("training agent...")
            
            goals,states,actions, rewards,   next_states ,dones= replay_buffer.get_batch(agent_batch_size)
            goals,states,   actions,rewards, next_states ,dones=th.tensor(goals,dtype=th.float32),th.tensor(states,dtype=th.float32),  th.tensor(actions,dtype=th.int64),th.tensor(rewards,dtype=th.float32),  th.tensor(next_states,dtype=th.float32) ,th.tensor(dones,dtype=th.bool)
            
            #with goal conditioning 
            states = th.cat([states, goals], dim=1)
            next_states = th.cat([next_states, goals], dim=1)
            if self.debug: print("rewards:", rewards.shape)
            if self.debug: print("dones:", dones.shape)



            # Calculating the Q-Value target
            with th.no_grad():
                _, next_probs = self.policy_network(next_states)
                next_log_probs = th.log(next_probs)
                next_q1 = self.q_value_target_network1(next_states)
                next_q2 = self.q_value_target_network2(next_states)
                next_q = th.min(next_q1, next_q2)
                next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(-1).unsqueeze(-1)
                target_q = rewards + self.gamma * (~dones) * next_v

            q1 = self.q_value_network1(states).gather(1, actions)
            q2 = self.q_value_network2(states).gather(1, actions)
            if DEBUG:
                print("rewards:", rewards.shape)
                print("dones:", dones.shape)
                print("next_v:", next_v.shape)
                print("target_q:", target_q.shape)

            q1_loss = F.mse_loss(q1, target_q)
            q2_loss = F.mse_loss(q2, target_q)

            # Calculating the Policy target
            _, probs = self.policy_network(states)
            log_probs = th.log(probs)
            with th.no_grad():
                q1 = self.q_value_network1(states)
                q2 = self.q_value_network2(states)
                q = th.min(q1, q2)

            policy_loss = (probs * (self.alpha.detach() * log_probs - q)).sum(-1).mean()

            self.q_value1_opt.zero_grad()
            q1_loss.backward()
            self.q_value1_opt.step()

            self.q_value2_opt.zero_grad()
            q2_loss.backward()
            self.q_value2_opt.step()

            self.policy_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()

            log_probs = (probs * log_probs).sum(-1)
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.entropy_target)).mean()

            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            self.update_counter += 1

            self.alpha = self.log_alpha.exp()

            if self.update_counter % self.fixed_network_update_freq == 0:
                self.hard_update_target_network()

            return alpha_loss.item(), 0.5 * (q1_loss + q2_loss).item(), policy_loss.item()

    def get_action(self, states, goals,do_greedy=False):
        states = np.array(states)
        goals = np.array(goals) 
        goals = th.tensor(goals,dtype=th.float32)
        states = th.tensor(states,dtype=th.float32)
        states = th.cat([states, goals], dim=0)
        states = np.expand_dims(states, axis=0)

        
        states = from_numpy(states).float().to(self.device)
        with th.no_grad():
            dist, p = self.policy_network(states)
            if do_greedy:
                action = p.argmax(-1)
            else:
                action = dist.sample()
        return action.detach().cpu().numpy()[0]

    def hard_update_target_network(self):
        self.q_value_target_network1.load_state_dict(self.q_value_network1.state_dict())
        self.q_value_target_network1.eval()
        self.q_value_target_network2.load_state_dict(self.q_value_network2.state_dict())
        self.q_value_target_network2.eval()

    def set_to_eval_mode(self):
        self.policy_network.eval()