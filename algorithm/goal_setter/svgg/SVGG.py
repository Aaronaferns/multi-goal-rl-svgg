
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from algorithm.goal_setter import GoalGenerator
from algorithm.goal_setter.svgg.anomalygoal import log_p_valid
from utils import *
import hydra


def rbf_kernel(x, h=None):
    """
    RBF kernel and its gradient.
    :param x: Tensor of shape (n_particles, dim)
    :param h: Bandwidth parameter. If None, use median heuristic.

            h = median(∥xi−xj∥^2) / log(n+1)

    :return: Kernel matrix K and gradient dK
    """
    pairwise_dists = torch.cdist(x, x, p=2).pow(2)  # shape: (n, n)   pairwise euclidian distances
    if h is None:
        h = torch.median(pairwise_dists)
        h = h / torch.log(torch.tensor(x.shape[0], dtype=torch.float32) + 1.0)
        h = torch.clamp(h, min=1e-4)

    K = torch.exp(-pairwise_dists / h)
    return K, h

def log_pskills(D_phi_g, alpha, beta_p, debug=False):
    """
    Computes the log-probability of goals under the skills distribution,
    using a Beta distribution to bias towards goals of desired difficulty.

    Parameters
    ----------
    D_phi_g : torch.Tensor
        Output of the skills model for a batch of goals, typically in [0,1].
        Represents the predicted success probability of achieving each goal.
    
    alpha : float
        α parameter of the Beta distribution. Controls preference for easier goals 
        when α > β, or harder goals when α < β.

    beta_p : float
        β parameter of the Beta distribution. Together with α, defines the target 
        difficulty distribution over the success probabilities.

    Returns
    -------
    torch.Tensor
        The log-probabilities of each goal according to the Beta distribution:

            log P_skills(g) ∝ log Beta(D_phi_g | α, β)
    
    Notes
    -----
    This is used to **weight goal sampling** in RL, favoring goals of specific 
    difficulty levels. The Beta distribution acts as a shaping function over the 
    predicted success probabilities D_phi_g.
    """
    D_phi_g = D_phi_g.clamp(1e-6, 1 - 1e-6)
    beta_dist = Beta(alpha, beta_p)
    return beta_dist.log_prob(D_phi_g)


def log_pgoals_(goals, skills_model, validity_model, alpha, beta, step, temperature,  debug=False):
    """
    Computes the unnormalized log-probability of goals under the combined
    goal distribution:

        p_goals(g) ∝ p_skills(g) · p_valid(g)

    Parameters
    ----------
    goals : torch.Tensor
        A batch of goal states.
        
    skills_model : callable or nn.Module
        The skills model p_skills(g) — a neural network predicting how 
        achievable or successful a goal is given the agent’s skill distribution.
        Typically outputs a scalar success score or log-probability.

    validity_model : callable or nn.Module
        The validity model p_valid(g). Determines if a goal is a valid / 
        reachable state. 
        - For simple state spaces: often implemented as a one-class SVM.
        - For complex (e.g. high-dimensional) spaces: implemented as an 
          autoencoder, where the reconstruction error defines an energy:
            
              E(x) = ||x - f_dec(f_enc(x))||²

          and the validity distribution is derived via a Boltzmann transform:
            
              p_valid(x) ∝ exp( -E(x) / T )

        where `temperature` (T) controls the sharpness of the distribution.

    alpha : float
        α parameter of the Beta distribution (used for regularizing goal priors).

    beta : float
        β parameter of the Beta distribution.

    temperature : float
        Boltzmann temperature used to scale reconstruction-energy-based validity.

    Returns
    -------
    torch.Tensor
        The (unnormalized) log-probability of each goal:
        
            log p_goals(g) = log p_skills(g) + log p_valid(g)
            
    """
    
    D_phi_g = skills_model(goals)
    
    log_p_skills = log_pskills(D_phi_g,alpha,beta)
    log_p_val = log_p_valid(goals,validity_model,step,temperature=temperature)
    if debug:
        print("in_logpgoals")
        print(goals)
        print(D_phi_g)
    
    return log_p_val + log_p_skills

def log_pgoals(goals, model, anomaly_model, alpha, beta, step, temperature=0.1, debug=False):
    logp = log_pgoals_(goals, model, anomaly_model, alpha, beta, step, temperature, debug)
    return logp

def log_pgoals_fn_wrapper(goals,params,step, debug=False):
    model=params["skills_model"]
    anomaly_model = params["validity_model"]
    a = params["a"]
    b = params["b"]
    temperature= params["temperature"]
    return log_pgoals(goals,model,anomaly_model,a,b,step, temperature, debug)







# def test_svgd_with_gaussian():
#     # Target: standard 2D Gaussian
#     n_particles = 100
#     goals = np.random.uniform(low=0.0, high=5.0, size=(n_particles, 2))
#     for step in range(1000):
#         goals = svgd_step(goals, log_p_gaussian, lr=0.1)
#         if step % 100 == 0:
#             print(f"Step {step}: Mean = {np.mean(goals, axis=0)}")

#             # Plot results
#             plt.figure(figsize=(6, 6))
#             plt.scatter(goals[:, 0], goals[:, 1], alpha=0.7)
#             plt.title("SVGD: Particles approximating 2D Gaussian")
#             plt.xlabel("x")
#             plt.ylabel("y")
#             plt.grid(True)
#             plt.show()


# def log_p_gaussian(goals):
#     return -0.5 * ((goals - 2.0)**2).sum(dim=1)



# if __name__ == "__main__":
#     test_svgd_with_gaussian()

class SVGG(GoalGenerator):
    def __init__(self, nG: int, num_traj : int, goals_init: int, num_of_goals: int,
                 LR_ANOMALY: float, LR_SKILLS: float, LR_SVGG: float,
                 BATCH_R: int, BATCH_S: int, SKILLS_ITTR: int,
                 ALPHA: float, BETA_P: float, TEMPERATURE: float, NUM_SVGD_STEPS: int, anomaly_model, skills_model,
                 device: str = "cpu", debug: bool = False, particle_eval_frequency = 1000):
        
        super().__init__()
       
        self.nG = nG
        self.num_traj = num_traj
        self.goals_init = goals_init
        self.num_of_goals = num_of_goals

        # store config values
        self.device = device
        self.ALPHA = ALPHA
        self.BETA_P = BETA_P
        self.TEMPERATURE = TEMPERATURE
        self.NUM_SVGD_STEPS = NUM_SVGD_STEPS
        self.LR_SVGG = LR_SVGG
        self.BATCH_R = BATCH_R
        self.BATCH_S = BATCH_S
        self.SKILLS_ITTR = SKILLS_ITTR
        self.debug = debug
        # models (env & replay_buffer added later)
        self.skills_model =  skills_model.to(self.device)
        self.anomaly_model = anomaly_model.to(self.device)
        
        self.optimizer_anomaly = optim.Adam(self.anomaly_model.parameters(), lr=LR_ANOMALY)
        self.optimizer_skills = optim.Adam(self.skills_model.parameters(), lr=LR_SKILLS)
        self.particle_eval_frequency = particle_eval_frequency

        # env & replay_buffer will be set later
        self.env = None
        self.replay_buffer = None
        self.anomaly_train_step = 0
        self.skills_train_step = 0
        
    def setup_runtime(self, env, replay_buffer):
        """Call this after Hydra creates the SVGG instance"""
        self.env = env
        self.replay_buffer = replay_buffer
        self.goals = self.create_goals(env, self.goals_init, self.num_of_goals)
    
    
    
    
    def svgd_step(self, step, goals, log_prob_fn, log_prob_fn_params={},  lr=1e-2, debug=False):
        """
        Perform one SVGD update step.
        :param goals: Tensor of shape (n_particles, dim)
        :param log_prob_fn: Function that returns log-prob and requires grad
        :param lr: Learning rate (ϵ)
        :return: Updated goals
        """
        goals = torch.tensor(goals,dtype=torch.float32,device=self.device)
        goals = goals.clone().detach().requires_grad_(True)  #use clone because original input goals might be required for logging, detach so that gradients are calculated only wrt to current iteration only
        if debug: print("goals shape:",goals.shape)
        
        n_particles = goals.shape[0]

        log_probs = log_prob_fn(goals,log_prob_fn_params,step, debug)  # shape: (n_particles,)
        grads = torch.autograd.grad(log_probs.sum(), goals)[0]  # shape: (n_particles, dim)

        K, h = rbf_kernel(goals) # inner product
        dK = -2 * (goals.unsqueeze(1) - goals.unsqueeze(0)) / h * K.unsqueeze(2)  # shape: (n, n, dim)

        phi = (K @ grads + dK.sum(dim=0)) / n_particles  # shape: (n_particles, dim)
        
        '''
        Mean log-probability
        Shows whether particles are moving toward high-probability regions of your target distribution.
        Increasing mean log-probability usually indicates progres
        
        variance tells you whether particles are exploring different density regions or have collapsed to similar probabilities.
        '''
        mean_logp = log_probs.mean().item()
        wandb_log("svgd/mean_logp", mean_logp, step)
        var_logp = log_probs.var().item()
        wandb_log("svgd/var_logp", var_logp, step)
        '''
        Distance between particles (diversity)
        Shows whether particles are collapsing or remaining diverse
        Important for SVGD because diversity is crucial.
        '''
        pairwise_dists = torch.cdist(goals, goals, p=2)
        mask = ~torch.eye(n_particles, dtype=torch.bool, device=goals.device)
        mean_dist = pairwise_dists[mask].mean().item()
        wandb_log("svgd/mean_pairwise_dist", mean_dist,step)
        
        '''
        φ magnitude (gradient strength)
        '''
        phi_norm = phi.norm(dim=1).mean().item()
        wandb_log("svgd/phi_norm", phi_norm, step)
        
        
        with torch.no_grad():
            goals += lr * phi
        
        return goals.detach().numpy().astype(np.float32)
    
    def svgd(self,step, goals: torch.Tensor, skills_model, validity_model, ALPHA, BETA_P, TEMPERATURE, NUM_SVGD_STEPS, LR_SVGG, debug = False) -> torch.Tensor: 
        '''
        for t(p) iterations do B Particles Update

            Compute the density of the target pgoals for the set of particles q using [ pgoals(g) ∝ pskills(g).pvalid(g)];
            Compute transforms: φ*(xi)= (1/m)Σj:1->m [k(xj,xi)∇xj logpgoals(xj) + ∇xj k(xj,xi)];
            Update particles xi ← xi + ϵφ∗(xi), ∀i = 1 · · · m;    
        '''
    
        log_prob_fn_params={}

        log_prob_fn_params['skills_model']=skills_model
        log_prob_fn_params['validity_model']=validity_model
        log_prob_fn_params['a']=ALPHA
        log_prob_fn_params['b']=BETA_P
        log_prob_fn_params['temperature']=TEMPERATURE
        
        skills_model.eval()
        validity_model.eval()

        
        for _ in range(NUM_SVGD_STEPS):
            
            goals = self.svgd_step(step, goals, log_pgoals_fn_wrapper, log_prob_fn_params, LR_SVGG, debug)
            
            
            
        return goals.astype(np.float32) 
    
    
    #Initial Goal Generation
    def create_goals(self, env, goals_init, num_of_goals):
        '''
        Parameters:
            goals_init: initial Goal states to Generate
            num_of_goals: More representative Goals. A subsection of goals_init
        
        '''
        #*********************************************#
        def collect_random_states(env, sample_goals = 5, num_goals=1000): 
            '''
            Parameters:
                sample_goals: Goals to sample from each random run
                num_of_goals: Total number of goal states we need
            
            '''
            print("collection goals - Performing preruns...")
            def hash_goal(goal):
                # Convert ndarray to bytes for hashing
                return goal.tobytes()
            seen = set()
            
            goals = []
            while len(goals) < num_goals:
                goals_reached = []
                obs, _ = env.reset()
                while True:
                    action = env.action_space.sample()
                    obs, _ , terminated, truncated, _ = env.step(action)
                    goals_reached.append(obs["achieved_goal"]) 
                    if terminated or truncated:
                        break
                    
                for _ in range(sample_goals):
                    g = random.choice(goals_reached)
                    h = hash_goal(np.asarray(g))
                    if h not in seen:  # ensure uniqueness
                        seen.add(h)
                        goals.append(g)
                    if len(goals) >= num_goals:
                        break
            print("Goals collected.")
            return np.array(goals[:num_goals])
        
        def kmeans_initial_particles(goal_buffer, n_particles=10):
            print("Using KMeans to get a more representative goals...")
            kmeans = KMeans(n_clusters=n_particles)
            kmeans.fit(goal_buffer)
            print("Kmeans complete.")
            return kmeans.cluster_centers_.astype(np.float32)

        
        goal_buffer = collect_random_states(env, num_goals=goals_init)
        initial_particles = kmeans_initial_particles(goal_buffer, n_particles=num_of_goals)
        return initial_particles
        
    def update_skillsmodel(self, step, iterations,batch_size,model,optimizer,loss_fn,debug = False):
        self.skills_train_step+=1
        model.train()
        for t in range(iterations):
            g,success = self.replay_buffer.get_batch_o(batch_size)    
            g,success = torch.tensor(g,dtype=torch.float32).to(self.device),torch.tensor(success,dtype=torch.float32).to(self.device)
            success = torch.unsqueeze(success, dim = 1)
            y=model(g)
            if debug: print("shapes of g, successes, y", g.shape, success.shape, y.shape)
            loss=loss_fn(y,success)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if self.skills_train_step %10==0: model.log(step)
            wandb_log("skills_model/loss", round(loss.item(), 4), step)
            if t % 100 == 0:
                print(f"Iter {t}: Loss = {loss.item():.4f}")

    def update_validmodel(self,step, model,optimizer,BATCH_R,TEMPERATURE,debug = False):
        self.anomaly_train_step+=1 
        batch = self.replay_buffer.get_batch_r(BATCH_R*10)
        if debug: 
            print("batch  = ",batch.shape)
            print(self.replay_buffer.size_r,"buffer r size")
    
        for start_idx in range(0,batch.shape[0],BATCH_R):
            model.train() 
            optimizer.zero_grad()  
            
            end_idx = min(start_idx + BATCH_R, batch.shape[0])
            mini_batch_goals = torch.tensor(batch[start_idx:end_idx], dtype=torch.float32).to(self.device)
            if mini_batch_goals.shape[0] == 0:
                continue
            
            
            if debug: print("mini batch goals = ",mini_batch_goals.shape)
            log_p_val = log_p_valid(mini_batch_goals, model, None, TEMPERATURE)

            # We want to minimize the negative log probability (i.e., minimize reconstruction error)
            loss = -log_p_val.mean()  # Negative because we are maximizing the log probability
            
            # Backpropagation and optimization step
            loss.backward()  # Compute gradients
            optimizer.step()  # Update the model parameters
            
            print(f"Validity Loss: {loss.item():.4f}")
            wandb_log("validity_model/loss", round(loss.item(), 4),step)
            if self.anomaly_train_step%10==0: model.log(step)
            
    def train(self, step):
        loss_fn = nn.BCELoss()
        print("updating skills model...")
        self.update_skillsmodel(step, self.SKILLS_ITTR,self.BATCH_S,self.skills_model,self.optimizer_skills,loss_fn, self.debug)
        print("Skills model updated")
        
        print("updating validity model...")
        self.update_validmodel(step, self.anomaly_model,self.optimizer_anomaly,self.BATCH_R, self.TEMPERATURE, self.debug)
        print("valid model updated")
       
        print("updating goals...")
        self.goals = self.svgd(step, self.goals,self.skills_model,self.anomaly_model, self.ALPHA, self.BETA_P, self.TEMPERATURE, self.NUM_SVGD_STEPS, self.LR_SVGG,self.debug)
        print("svgg goals updated")
    
    def getGoals(self):
        return self.goals
    
    def sample_goal(self):
        row_idx = np.random.choice(len(self.goals))
        sampled_goal = self.goals[row_idx]
        return sampled_goal