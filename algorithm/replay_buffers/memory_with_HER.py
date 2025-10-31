import numpy as np
class Buffer:
    def __init__(self,env,nS,nA,nG,reward_fn,buffer_size=50000,useHindsight=True, debug = False):
        self.useHindsight=useHindsight
        buffer_size = int(buffer_size)
        self.buffer_size = buffer_size
        self.env = env
        self.g = np.empty((buffer_size,nG))
        self.s = np.empty((buffer_size,nS))
        self.a = np.empty((buffer_size,nA))
        self.s_ = np.empty((buffer_size,nS))
        self.r = np.empty((buffer_size,1))
        self.success = np.empty((buffer_size,1))
        
        self.R = np.empty((buffer_size, nG))
        self.O = np.empty((buffer_size, nG+1))
        self.size = 0
        self.idx = -1
        self.idx_o = -1
        self.size_o = 0
        self.idx_r = -1
        self.size_r = 0
        self.full = False
        self.k = 2
        self.reward_fn = reward_fn
        self.debug = debug
    
    def add(self,trajectory_memory):
        
        tm=trajectory_memory #its a list of len tlen (g,s,a,r,s_,success,info)
        
        len_tm=len(tm)
        for t in range(len_tm):
           
            self.idx+=1
            if self.idx>=self.buffer_size-1: self.full=True
            if self.idx>=self.buffer_size:
                self.idx=0
            if self.size!=self.buffer_size:
                self.size+=1
                
            g,obs,a,r,obs_,success,info = tm[t]
            self.s[self.idx] = obs["observation"]
            self.s_[self.idx] = obs_["observation"]
            self.a[self.idx] = a
            self.r[self.idx] = r
            self.g[self.idx] = g
            self.success[self.idx] = success
            
            self.add_R(obs["achieved_goal"])
            if success: self.add_R(obs_["achieved_goal"]) #make sure to not miss adding goal states

            if self.useHindsight:
                
                #use future strategy
                future_range = np.arange(t+1,len_tm)
                if len(future_range) == 0:
                    continue
                future_idxs = np.random.choice(future_range,size=self.k,replace=True)
                for f_idx in future_idxs:
                    self.idx+=1
                    if self.idx>=self.buffer_size-1: self.full=True
                    if self.idx>=self.buffer_size:
                        self.idx=0
                    if self.size!=self.buffer_size:
                        self.size+=1
                        
                    _,obs_f,_,_,_,_,_=tm[f_idx]
                    self.s[self.idx] = obs["observation"]
                    self.s_[self.idx] = obs_["observation"]
                    self.a[self.idx]=a
                    substitute_goal = obs_f["achieved_goal"]
                    self.g[self.idx]= substitute_goal
                    reward = self.env.unwrapped.compute_reward(obs_["achieved_goal"], substitute_goal, info)
                    self.r[self.idx] = reward
                    self.success[self.idx] = 1 if reward == 0 else 0  
                    self.add_O([self.g[self.idx], self.success[self.idx]])   
        self.add_O([g, int(success)])
        
        
    def get_batch(self,batch_size):
        indices = np.random.choice(np.arange(0,self.size),size=batch_size,replace=False)
        if self.debug:print(self.r.shape)
        return self.g[indices],self.s[indices],self.a[indices],self.r[indices],self.s_[indices],self.success[indices]
        
    
    def add_O(self,outcome):
        self.idx_o+=1
        if self.idx_o>=self.buffer_size:
            self.idx_o=0
        if self.size_o!=self.buffer_size:
            self.size_o+=1
        self.O[self.idx_o,:-1]=outcome[0]
        self.O[self.idx_o,-1]=outcome[1]
        
    def get_batch_o(self,batch_size):
        success = self.O[:self.size_o, -1].astype(int)  
        success_idx = np.where(success == 1)[0]
        failure_idx = np.where(success == 0)[0]
        if self.debug:
            print(success_idx.shape)
            print(failure_idx.shape)
        half = batch_size // 2
        sampled_success = np.random.choice(success_idx, size=half, replace=len(success_idx) < half)
        sampled_failure = np.random.choice(failure_idx, size=half, replace=len(failure_idx) < half)
        indices = np.concatenate([sampled_success, sampled_failure])
        np.random.shuffle(indices)
        return self.O[indices,:-1],self.O[indices,-1]
    
    def add_R(self,g):
        self.idx_r+=1
        if self.idx_r>=self.buffer_size:
            self.idx_r=0
        if self.size_r!=self.buffer_size:
            self.size_r+=1
        self.R[self.idx_r,:] = g
    def get_batch_r(self, batch_size):
        replace_flag = batch_size > self.size_r
        indices = np.random.choice(self.size_r, size=batch_size, replace=replace_flag)
        return self.R[indices]

        

        
        