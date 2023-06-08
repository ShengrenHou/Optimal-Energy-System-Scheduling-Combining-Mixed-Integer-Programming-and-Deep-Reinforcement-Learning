# ------------------------------------------------------------------------
# Optimal-Energy-System-Scheduling-Combining-Mixed-Integer-Programming-and-Deep-Reinforcement-Learning
# MIP-DQN algorithm developed by
# Hou Shengren, TU Delft, h.shengren@tudelft.nl
# Pedro, TU Delft, p.p.vergara.barrios@tudeflt.nl
# ------------------------------------------------------------------------
import pickle
import torch
import os
import numpy as np
import numpy.random as rd
import pandas as pd
import pyomo.environ as pyo
import pyomo.kernel as pmo
from omlt import OmltBlock

from gurobipy import *
from omlt.neuralnet import NetworkDefinition, FullSpaceNNFormulation,ReluBigMFormulation
from omlt.io.onnx import write_onnx_model_with_bounds,load_onnx_neural_network_with_bounds
import tempfile
import torch.onnx
import torch.nn as nn
from copy import deepcopy
import wandb
from random_generator_battery import ESSEnv
## define net
class ReplayBuffer:
    def __init__(self, max_len, state_dim, action_dim, gpu_id=0):
        self.now_len = 0
        self.next_idx = 0
        self.if_full = False
        self.max_len = max_len
        self.data_type = torch.float32
        self.action_dim = action_dim
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        other_dim = 1 + 1 + self.action_dim
        self.buf_other = torch.empty(size=(max_len, other_dim), dtype=self.data_type, device=self.device)

        if isinstance(state_dim, int):  # state is pixel
            self.buf_state = torch.empty((max_len, state_dim), dtype=torch.float32, device=self.device)
        elif isinstance(state_dim, tuple):
            self.buf_state = torch.empty((max_len, *state_dim), dtype=torch.uint8, device=self.device)
        else:
            raise ValueError('state_dim')

    def extend_buffer(self, state, other):  # CPU array to CPU array
        size = len(other)
        next_idx = self.next_idx + size

        if next_idx > self.max_len:
            self.buf_state[self.next_idx:self.max_len] = state[:self.max_len - self.next_idx]
            self.buf_other[self.next_idx:self.max_len] = other[:self.max_len - self.next_idx]
            self.if_full = True

            next_idx = next_idx - self.max_len
            self.buf_state[0:next_idx] = state[-next_idx:]
            self.buf_other[0:next_idx] = other[-next_idx:]
        else:
            self.buf_state[self.next_idx:next_idx] = state
            self.buf_other[self.next_idx:next_idx] = other
        self.next_idx = next_idx

    def sample_batch(self, batch_size) -> tuple:
        indices = rd.randint(self.now_len - 1, size=batch_size)
        r_m_a = self.buf_other[indices]
        return (r_m_a[:, 0:1],
                r_m_a[:, 1:2],
                r_m_a[:, 2:],
                self.buf_state[indices],
                self.buf_state[indices + 1])

    def update_now_len(self):
        self.now_len = self.max_len if self.if_full else self.next_idx
class Arguments:
    def __init__(self, agent=None, env=None):

        self.agent = agent  # Deep Reinforcement Learning algorithm
        self.env = env  # the environment for training
        self.cwd = None  # current work directory. None means set automatically
        self.if_remove = False  # remove the cwd folder? (True, False, None:ask me)
        self.visible_gpu = '0,1,2,3'  # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.num_threads = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)

        '''Arguments for training'''
        self.num_episode=3000
        self.gamma = 0.995  # discount factor of future rewards
        self.learning_rate = 1e-4  # 2 ** -14 ~= 6e-5
        self.soft_update_tau = 1e-2  # 2 ** -8 ~= 5e-3

        self.net_dim = 64  # the network width 256
        self.batch_size = 256  # num of transitions sampled from replay buffer.
        self.repeat_times = 2 ** 3  # repeatedly update network to keep critic's loss small
        self.target_step = 1000 # collect target_step experiences , then update network, 1024
        self.max_memo = 50000  # capacity of replay buffer
        ## arguments for controlling exploration
        self.explorate_decay=0.99
        self.explorate_min=0.3
        '''Arguments for evaluate'''
        self.random_seed_list=[1234,2234,3234,4234,5234]
        # self.random_seed_list=[2234]
        self.run_name='MIP_DQN_experiments'
        '''Arguments for save'''
        self.train=True
        self.save_network=True

    def init_before_training(self, if_main):
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            self.cwd = f'./{agent_name}/{self.run_name}'

        if if_main:
            import shutil  # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
            elif self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print(f"| Remove cwd: {self.cwd}")
            os.makedirs(self.cwd, exist_ok=True)

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)# control how many GPU is used 　
class Actor(nn.Module):
    def __init__(self,mid_dim,state_dim,action_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(state_dim,mid_dim),nn.ReLU(),
                               nn.Linear(mid_dim,mid_dim),nn.ReLU(),
                               nn.Linear(mid_dim,mid_dim),nn.ReLU(),
                               nn.Linear(mid_dim,action_dim))
    def forward(self,state):
        return self.net(state).tanh()# make the data from -1 to 1
    def get_action(self,state,action_std):#
        action=self.net(state).tanh()
        noise=(torch.randn_like(action)*action_std).clamp(-0.5,0.5)#
        return (action+noise).clamp(-1.0,1.0)
class CriticQ(nn.Module):
    def __init__(self,mid_dim,state_dim,action_dim):
        super().__init__()
        self.net_head=nn.Sequential(nn.Linear(state_dim+action_dim,mid_dim),nn.ReLU(),
                                    nn.Linear(mid_dim,mid_dim),nn.ReLU())
        self.net_q1=nn.Sequential(nn.Linear(mid_dim,mid_dim),nn.ReLU(),
                                  nn.Linear(mid_dim,1))# we get q1 value
        self.net_q2=nn.Sequential(nn.Linear(mid_dim,mid_dim),nn.ReLU(),
                                  nn.Linear(mid_dim,1))# we get q2 value
    def forward(self,value):
        mid=self.net_head(value)
        return self.net_q1(mid)
    def get_q1_q2(self,value):
        mid=self.net_head(value)
        return self.net_q1(mid),self.net_q2(mid)
class AgentBase:
    def __init__(self):
        self.state = None
        self.device = None
        self.action_dim = None
        self.if_off_policy = None
        self.explore_noise = None
        self.trajectory_list = None
        self.explore_rate = 1.0

        self.criterion = torch.nn.SmoothL1Loss()

    def init(self, net_dim, state_dim, action_dim, learning_rate=1e-4, _if_per_or_gae=False, gpu_id=0):
        self.device = torch.device(
            f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.action_dim = action_dim

        self.cri = self.ClassCri(net_dim, state_dim, action_dim).to(self.device)
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(
            self.device) if self.ClassAct else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(),
                                          learning_rate) if self.ClassAct else self.cri
        del self.ClassCri, self.ClassAct

    def select_action(self, state) -> np.ndarray:
        states = torch.as_tensor((state,), dtype=torch.float32, device=self.device)
        action = self.act(states)[0]
        if rd.rand()<self.explore_rate:
            action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1)
        return action.detach().cpu().numpy()

    def explore_env(self, env, target_step):
        trajectory = list()

        state = self.state
        for _ in range(target_step):
            action = self.select_action(state)

            state, next_state, reward, done, = env.step(action)

            trajectory.append((state, (reward, done, *action)))
            state = env.reset() if done else next_state
        self.state = state
        return trajectory

    @staticmethod
    def optim_update(optimizer, objective):
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd, if_save):
        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optim),
                         ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optim), ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]
        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None

    def _update_exploration_rate(self,explorate_decay,explore_rate_min):
        self.explore_rate = max(self.explore_rate * explorate_decay, explore_rate_min)
        '''this function is used to update the explorate probability when select action'''
class AgentMIPDQN(AgentBase):
    def __init__(self):
        super().__init__()
        self.explore_noise = 0.5  # standard deviation of exploration noise
        self.policy_noise = 0.2  # standard deviation of policy noise
        self.update_freq = 2  # delay update frequency
        self.if_use_cri_target = self.if_use_act_target = True
        self.ClassCri = CriticQ
        self.ClassAct = Actor

    def update_net(self, buffer, batch_size, repeat_times, soft_update_tau) -> tuple:
        buffer.update_now_len()
        obj_critic = obj_actor = None
        for update_c in range(int(buffer.now_len / batch_size * repeat_times)):# we update too much time?
            obj_critic, state = self.get_obj_critic(buffer, batch_size)
            self.optim_update(self.cri_optim, obj_critic)

            action_pg = self.act(state)  # policy gradient
            obj_actor = -self.cri_target(torch.cat((state, action_pg),dim=-1)).mean()  # use cri_target instead of cri for stable training
            self.optim_update(self.act_optim, obj_actor)
            if update_c % self.update_freq == 0:  # delay update
                self.soft_update(self.cri_target, self.cri, soft_update_tau)
                self.soft_update(self.act_target, self.act, soft_update_tau)
        return obj_critic.item() / 2, obj_actor.item()

    def get_obj_critic(self, buffer, batch_size) -> (torch.Tensor, torch.Tensor):
        with torch.no_grad():
            reward, mask, action, state, next_s = buffer.sample_batch(batch_size)
            next_a = self.act_target.get_action(next_s, self.policy_noise)  # policy noise,
            next_q = torch.min(*self.cri_target.get_q1_q2(torch.cat((next_s, next_a),dim=-1)))  # twin critics
            q_label = reward + mask * next_q

        q1, q2 = self.cri.get_q1_q2(torch.cat((state, action),dim=-1))
        obj_critic = self.criterion(q1, q_label) + self.criterion(q2, q_label)  # twin critics
        return obj_critic, state



def update_buffer(_trajectory):
    ten_state = torch.as_tensor([item[0] for item in _trajectory], dtype=torch.float32)
    ary_other = torch.as_tensor([item[1] for item in _trajectory])
    ary_other[:, 0] = ary_other[:, 0]   # ten_reward
    ary_other[:, 1] = (1.0 - ary_other[:, 1]) * gamma  # ten_mask = (1.0 - ary_done) * gamma

    buffer.extend_buffer(ten_state, ary_other)

    _steps = ten_state.shape[0]
    _r_exp = ary_other[:, 0].mean()  # other = (reward, mask, action)
    return _steps, _r_exp


def get_episode_return(env, act, device):
    '''get information of one episode during the training'''
    episode_return = 0.0  # sum of rewards in an episode
    episode_unbalance=0.0
    episode_operation_cost=0.0
    state = env.reset()
    for i in range(24):
        s_tensor = torch.as_tensor((state,), device=device)
        a_tensor = act(s_tensor)
        action = a_tensor.detach().cpu().numpy()[0]  # not need detach(), because with torch.no_grad() outside
        state, next_state, reward, done,= env.step(action)
        state=next_state
        episode_return += reward
        episode_unbalance+=env.real_unbalance
        episode_operation_cost+=env.operation_cost
        if done:
            break
    return episode_return,episode_unbalance,episode_operation_cost
class Actor_MIP:
    '''this actor is used to get the best action and Q function, the only input should be batch tensor state, action, and network, while the output should be
    batch tensor max_action, batch tensor max_Q'''
    def __init__(self,scaled_parameters,batch_size,net,state_dim,action_dim,env,constrain_on=False):
        self.batch_size = batch_size
        self.net = net
        self.state_dim = state_dim
        self.action_dim =action_dim
        self.env = env
        self.constrain_on=constrain_on
        self.scaled_parameters=scaled_parameters

    def get_input_bounds(self,input_batch_state):
        batch_size = self.batch_size
        batch_input_bounds = []
        lbs_states = input_batch_state.detach().numpy()
        ubs_states = lbs_states

        for i in range(batch_size):
            input_bounds = {}
            for j in range(self.action_dim + self.state_dim):
                if j < self.state_dim:
                    input_bounds[j] = (float(lbs_states[i][j]), float(ubs_states[i][j]))
                else:
                    input_bounds[j] = (float(-1), float(1))
            batch_input_bounds.append(input_bounds)
        return batch_input_bounds

    def predict_best_action(self, state):
        state=state.detach().cpu().numpy()
        v1 = torch.zeros((1, self.state_dim+self.action_dim), dtype=torch.float32)
        '''this function is used to get the best action based on current net'''
        model = self.net.to('cpu')
        input_bounds = {}
        lb_state = state
        ub_state = state
        for i in range(self.action_dim + self.state_dim):
            if i < self.state_dim:
                input_bounds[i] = (float(lb_state[0][i]), float(ub_state[0][i]))
            else:
                input_bounds[i] = (float(-1), float(1))

        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            # export neural network to ONNX
            torch.onnx.export(
                model,
                v1,
                f,
                input_names=['state_action'],
                output_names=['Q_value'],
                dynamic_axes={
                    'state_action': {0: 'batch_size'},
                    'Q_value': {0: 'batch_size'}
                }
            )
            # write ONNX model and its bounds using OMLT
        write_onnx_model_with_bounds(f.name, None, input_bounds)
        # load the network definition from the ONNX model
        network_definition = load_onnx_neural_network_with_bounds(f.name)
        # global optimality
        formulation = ReluBigMFormulation(network_definition)
        m = pyo.ConcreteModel()
        m.nn = OmltBlock()
        m.nn.build_formulation(formulation)
        '''# we are now building the surrogate model between action and state'''
        # constrain for battery，
        if self.constrain_on:
            m.power_balance_con1 = pyo.Constraint(expr=(
                    (-m.nn.inputs[7] * self.scaled_parameters[0])+\
                    ((m.nn.inputs[8] * self.scaled_parameters[1])+m.nn.inputs[4]*self.scaled_parameters[5]) +\
                    ((m.nn.inputs[9] * self.scaled_parameters[2])+m.nn.inputs[5]*self.scaled_parameters[6]) +\
                    ((m.nn.inputs[10] * self.scaled_parameters[3])+m.nn.inputs[6]*self.scaled_parameters[7])>=\
                    m.nn.inputs[3] *self.scaled_parameters[4]-self.env.grid.exchange_ability))
            m.power_balance_con2 = pyo.Constraint(expr=(
                    (-m.nn.inputs[7] * self.scaled_parameters[0])+\
                    (m.nn.inputs[8] * self.scaled_parameters[1]+m.nn.inputs[4]*self.scaled_parameters[5]) +\
                    (m.nn.inputs[9] * self.scaled_parameters[2]+m.nn.inputs[5]*self.scaled_parameters[6]) +\
                    (m.nn.inputs[10] * self.scaled_parameters[3]+m.nn.inputs[6]*self.scaled_parameters[7])<=\
                    m.nn.inputs[3] *self.scaled_parameters[4]+self.env.grid.exchange_ability))
        m.obj = pyo.Objective(expr=(m.nn.outputs[0]), sense=pyo.maximize)

        pyo.SolverFactory('gurobi').solve(m, tee=False)

        best_input = pyo.value(m.nn.inputs[:])

        best_action = (best_input[self.state_dim::])
        return best_action
# define test function
if __name__ == '__main__':
    args = Arguments()
    '''here record real unbalance'''
    reward_record = {'episode': [], 'steps': [], 'mean_episode_reward': [], 'unbalance': [],
                     'episode_operation_cost': []}
    loss_record = {'episode': [], 'steps': [], 'critic_loss': [], 'actor_loss': [], 'entropy_loss': []}
    args.visible_gpu = '2'
    for seed in args.random_seed_list:
        args.random_seed = seed
        # set different seed
        args.agent = AgentMIPDQN()
        agent_name = f'{args.agent.__class__.__name__}'
        args.agent.cri_target = True
        args.env = ESSEnv()
        args.init_before_training(if_main=True)
        '''init agent and environment'''
        agent = args.agent
        env = args.env
        agent.init(args.net_dim, env.state_space.shape[0], env.action_space.shape[0], args.learning_rate,
                   args.if_per_or_gae)
        '''init replay buffer'''
        buffer = ReplayBuffer(max_len=args.max_memo, state_dim=env.state_space.shape[0],
                              action_dim=env.action_space.shape[0])
        '''start training'''
        cwd = args.cwd
        gamma = args.gamma
        batch_size = args.batch_size  # how much data should be used to update net
        target_step = args.target_step  # how manysteps of one episode should stop
        repeat_times = args.repeat_times  # how many times should update for one batch size data
        soft_update_tau = args.soft_update_tau
        agent.state = env.reset()
        '''collect data and train and update network'''
        num_episode = args.num_episode
        args.train=True
        args.save_network=True
        wandb.init(project='MIP_DQN_experiments',name=args.run_name,settings=wandb.Settings(start_method="fork"))
        wandb.config = {
            "epochs": num_episode,
            "batch_size": batch_size}
        wandb.define_metric('custom_step')
        if args.train:
            collect_data = True
            while collect_data:
                print(f'buffer:{buffer.now_len}')
                with torch.no_grad():
                    trajectory = agent.explore_env(env, target_step)

                    steps, r_exp = update_buffer(trajectory)
                    buffer.update_now_len()
                if buffer.now_len >= 10000:
                    collect_data = False
            for i_episode in range(num_episode):
                critic_loss, actor_loss = agent.update_net(buffer, batch_size, repeat_times, soft_update_tau)
                wandb.log({'critic loss':critic_loss,'custom_step':i_episode})
                wandb.log({'actor loss': actor_loss,'custom_step':i_episode})
                loss_record['critic_loss'].append(critic_loss)
                loss_record['actor_loss'].append(actor_loss)
                with torch.no_grad():
                    episode_reward, episode_unbalance, episode_operation_cost = get_episode_return(env, agent.act,
                                                                                             agent.device)
                    wandb.log({'mean_episode_reward': episode_reward,'custom_step':i_episode})
                    wandb.log({'unbalance':episode_unbalance,'custom_step':i_episode})
                    wandb.log({'episode_operation_cost':episode_operation_cost,'custom_step':i_episode})
                    reward_record['mean_episode_reward'].append(episode_reward)
                    reward_record['unbalance'].append(episode_unbalance)
                    reward_record['episode_operation_cost'].append(episode_operation_cost)

                print(
                    f'curren epsiode is {i_episode}, reward:{episode_reward},unbalance:{episode_unbalance},buffer_length: {buffer.now_len}')
                if i_episode % 10 == 0:
                    # target_step
                    with torch.no_grad():
                        agent._update_exploration_rate(args.explorate_decay,args.explorate_min)
                        trajectory = agent.explore_env(env, target_step)
                        steps, r_exp = update_buffer(trajectory)
        wandb.finish()
    if args.update_training_data:
        loss_record_path = f'{args.cwd}/loss_data.pkl'
        reward_record_path = f'{args.cwd}/reward_data.pkl'
        with open(loss_record_path, 'wb') as tf:
            pickle.dump(loss_record, tf)
        with open(reward_record_path, 'wb') as tf:
            pickle.dump(reward_record, tf)
    act_save_path = f'{args.cwd}/actor.pth'
    cri_save_path = f'{args.cwd}/critic.pth'

    print('training data have been saved')
    if args.save_network:
        torch.save(agent.act.state_dict(), act_save_path)
        torch.save(agent.cri.state_dict(), cri_save_path)
        print('training finished and actor and critic parameters have been saved')


