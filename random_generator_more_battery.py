
import random
import numpy as np

import pandas as pd 
import gym
from gym import spaces 
import math 
import os 
import sys
from Parameters import battery_parameters,dg_parameters

class Constant:
	MONTHS_LEN = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
	MAX_STEP_HOURS = 24 * 30
class DataManager():
    def __init__(self) -> None:
        self.PV_Generation=[]
        self.Prices=[]
        self.Electricity_Consumption=[]

    def add_pv_element(self,element):self.PV_Generation.append(element)
    def add_price_element(self,element):self.Prices.append(element)
    def add_electricity_element(self,element):self.Electricity_Consumption.append(element)

    # get current time data based on given month day, and day_time
    def get_pv_data(self,month,day,day_time):return self.PV_Generation[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+day_time]
    def get_price_data(self,month,day,day_time):return self.Prices[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+day_time]
    def get_electricity_cons_data(self,month,day,day_time):return self.Electricity_Consumption[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+day_time]
    # get series data for one episode
    def get_series_pv_data(self,month,day): return self.PV_Generation[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24:(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+24]
    def get_series_price_data(self,month,day):return self.Prices[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24:(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+24]
    def get_series_electricity_cons_data(self,month,day):return self.Electricity_Consumption[(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24:(sum(Constant.MONTHS_LEN[:month-1])+day-1)*24+24]
class DG():
    '''simulate a simple diesel generator here'''
    def __init__(self,parameters):
        self.name=parameters.keys()
        self.a_factor=parameters['a']
        self.b_factor=parameters['b']
        self.c_factor=parameters['c']
        self.power_output_max=parameters['power_output_max']
        self.power_output_min=parameters['power_output_min']
        self.ramping_up=parameters['ramping_up']
        self.ramping_down=parameters['ramping_down']
        self.last_step_output=None 
    def step(self,action_gen):
        ##god damn fuck, I forget to set each generator could be zero. 
        output_change=action_gen*self.ramping_up# constrain the output_change with ramping up boundary
        output=self.current_output+output_change
        if output>0:
            output=max(self.power_output_min,min(self.power_output_max,output))# meet the constrain 
        else:
            output=0
        self.current_output=output
    def _get_cost(self,output):
        # here transfer mw parameters to kw parameters, avarage max cost per unit max [15,22]
        if output<=0:
            cost=0
        else:
            cost=(self.a_factor*pow(output,2)+self.b_factor*output+self.c_factor)
        # print(cost)
        return cost 
    def reset(self):
        self.current_output=0
class Battery():
    def __init__(self,parameters):
        self.capacity=parameters['capacity']# 容量
        self.max_soc=parameters['max_soc']# max soc 0.8
        self.initial_capacity=parameters['initial_capacity']# initial soc 0.4
        self.min_soc=parameters['min_soc']# 0.2
        self.degradation=parameters['degradation']# degradation cost 0，
        self.max_charge=parameters['max_charge']# max charge ability
        self.max_discharge=parameters['max_discharge']# max discharge ability
        self.efficiency=parameters['efficiency']# charge and discharge efficiency
    def step(self,action_battery):
        '''receive battery action, here is the action [-1,1] spaces and then update SOC with the constrains of charge/discharge, SOC boundaries'''
        # max(min_state_value,min(max_state_value,s+action))
        energy=action_battery*self.max_charge
        updated_capacity=max(self.min_soc,min(self.max_soc,(self.current_capacity*self.capacity+energy)/self.capacity))
        self.energy_change=(updated_capacity-self.current_capacity)*self.capacity# if charge, positive, if discharge, negative
        self.current_capacity=updated_capacity# update capacity to current codition
    def _get_cost(self,energy):# calculate the cost depends on the energy change
        cost=energy**2*self.degradation
        return cost  
    def SOC(self):
        return self.current_capacity
    def reset(self):
        self.current_capacity=np.random.uniform(0.2,0.8)
class Grid():
    def __init__(self):
        
        self.on=True
        if self.on:
            self.exchange_ability=30
        else:
            self.exchange_ability=0
    def _get_cost(self,current_price,energy_exchange):##energy if charge, will be positive, if discharge will be negative
        return current_price*energy_exchange
    def retrive_past_price(self):
        result=[]
        if self.day<1:
            past_price=self.past_price# self.past price is fixed as the last days price
        else:
            past_price=self.price[24*(self.day-1):24*self.day]# get the price data of previous day 
            # print(past_price)
        for item in past_price[(self.time-24)::]:# here if current time_step is 10, then the 10th data of past price is extrated to the result as the  first value
            result.append(item)
        for item in self.price[24*self.day:(24*self.day+self.time)]:# continue to retrive data from the past and attend it to the result. as past price is change everytime. 
            result.append(item)
        return result 
class ESSEnv(gym.Env):
    '''ENV descirption: 
    the agent learn to charge with low price and then discharge at high price, in this way, it could get benefits'''
    def __init__(self,**kwargs):
        super(ESSEnv,self).__init__()
        #parameters 
        self.data_manager=DataManager()
        self._load_year_data()
        self.episode_length=kwargs.get('episode_length',24)
        self.month=None
        self.day=None
        # Control training set and validation set with reset function
        self.TRAIN=True
        self.current_time=None
        self.battery_parameters=kwargs.get('battery_parameters',battery_parameters)
        self.dg_parameters=kwargs.get('dg_parameters',dg_parameters)
        self.penalty_coefficient=20#control soft penalty constrain
        self.sell_coefficient=0.5# control sell benefits
        # instant the components of the environment
        self.grid=Grid()
        self.battery1=Battery(self.battery_parameters)
        self.battery2=Battery(self.battery_parameters)
        self.battery3=Battery(self.battery_parameters)
        self.dg1=DG(self.dg_parameters['gen_1'])
        self.dg2=DG(self.dg_parameters['gen_2'])
        self.dg3=DG(self.dg_parameters['gen_3'])

        # define normalized action space 
        self.action_space=spaces.Box(low=-1,high=1,shape=(6,),dtype=np.float32)# seems here doesn't used
        self.state_space=spaces.Box(low=0,high=1,shape=(9,),dtype=np.float32)
        self.Length_max=24
        self.Price_max=max(self.data_manager.Prices)
        self.Netload_max = max(self.data_manager.Electricity_Consumption)
        self.SOC_max=self.battery1.max_soc
        self.DG1_max=self.dg1.power_output_max
        self.DG2_max=self.dg2.power_output_max
        self.DG3_max=self.dg3.power_output_max


    def reset(self):
        '''reset is used for initialize the environment, decide the day of month.'''
        self.month=np.random.randint(1,13)# here we choose 12 month

        if self.TRAIN:
            self.day=np.random.randint(1,21)
        else:
            self.day=np.random.randint(21,Constant.MONTHS_LEN[self.month-1])
        self.current_time=0
        self.battery1.reset()
        self.battery2.reset()
        self.battery3.reset()
        self.dg1.reset()
        self.dg2.reset()
        self.dg3.reset()
        return self._build_state()
    def _build_state(self):
        soc1=self.battery1.SOC()/self.SOC_max
        soc2=self.battery2.SOC()/self.SOC_max
        soc3 = self.battery3.SOC() / self.SOC_max
        dg1_output=self.dg1.current_output/self.DG1_max
        dg2_output=self.dg2.current_output/self.DG2_max
        dg3_output=self.dg3.current_output/self.DG3_max
        time_step=self.current_time/(self.Length_max-1)
        electricity_demand=self.data_manager.get_electricity_cons_data(self.month,self.day,self.current_time)
        pv_generation=self.data_manager.get_pv_data(self.month,self.day,self.current_time)
        price=self.data_manager.get_price_data(self.month,self.day,self.current_time)/self.Price_max
        net_load=(electricity_demand-pv_generation)/self.Netload_max
        obs=np.concatenate((np.float32(time_step),np.float32(price),np.float32(soc1),np.float32(soc2),np.float32(soc3),np.float32(net_load),np.float32(dg1_output),np.float32(dg2_output),np.float32(dg3_output)),axis=None)
        return obs

    def step(self,action):# state transition here current_obs--take_action--get reward-- get_finish--next_obs
        ## here we want to put take action into each components
        current_obs=self._build_state()
        self.battery1.step(action[0])# here execute the state-transition part, battery.current_capacity also changed
        self.battery2.step(action[1])
        self.battery3.step(action[2])
        self.dg1.step(action[3])
        self.dg2.step(action[4])
        self.dg3.step(action[5])
        current_output=np.array((self.dg1.current_output,self.dg2.current_output,self.dg3.current_output,-self.battery1.energy_change,-self.battery2.energy_change,-self.battery3.energy_change))#truely corresonding to the result
        self.current_output=current_output
        actual_production=sum(current_output)
        netload=current_obs[5]*self.Netload_max
        price=current_obs[1]*self.Price_max

        unbalance=actual_production-netload
        reward=0
        excess_penalty=0
        deficient_penalty=0
        sell_benefit=0
        buy_cost=0
        self.excess=0
        self.shedding=0
        # logic here is: if unbalance >0 then it is production excess, so the excessed output should sold to power grid to get benefits 
        if unbalance>=0:# it is now in excess condition
            if unbalance<=self.grid.exchange_ability:
                sell_benefit=self.grid._get_cost(price,unbalance)*self.sell_coefficient #sell money to grid is little [0.029,0.1]
            else:
                sell_benefit=self.grid._get_cost(price,self.grid.exchange_ability)*self.sell_coefficient
                #real unbalance that even grid could not meet 
                self.excess=unbalance-self.grid.exchange_ability
                excess_penalty=self.excess*self.penalty_coefficient
        else:# unbalance <0, its load shedding model, in this case, deficient penalty is used 
            if abs(unbalance)<=self.grid.exchange_ability:
                buy_cost=self.grid._get_cost(price,abs(unbalance))
            else:
                buy_cost=self.grid._get_cost(price,self.grid.exchange_ability)
                self.shedding=abs(unbalance)-self.grid.exchange_ability
                deficient_penalty=self.shedding*self.penalty_coefficient
        battery1_cost=self.battery1._get_cost(self.battery1.energy_change)# we set it as 0 this time
        battery2_cost=self.battery2._get_cost(self.battery2.energy_change)
        battery3_cost = self.battery3._get_cost(self.battery3.energy_change)
        dg1_cost=self.dg1._get_cost(self.dg1.current_output)
        dg2_cost=self.dg2._get_cost(self.dg2.current_output)
        dg3_cost=self.dg3._get_cost(self.dg3.current_output)

        reward=-(battery1_cost+battery2_cost+battery3_cost+dg1_cost+dg2_cost+dg3_cost+excess_penalty+
        deficient_penalty-sell_benefit+buy_cost)/2e3

        self.operation_cost=battery1_cost+battery2_cost+battery3_cost+dg1_cost+dg2_cost+dg3_cost+buy_cost-sell_benefit+(self.shedding+self.excess)*self.penalty_coefficient

        self.unbalance=unbalance
        self.real_unbalance=self.shedding+self.excess
        '''here we also need to store the final step outputs for the final steps including, soc, output of units for seeing the final states'''
        final_step_outputs=[self.dg1.current_output,self.dg2.current_output,self.dg3.current_output,self.battery1.current_capacity,self.battery2.current_capacity,self.battery3.current_capacity]
        self.current_time+=1
        finish=(self.current_time==self.episode_length)
        if finish:
            self.final_step_outputs=final_step_outputs
            self.current_time=0
            next_obs=self.reset()
            
        else:
            next_obs=self._build_state()
        return current_obs,next_obs,float(reward),finish
    def render(self, current_obs, next_obs, reward, finish):
        # print('day={}'.format(self.day))
        print('day={},hour={:2d}, state={}, next_state={}, reward={:.4f}, terminal={}\n'.format(self.day,self.current_time, current_obs, next_obs, reward, finish))
    def _load_year_data(self):
        '''this private function is used to load the electricity consumption, pv generation and related prices in a year as 
        a one hour resolution, with the cooperation of class DataProcesser and then all these data are stored in data processor'''
        pv_df=pd.read_csv('data/PV.csv',sep=';')
        #hourly price data for a year 
        price_df=pd.read_csv('data/Prices.csv',sep=';')
        # mins electricity consumption data for a year 
        electricity_df=pd.read_csv('data/H4.csv',sep=';')
        pv_data=pv_df['P_PV_'].apply(lambda x: x.replace(',','.')).to_numpy(dtype=float)
        price=price_df['Price'].apply(lambda x:x.replace(',','.')).to_numpy(dtype=float)
        electricity=electricity_df['Power'].apply(lambda x:x.replace(',','.')).to_numpy(dtype=float)
        # netload=electricity-pv_data
        '''we carefully redesign the magnitude for price and amount of generation as well as demand'''
        for element in pv_data:
            self.data_manager.add_pv_element(element*100)
        for element in price:
            element/=10
            if element<=0.5:
                element=0.5
            self.data_manager.add_price_element(element)
        for i in range(0,electricity.shape[0],60):
            element=electricity[i:i+60]
            self.data_manager.add_electricity_element(sum(element)*300)
    ## test environment
if __name__ == '__main__': 
    '''here we need a function that could validate 
    whether the current month, day and time could coordinate to sent data
    8,December coordination of data is test from this way, that after 24 steps, we rechoose the month, day and reset current time= 0 '''
    env=ESSEnv()
    env.TRAIN=False
    rewards=[]
    env.reset()
    tem_action=[0.1,0.1,0.1,0.1,0.1,0.1]
    for _ in range (240):
        print(f'current month is {env.month}, current day is {env.day}, current time is {env.current_time}')
        current_obs,next_obs,reward,finish=env.step(tem_action)
        env.render(current_obs,next_obs,reward,finish)
        current_obs=next_obs
        rewards.append(reward)
        
    # print(f'total reward{sum(rewards)}')

## after debug, it could work now. 