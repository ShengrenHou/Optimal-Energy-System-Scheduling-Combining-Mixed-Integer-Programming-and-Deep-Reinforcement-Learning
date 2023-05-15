battery_parameters={
'capacity':500,# kw
'max_charge':100, # kw
'max_discharge':100, #kw
'efficiency':0.9,
'degradation':0, #euro/kw
'max_soc':0.8,
'min_soc':0.2,
'initial_capacity':0.2}


dg_parameters={
'gen_1':{'a':0.0034
,'b': 3
,'c':30
,'d': 0.03,'e':4.2,'f': 0.031,'power_output_max':150,'power_output_min':0,'heat_output_max':None,'heat_output_min':None,\
'ramping_up':100,'ramping_down':100,'min_up':2,'min_down':1},

'gen_2':{'a':0.001
,'b': 10
,'c': 40
,'d': 0.03,'e':4.2,'f': 0.031,'power_output_max':375,'power_output_min':0,'heat_output_max':None,'heat_output_min':None,\
    'ramping_up':100,'ramping_down':100,'min_up':2,'min_down':1},

'gen_3':{'a':0.001
,'b': 15
,'c': 70
,'d': 0.03,'e':4.2,'f': 0.031,'power_output_max':500,'power_output_min':0,'heat_output_max':None,'heat_output_min':None,\
    'ramping_up':200,'ramping_down':200,'min_up':2,'min_down':1}}

