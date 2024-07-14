import numpy as np

def simulation_params(t_min,t_max,t_step): 
    t_span = (t_min, t_max)
    num_step = int((t_max-t_min)/t_step)
    t_eval = np.linspace(t_min,t_max, num_step)
    return t_span,t_eval,num_step
