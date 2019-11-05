from __future__ import division
from itertools import permutations,combinations
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
AGENTS = [1,2,3,4,5,6,7]
AGENT_X_LOCATION = [0,-2.4,2.15,2.15,5.1,5.5,4]
AGENT_X_NOISE = [0,0.7,0.6,0.6,1.2,2,6]
AGENT_Y_NOISE = [1,4,2,2,3.5,1,2]
AGENT_DATAPOINTS = [10,100,100,100,100,97,160]
AGENT_COLOR = "bgrcmykw"
PARAMETER = 3
CONTRIBUTION = [0,150,250,250,750,3000,5000]
PRIOR_COV = 1

AGENT_X_LOCATION = [0,-2.4,2.8,2.8,5.3,5.3,4.5]
AGENT_X_NOISE = [0,0.7,0.6,0.6,1.2,2,6]
AGENT_Y_NOISE = [1,4,1.8,1.8,3.5,1,2]
AGENT_DATAPOINTS = [1,100,100,100,120,97,110]
AGENT_COLOR = "bgrcmykw"
PARAMETER = 3
CONTRIBUTION = [0,150,250,250,750,3000,5000]
PRIOR_COV = 1

"""
This module contains code for the calculation of the Shapley value in cooperative games.
"""

def power_set(List):
    """
    function to return the powerset of a list
    """
    subs = [list(j) for i in range(len(List)) for j in combinations(List, i+1)]
    return subs


def characteristic_function_check(player_list,characteristic_function):
    """
    A function to check if a characteristic_function is valid
    """
    r=True
    player_power_set=power_set(player_list)
    for e in player_power_set:
        if ",".join(e) not in characteristic_function:
            print ("ERROR: characteristic_function domain does not match players.")
            return False
    for e in player_power_set:
        e=",".join(e)
        for b in player_power_set:
            b=",".join(b)
            if e in b:
                if characteristic_function[e]>characteristic_function[b]:
                    print ("ERROR: game is not Monotone")
                    return False
    return r

def predecessors(player,player_permutation):
    """
    A function to return the predecessors of a player
    """
    r=[]
    for e in player_permutation:
        if e!=player:
            r.append(e)
        else:
            break
    return r

def Marginal_Contribution(player,player_permutation,characteristic_function):
    """
    A function to return the marginal contribution of a player for a given permutation
    """
    pred=predecessors(player,player_permutation)
    predecessor_contribution=0
    for e in permutations(pred):
        e=",".join(e)
        if e in characteristic_function:
            predecessor_contribution=characteristic_function[e]
    pred.append(player)
    for e in permutations(pred):
        e=",".join(e)
        if e in characteristic_function:
            return characteristic_function[e]-predecessor_contribution


def Shapley_calculation(player_list,characteristic_function):
    """
    A function to return the shapley value of a game
    """
    Marginal_Contribution_dict={}
    for e in player_list:
        Marginal_Contribution_dict[e]=0
    k=0
    for pi in permutations(player_list):
        k+=1
        for e in player_list:
            Marginal_Contribution_dict[e]+=Marginal_Contribution(e,pi,characteristic_function)
    for e in Marginal_Contribution_dict:
        Marginal_Contribution_dict[e]/=k
    return Marginal_Contribution_dict

def MI(X, noiseVector, wCovariance):
    '''
    Python code to calculate MI for Bayesian Linear Regression with known noise variance:
    X is the design matrix (input locations),
    noiseVector is an array of known noise variance for each point
    and wCovariance is the prior covariance of the weight parameters.
    '''
    A = np.diagflat(noiseVector)

    det = np.linalg.det(wCovariance) * np.linalg.det(np.linalg.inv(wCovariance) + X.T @ np.linalg.inv(A) @ X)
    return 0.5 * np.log2(det)

def create_data_linear_model(mean_location,std_location,std_output,parameter,n_points=20):
    x_input = std_location*np.random.randn(n_points) + mean_location
    output_error = std_output*np.random.randn(n_points)
    y_output = parameter * x_input + output_error

    return x_input, y_output, std_output

def get_char_function_coalition(coalition, agent_data):
    # coalition is a list of integer
    all_X = np.array([])
    all_y_std = np.array([])
    for agent in coalition:
        # agent is an integer
        X_agent = agent_data[agent-1][0] # list of X of agent
        y_std_agent = agent_data[agent-1][2] # an integer denoting noise of agent
        y_std_agent_list = [y_std_agent] * len(X_agent) # list of y std (duplicates of noise of agent)
        all_X=np.append(all_X, X_agent)
        all_y_std=np.append(all_y_std,y_std_agent_list)
    prior_cov = np.diag([PRIOR_COV] * len(all_X))
    print(coalition)
    print(MI(np.array(all_X).T, all_y_std, prior_cov))
    return MI(np.array(all_X).T, all_y_std, prior_cov)

def create_all_char_function_dict(p_set, agent_data):
    char_functions = {}
    for coalition in p_set:
        key = ','.join([str(x) for x in coalition])
        # get char of this coalition
        MI_coalition = get_char_function_coalition(coalition, agent_data)
        char_functions[key]=MI_coalition
    return char_functions

# create data
agent_data = []
for mean_location, std_location, std_output, n_points in zip(AGENT_X_LOCATION, AGENT_X_NOISE, AGENT_Y_NOISE, AGENT_DATAPOINTS):
    agent_data.append(create_data_linear_model(mean_location,std_location,std_output,PARAMETER, n_points))

p_set = (power_set(AGENTS))
char_functions = create_all_char_function_dict(p_set, agent_data)
shapley = (Shapley_calculation([str(x) for x in AGENTS],char_functions))


CONTRIBUTION = list(shapley.values())
CONTRIBUTION= [0.002, 1.404 , 1.613 , 1.617 , 1.884 , 2.314 , 2.327]
print(CONTRIBUTION)

def get_objective_function(x,dev="sum"):
    obj_function = []
    for agent_index in range(0,len(x)):
        for larger_agent_index in range(agent_index+1,len(x)):
            obj_function.append(abs(x[agent_index]/x[larger_agent_index]-CONTRIBUTION[agent_index]/CONTRIBUTION[larger_agent_index]))
    if dev == "sum":
        print ((obj_function))
        return np.sum(obj_function)
    elif dev == "max":
        print (obj_function)
        return np.max(obj_function)

def get_stable_bounds():
    bounds = []
    grand_coalition_value = np.sum(CONTRIBUTION)+0.01
    coalition_bound = []
    for x in range(len(AGENTS)):
        coalition = [str(agent) for agent in AGENTS[:x + 1]]
        coalition_bound.append(','.join(coalition))
    stable_bound = [(char_functions[coalition],grand_coalition_value) for coalition in coalition_bound]
    return stable_bound

def get_proportional_solution():
    solution = []
    grand_coalition_value = np.sum(CONTRIBUTION)
    for x in CONTRIBUTION:
        solution.append(x*grand_coalition_value/CONTRIBUTION[-1])
    return solution

const = ()

xinit = get_proportional_solution()
bnds = get_stable_bounds()
print(bnds)
res = minimize(get_objective_function, x0=xinit, bounds=bnds, constraints=const,tol=1e-5)
print (res)
print ("stable bounds:" + str(bnds))
print ("proportional solution: " + str(xinit))
print ("shapley: " + str(CONTRIBUTION))
print(char_functions)
stable_bound = [x[0] for x in bnds]
proportional_solution = xinit
num_players = len(stable_bound)

prop = plt.bar(range(1, num_players+1), proportional_solution, color="pink",alpha=0.5)

# plot stable bound
for index,bound in enumerate(stable_bound):
    start = (prop.patches[index].xy[0])
    stble = plt.hlines(bound, xmin=start,xmax=start+0.8, color="blue",linestyles="dotted",alpha=0.7)

# plot 'best' solution based on proportionality deviation
for index,best in enumerate(res.x):
    start = (prop.patches[index].xy[0])
    if index == 2:
        best = best + 0.7
    best = plt.hlines(best, xmin=start,xmax=start+0.8, color="green",linestyles="-",alpha=0.3)
best.set_label("'best' outcome based on minimisation")

prop.set_label("Proportional outcome")
stble.set_label("Each player's lower bound for stable outcome")
plt.ylabel("Mutual Information")
plt.xlabel("Players")
plt.title(r"Outcome based on $v_2(.)$")
plt.legend()
plt.show()
