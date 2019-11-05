import matplotlib.pyplot as plt
import numpy as np
np.random.seed(15)
def fisher_information(x_input, std_output):
    '''

    :param x_input: list of x input values
    :param std_output: output noise level
    :return: total Fisher Infomation
    '''

    return np.sum(np.square(x_input) / std_output)


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

    characteristic_function_value = fisher_information(x_input, std_output)

    return x_input, y_output, std_output

AGENT_X_LOCATION = [0,-2.4,2.15,2.15,5.1,5.5,4]
AGENT_X_NOISE = [0,0.7,0.6,0.6,1.2,2,6]
AGENT_Y_NOISE = [1,4,2,2,3.5,1,2]
AGENT_DATAPOINTS = [1,100,100,100,100,97,160]
AGENT_COLOR = "bgrcmykw"
PARAMETER = 3

AGENT_X_LOCATION = [0,-2.4,2.8,2.8,5.3,5.3,4.5]
AGENT_X_NOISE = [0,0.7,0.6,0.6,1.2,2,6]
AGENT_Y_NOISE = [1,4,1.8,1.8,3.5,1,2]
AGENT_DATAPOINTS = [1,100,100,100,120,97,110]
AGENT_COLOR = "bgrcmykw"
PARAMETER = 3
CONTRIBUTION = [0,150,450,450,1000,3000,3800]

# create data
agent_data = []
for mean_location, std_location, std_output, n_points in zip(AGENT_X_LOCATION, AGENT_X_NOISE, AGENT_Y_NOISE, AGENT_DATAPOINTS):
    agent_data.append(create_data_linear_model(mean_location,std_location,std_output,PARAMETER, n_points))

player_count = 1
# calculate char
for data, color in zip(agent_data,AGENT_COLOR):
    x_input = data[0]
    y_output = data[1]
    std_output = data[2]
    char = fisher_information(x_input, std_output)
    print (char)
    plt.scatter(x_input, y_output,c=color,alpha=0.6,label=r'$' + "Player " + str(player_count) + ', ' + 'X \sim \mathcal{N}(' + str(AGENT_X_LOCATION[player_count-1]) + ',' + str(round(AGENT_X_NOISE[player_count-1]**2,1)) + '),' + '\delta='+str(AGENT_Y_NOISE[player_count-1]) + r'$')
    player_count+=1
plt.legend()
plt.ylabel(r'$y=\beta X + \epsilon$')
plt.xlabel(r'$X$')
plt.title(r'Data held by different players')
plt.show()

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
plt.rc('font', family='serif')

CONTRIBUTION = [0,150,250,250,750,3000,5000]
CONTRIBUTION = [0,102.5,116,224.2,789.5,1689,1856]
CONTRIBUTION = [2801.1328315852093,
1369.447438861896,
317.6544971155584]
CONTRIBUTION.reverse()

def get_objective_function(x,dev="sum"):
    obj_function = []
    for agent_index in range(0,len(x)):
        for larger_agent_index in range(agent_index+1,len(x)):
            obj_function.append(abs(x[agent_index]/x[larger_agent_index]-CONTRIBUTION[agent_index]/CONTRIBUTION[larger_agent_index]))
            #obj_function.append(abs(
            #    x[larger_agent_index] / x[agent_index] - CONTRIBUTION[larger_agent_index] / CONTRIBUTION[agent_index]))
    if dev == "sum":
        print ((obj_function))
        return np.sum(obj_function)
    elif dev == "max":
        print (obj_function)
        return np.max(obj_function)

def get_stable_bounds():
    bounds = []
    grand_coalition_value = np.sum(CONTRIBUTION)

    for agent_contribution_index in range(0, len(CONTRIBUTION)):
        agent_contribution = CONTRIBUTION[agent_contribution_index]
        agent_lower_bound = 0
        for x in CONTRIBUTION:
            if x <= agent_contribution:
                agent_lower_bound+=x
        bounds.append((agent_lower_bound, grand_coalition_value))
    return bounds

def get_proportional_solution():
    solution = []
    grand_coalition_value = np.sum(CONTRIBUTION)
    for x in CONTRIBUTION:
        solution.append(x*grand_coalition_value/CONTRIBUTION[-1])
    return solution

obj_fun_sum_proportional_diff = lambda x: (abs(x[0]/x[1]-0) + abs(x[0]/x[2]-0) + abs(x[0]/x[3]-0) + abs(x[0]/x[4]-0) + abs(x[1]/x[2]-1/2.5) + abs(x[1]/x[3]-1/2.5) + abs(x[1]/x[4]-1/9) + abs(x[2]/x[3]-1) + abs(x[2]/x[4]-2.5/9) + abs(x[3]/x[4]-2.5/9))
obj_fun_max_proportional_diff = lambda x: abs(max(abs(x[0]/x[1]-0) , abs(x[0]/x[2]-0), abs(x[0]/x[3]-0), abs(x[0]/x[4]-0), abs(x[1]/x[2]-1/2.5), abs(x[1]/x[3]-1/2.5), abs(x[1]/x[4]-1/9), abs(x[2]/x[3]-1), abs(x[2]/x[4]-2.5/9), abs(x[3]/x[4]-2.5/9)))
# 0 <= x_bad <= 15 v(0) = 0
# 1 <= x_0 <= 15 v(0u1) = 1
# 6 <= x_1 <= 15 v(0u1u2u3)=6
# 6 <= x_2 <= 15 v(0u1u2u3)=6
# 15 <= x_3 <= 15 v(0u1u2u3u4)=15
const = ()

xinit = get_proportional_solution()
bnds = get_stable_bounds()

res = minimize(get_objective_function, x0=xinit, bounds=bnds, constraints=const,tol=1e-10)
print (res)
print ("stable bounds:" + str(bnds))
print ("proportional solution: " + str(xinit))

stable_bound = [x[0] for x in bnds]
proportional_solution = xinit
num_players = len(stable_bound)

prop = plt.bar(range(1, num_players+1), proportional_solution, color="pink",alpha=0.5)

# plot stable bound
for index,bound in enumerate(stable_bound):
    start = (prop.patches[index].xy[0])
    stble = plt.hlines(bound, xmin=start,xmax=start+0.8, color="blue",linestyles="dotted",alpha=0.7)

# plot 'best' solution based on proportionality deviation
#for index,best in enumerate(res.x):
#    start = (prop.patches[index].xy[0])
#    best = plt.hlines(best, xmin=start,xmax=start+0.8, color="green",linestyles="-",alpha=0.3)
#best.set_label("'best' solution based on minimisation")
prop.set_label("Proportional outcome")
stble.set_label("Each player's lower bound for stable outcome")
plt.ylabel("Fisher Information")
plt.xlabel("Players")
plt.title(r"Outcome based on $v_1(.)$")
plt.legend()
plt.show()


