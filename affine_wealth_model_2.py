import random
import numpy
import os
import scipy.stats
from random import randint
import math
import importlib
import graph_maker
from graph_maker import *
importlib.reload(graph_maker)
import helper_functions
from helper_functions import *
importlib.reload(graph_maker)

sum_wealth_advantages = 0
sum_skill_advantages = 0

class Node():
    def __init__(self,start_money,node_type,skill):
        self.money = start_money
        self.node_type = node_type
        self.skill = skill

    def __lt__(self, other):
        return (self.skill) < (other.skill)
    def __gt__(self, other):
        return (self.skill) > (other.skill)
    def __le__(self, other):
        return (self.skill) <= (other.skill)
    def __ge__(self, other):
        return (self.skill) <= (other.skill)
    def __eq__(self, other):
        return (self.skill) == (other.skill)

def simulate_segragation(N, phi, inital_money_per_node, omega, max_number_of_rounds, chi, zeta, kappa, tau,
                         folder_to_save_graphs, step_size):
    W = inital_money_per_node * N
    average_wealth = W / N
    
    #default value of delta t
    delta_t = 0.25
    
#     #multiplying upper bound by 18000 seems to be an OK starting value
#     delta_t = 0
#     if ((zeta * N) + tau) != 0:
#         delta_t = 200000 * (1/(((zeta * N) + tau))**2)
#     else:
#         #arbitrary default value
#         delta_t = 0.25
    
    successful_simulation = False
    people = []
    
    while successful_simulation == False:
        remove_contents_of_folder(folder_to_save_graphs)
        successful_simulation, people, overshot_bias, lorenz_curve = simulate_segregation_with_delta_t(N, phi, 
                                                                                                       inital_money_per_node, 
                                                                                                       omega, 
                                                                                                       max_number_of_rounds,
                                                                                                       chi, zeta, kappa, tau, 
                                                                                                       delta_t, average_wealth, 
                                                                                                       folder_to_save_graphs,
                                                                                                       step_size)
        if overshot_bias != 0:
            print("old delta t: " + str(delta_t))
            delta_t = (delta_t/(overshot_bias**2)) * 0.8
            print("restarting simulation with new delta t: " + str(delta_t))
            print("********************************************************")
            print("********************************************************")
            print("")

    return people, lorenz_curve

def get_majorities(people):
    majorities = []
    for person in people:
        if person.node_type == "majority":
            majorities.append(person)
    return majorities

def get_minorities(people):
    minorities = []
    for person in people:
        if person.node_type == "minority":
            minorities.append(person)
    return minorities

def simulate_segregation_with_delta_t(N, phi, inital_money_per_node, omega, max_number_of_rounds, chi, zeta, kappa, tau, 
                                      delta_t, average_wealth, folder_to_save_graphs, step_size):
    
    majorities = []
    minorities = []

    for i in range(round(phi*N)):
        majorities.append(Node(inital_money_per_node,"majority",get_skill_level()))

    for i in range(N - round(phi*N)):
        minorities.append(Node(inital_money_per_node,"minority",get_skill_level()))

    people = majorities + minorities
    indices = list(range(N))
    
    gini_coefficients = []
    steve_coefficients = []
    pearson_correlation_coefficients = []
    
    gini_converged = False
    steve_converged = False
    
    for current_round in range(max_number_of_rounds):
        if current_round % step_size == 0:
            #plot_lorenz_curve(get_minorities(people), folder_to_save_graphs, False, current_round, " of Minority Agents ")
            #plot_lorenz_curve(get_majorities(people), folder_to_save_graphs, False, current_round, " of Majority Agents ")
            #print("wealth held by majority nodes: ")
            #print(calculate_majority_wealth_percentage(people, N, inital_money_per_node))
            #print("phi: " + str(phi))
            #gini_coefficients.append(calculate_gini_coefficient(people, False)[0])
            gini_coefficients.append(plot_lorenz_curve(people, folder_to_save_graphs, False, current_round)[0])
            steve_coefficients.append(plot_lorenz_curve(people, folder_to_save_graphs, True, current_round)[0])
            pearson_correlation_coefficients.append(plot_skill_vs_net_wealth(people, folder_to_save_graphs, current_round))
            plot_coefficients(gini_coefficients, steve_coefficients, pearson_correlation_coefficients, step_size, current_round,
                              folder_to_save_graphs, tau)
            curr_coefficient = len(gini_coefficients) - 1
            print("")
            print("number of transactions: " + str(current_round))
            print("current coefficient values:")
            print("Gini coefficient:                                             " + str(gini_coefficients[curr_coefficient]))
            print("Steve coefficient:                                            " + str(steve_coefficients[curr_coefficient]))
            print("Pearson r correlation coefficient for wealth vs. skill level: " + str(pearson_correlation_coefficients[curr_coefficient]))
            
            gini_converged = check_if_gini_converged(gini_coefficients)
            steve_converged = check_if_steve_converged(steve_coefficients, tau)
        
        if gini_converged and steve_converged:
            print("converged after " + str(current_round) + " transactions")
            break
        
        if current_round / step_size > 100:
            print("Did not converge. Stopped early.")
            break
        
        two_people_indices = random.sample(indices, 2)
        
#         #Algorithm 2.2
#         if people[two_people_indices[0]].node_type != people[two_people_indices[1]].node_type:
#             if random.uniform(0, 1) > omega:
#                 while people[two_people_indices[0]].node_type != people[two_people_indices[1]].node_type:
#                     two_people_indices = random.sample(indices, 2)
        
        #Algorithm 2.3
        if people[two_people_indices[0]].node_type != people[two_people_indices[1]].node_type:
            if random.uniform(0, 1) > omega:
                while people[two_people_indices[0]].node_type != people[two_people_indices[1]].node_type:
                    two_people_indices[1] = random_int_exclude_int(N, two_people_indices[0])
        
        success, overshot_bias = transact(two_people_indices[0], two_people_indices[1], people, chi, zeta, average_wealth, 
                                          kappa, tau, delta_t)
            
        if not success:
            return False, people, overshot_bias, 0
        
        if current_round == (max_number_of_rounds - 1):
            print("Reached maximum number of transactions!")
    
    _, lorenz_curve = plot_lorenz_curve(people, folder_to_save_graphs, False)
    plot_lorenz_curve(people, folder_to_save_graphs, True)
    plot_skill_vs_net_wealth(people, folder_to_save_graphs)
    
    return True, people, 0, lorenz_curve

def transact(person_1_index, person_2_index, people, chi, zeta, average_wealth, kappa, tau, delta_t):
    
    for person in people:
        person.money = person.money + chi * delta_t * (average_wealth - person.money)

    difference_in_money = people[person_1_index].money - people[person_2_index].money
    difference_in_skill = people[person_1_index].skill - people[person_2_index].skill
    
    mean_coin_flip_based_on_wealth = zeta * math.sqrt(delta_t) * difference_in_money/average_wealth
    mean_coin_flip_based_on_skill = tau * math.sqrt(delta_t) * difference_in_skill/100
    
    global sum_wealth_advantages
    global sum_skill_advantages
    sum_wealth_advantages = sum_wealth_advantages + abs(mean_coin_flip_based_on_wealth)
    sum_skill_advantages = sum_skill_advantages + abs(mean_coin_flip_based_on_skill)

    mean_coin_flip = mean_coin_flip_based_on_wealth + mean_coin_flip_based_on_skill
    
    if mean_coin_flip > 1 or mean_coin_flip < -1:
        print("")
        print("********************************************************")
        print("********************************************************")
        print("Delta t is too large!!!!! Got a mean coin flip of: " + str(mean_coin_flip))
        print("Reducing values of all parameters is recommended.")
        return False, mean_coin_flip
    
    loan = kappa * average_wealth
    
    people[person_1_index].money = people[person_1_index].money + loan
    people[person_2_index].money = people[person_2_index].money + loan
    
    transaction_cost = math.sqrt(delta_t) * min(people[person_1_index].money, people[person_2_index].money)
    
    #flip 1 with probability threshold_for_coin_flip and flip -1 with probability 1 - threshold_for_coin_flip
    #threshold for coin flip is the same as the negative of the mean coin flip
    if random.uniform(-1, 1) < mean_coin_flip:
        people[person_1_index].money = people[person_1_index].money + transaction_cost
        people[person_2_index].money = people[person_2_index].money - transaction_cost
    else:
        people[person_1_index].money = people[person_1_index].money - transaction_cost
        people[person_2_index].money = people[person_2_index].money + transaction_cost
        
    people[person_1_index].money = people[person_1_index].money - loan
    people[person_2_index].money = people[person_2_index].money - loan
    
    return True, 0

def run_experiments(num_experiments, folder_to_save_graphs, N, phi, inital_money_per_node, omega, max_number_of_rounds, chi, 
                    zeta, kappa, tau, fig_title, fig_legend, graph_colours, file_to_save, x_label, y_label, is_lorenz):
    lorenz_curves = []
    wealthes_per_majority_node = []
    gini_coefficients = []
    for i in range(num_experiments):
        folder_to_save_graphs_for_experiment = os.path.join(folder_to_save_graphs, "experiment_" + str(i))
        try_make_directory(folder_to_save_graphs_for_experiment)
        people, lorenz_curve = simulate_segragation(N, phi, inital_money_per_node, omega, max_number_of_rounds, 
                                                                      chi, zeta, kappa, tau, folder_to_save_graphs_for_experiment)
        wealth_per_majority_node = calculate_majority_wealth_percentage(people, N, inital_money_per_node)

        wealthes_per_majority_node.append(wealth_per_majority_node)
        gini_coefficients.append(calculate_gini_coefficient(people)[0])
        lorenz_curves.append(lorenz_curve)
    
    file_to_save = os.path.join(folder_to_save_graphs, "test")
    data = lorenz_curves
    x = numpy.arange(0,N+1)
    is_lorenz = True
    mean_lorenz_curve, c_i_lorenz_curve = generate_confidence_interval_graph(fig_title, fig_legend, graph_colours, file_to_save, 
                                                                             x_label, y_label, x, data, is_lorenz)
    
    mean_majority, c_i_majority = get_confidence_interval_and_mean(wealthes_per_majority_node)
    mean_gini_coeffic, c_i_gini_coeffic = get_confidence_interval_and_mean(gini_coefficients)
    
    
    return mean_lorenz_curve, c_i_lorenz_curve, mean_majority, c_i_majority, mean_gini_coeffic, c_i_gini_coeffic



#num_nodes = 1000
N = 1000
#fraction of majority_nodes
phi = 1.0
#average net worth per person US, in millions
inital_money_per_node = 0.432365
#weak link modifier
omega = 1.0
max_number_of_rounds = 2000000
#taxation
chi = 0.000025
#wealth advantage
zeta = 0.0000001
#loan ability
kappa = 0.0000020
#skill advantage
tau = 0.4
#creates a new set of graphs after every amount of 'step_size' transactions
step_size = 10000

date_time_str = get_date_and_time()
param_folder_name = convert_variables_to_path_string(phi, inital_money_per_node, omega, max_number_of_rounds, 
                                                     chi, zeta, kappa, tau)
folder_to_save_graphs = os.path.join(os.getcwd(),"graphs", param_folder_name,date_time_str)
try_make_directory(folder_to_save_graphs)

simulate_segragation(N, phi, inital_money_per_node, omega, max_number_of_rounds, 
                                                                      chi, zeta, kappa, tau, folder_to_save_graphs, step_size)


#print(sum_wealth_advantages)
#print(sum_skill_advantages)
#print(sum_wealth_advantages/700000)
#print(sum_skill_advantages/700000)


'''
#num_nodes = 1000
N = 1000
#fraction_of_majority_nodes
phi = 0.6
#average net worth per person US, in millions
inital_money_per_node = 0.432365
#weak_link_modifier
omega = 0.8
max_number_of_rounds = 20000
#taxation = 0.001
chi = 0.001
#wealth advantage
zeta = 0.2
#loan_ability
kappa = 0.058
#skill_advantage
tau = 0.0

date_time_str = get_date_and_time()
param_folder_name = convert_variables_to_path_string(phi, inital_money_per_node, omega, max_number_of_rounds, 
                                                     chi, zeta, kappa, tau)
folder_to_save_graphs = os.path.join(os.getcwd(),"graphs", param_folder_name,date_time_str)
try_make_directory(folder_to_save_graphs)

num_experiments = 2
fig_title = "Confidence Interval of Lorenz Curve from Simulation, φ = 0.6, ω = 0.8"
fig_legend = []
graph_colours = []
file_to_save = "blahhhh"
x_label = "wee"
y_label = "yahoo"
is_lorenz = True
m_lorenz_curve, c_i_lorenz_curve, m_maj, c_i_maj, m_gini_coeffic, c_i_gini_coeffic = run_experiments(num_experiments, 
                                                                                                     folder_to_save_graphs, N, 
                                                                                                     phi, inital_money_per_node, omega, max_number_of_rounds, chi, 
                                                                                                     zeta, kappa, tau,
                                                                                                     fig_title, fig_legend,
                                                                                                     graph_colours,
                                                                                                     file_to_save,
                                                                                                     x_label, y_label,
                                                                                                     is_lorenz)
    


#fraction_of_majority_nodes
phi = 1.0
#weak_link_modifier
omega = 1.0
fig_title = "Confidence Interval of Lorenz Curve from Simulation, φ = 1.0, ω = 1.0"

date_time_str = get_date_and_time()
param_folder_name = convert_variables_to_path_string(phi, inital_money_per_node, omega, max_number_of_rounds, 
                                                     chi, zeta, kappa, tau)
folder_to_save_graphs = os.path.join(os.getcwd(),"graphs", param_folder_name,date_time_str)
try_make_directory(folder_to_save_graphs)

m_lorenz_curve_2, c_i_lorenz_curve_2, m_maj_2, c_i_maj_2, m_gini_coeffic_2, c_i_gini_coeffic_2 = run_experiments(num_experiments, 
                                                                                                     folder_to_save_graphs, N, 
                                                                                                     phi, inital_money_per_node, omega, max_number_of_rounds, chi, 
                                                                                                     zeta, kappa, tau,
                                                                                                     fig_title, fig_legend,
                                                                                                     graph_colours,
                                                                                                     file_to_save,
                                                                                                     x_label, y_label,
                                                                                                     is_lorenz)

print("with segregation")
print("percentage of wealth held by majority:")
print(m_maj)
print(c_i_maj)
print("gini coefficient:")
print(m_gini_coeffic)
print(c_i_gini_coeffic)

print("without segregation")
print("percentage of wealth held by majority:")
print(m_maj_2)
print(c_i_maj_2)
print("gini coefficient:")
print(m_gini_coeffic_2)
print(c_i_gini_coeffic_2)


print("are the confidence intervals different at all?")
print(is_difference_in_confidence_intervals(m_lorenz_curve, c_i_lorenz_curve, m_lorenz_curve_2, c_i_lorenz_curve_2))
print(is_difference_in_confidence_intervals(m_maj, c_i_maj, m_maj_2, c_i_maj_2))
print(is_difference_in_confidence_intervals(m_gini_coeffic, c_i_gini_coeffic, m_gini_coeffic_2, c_i_gini_coeffic_2))
'''