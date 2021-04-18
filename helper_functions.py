import random
import numpy
import os
import scipy.stats
from random import randint
import math
import shutil

#some code taken from: https://stackoverflow.com/questions/185936/how-to-delete-the-contents-of-a-folder
def remove_contents_of_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
def is_array(n):
    return isinstance(n , numpy.ndarray)
            
def is_difference_in_confidence_intervals(mean_1, confidence_interval_1, mean_2, confidence_interval_2):
    if is_array(mean_1) and is_array(confidence_interval_1) and is_array(mean_2) and is_array(confidence_interval_2):
        for i in range(mean_1.shape[0]):
            if mean_1[i] + confidence_interval_1[i] < mean_2[i] - confidence_interval_2[i]:
                return True
            if mean_1[i] - confidence_interval_1[i] > mean_2[i] + confidence_interval_2[i]:
                return True
            
    if mean_1 + confidence_interval_1 < mean_2 - confidence_interval_2:
        return True
    if mean_1 - confidence_interval_1 > mean_2 + confidence_interval_2:
        return True
        
    return False

def calculate_majority_wealth_percentage(people, N, initial_money_per_node):
    total_wealth = initial_money_per_node * N
    
    total_people_money_list = []
    total_majority_wealth = 0
    for i in people:
        total_people_money_list.append(i.money)
        if i.node_type == "majority":
            total_majority_wealth = total_majority_wealth + i.money

    return total_majority_wealth / total_wealth
    
def random_int_exclude_int(num_ints, excluded_int):
    random_int = randint(0,num_ints-1)
    return random_int_exclude_int(num_ints,excluded_int) if random_int is excluded_int else random_int

def get_skill_level():
    skill_level = numpy.random.normal(100, 15)
    while skill_level < 50 or skill_level > 150:
        skill_level = numpy.random.normal(100, 15)
    return skill_level

def check_if_gini_converged(gini_coefficients):
    num_coefficients = len(gini_coefficients)
    if num_coefficients > 3:
        if gini_coefficients[num_coefficients-1] <= gini_coefficients[num_coefficients-4]:
            return True
    return False

def check_if_steve_converged(steve_coefficients, skill_advantage):
    if skill_advantage == 0:
        return True
    num_coefficients = len(steve_coefficients)
    if skill_advantage != 0:
        if num_coefficients > 3:
            if steve_coefficients[num_coefficients-1] <= steve_coefficients[num_coefficients-4]:
                return True
    return False

def float_to_path_string(float_to_convert):
    float_to_convert = str(float_to_convert)
    float_to_convert = float_to_convert.replace(".","_")
    return float_to_convert

def convert_variables_to_path_string(phi, inital_money_per_node, omega, max_number_of_rounds, kai, zeta, kappa, tau):
    folder_name = "phi_" + float_to_path_string(phi)
    folder_name = folder_name + "_omega_" + float_to_path_string(omega)
    folder_name = folder_name + "_kai_" + float_to_path_string(kai)
    folder_name = folder_name + "_zeta_" + float_to_path_string(zeta)
    folder_name = folder_name + "_kappa_" + float_to_path_string(kappa)
    folder_name = folder_name + "_tau_" + float_to_path_string(tau)
    return folder_name