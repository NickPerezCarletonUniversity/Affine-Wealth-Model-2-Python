import random
import numpy
import os
import scipy.stats
from random import randint
from datetime import datetime
import math
from matplotlib import pyplot as plt
#from scipy.stats import binom

#some matplotlib settings code taken from here:
#https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot
import matplotlib.pyplot as plt
#change default font sizes of matplotlib plots
SMALL_SIZE = 16
MEDIUM_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the figure title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the x axis title

def try_make_directory(directory_name):
    try:
        os.makedirs(directory_name)    
    except FileExistsError:
        return False
    return True
    
def get_confidence_interval_and_mean(data):
    #ax.plot(x,mean)
    #ax.fill_between(x, (mean+ci), (mean-ci), alpha=.1)
    data = numpy.array(data)
    print("data shape: ")
    print(data.shape)
    confidence_interval = 1.96 * numpy.std(data, axis=0)/math.sqrt(data.shape[0])
    mean = numpy.mean(data, axis=0)
    return mean, confidence_interval
    
def generate_confidence_interval_graph(fig_title, fig_legend, graph_colours, file_to_save, x_label, y_label, x, data, is_lorenz):
    mean, confidence_interval = get_confidence_interval_and_mean(data)
    
    fig, ax = plt.subplots(figsize=[6,6])
    
    N = numpy.array(data).shape[1] - 1
    plt.plot(x,mean, color='y', label="Income Inequality from Simulation")
    plt.plot([0,N], [0,1], color='k', label="No Income Inequality")
    
    plt.fill_between(x, (mean+confidence_interval), (mean-confidence_interval), alpha=.1)
    
    if is_lorenz:
        plt.xlim(0,N)
        plt.ylim(numpy.amin(mean-confidence_interval),1)
        positions = (0, N * 0.25, N * 0.5, N * 0.75, N)
        labels = ("0%", "25%", "50%", "75%", "100%")
        plt.xticks(positions, labels)
        positions = (0, 0.25, 0.5, 0.75, 1)
        plt.yticks(positions, labels)
        ax.set_xlabel("Poorest Percentage of People")
        plt.title(fig_title, y=1.04)
        ax.set_ylabel("Percentage of Total Wealth")
        plt.legend(bbox_to_anchor = (1.05, 0.6))
    
    plt.savefig(file_to_save,pil_kwargs={"quality":95}, dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close('all')
    return mean, confidence_interval

def plot_coefficients(gini_coefficients, steve_coefficients, pearson_correlation_coefficients,
                      rounds_per_entry, rounds, folder_to_save_graphs, tau):
    gini_coefficients = numpy.array(gini_coefficients)
    steve_coefficients = numpy.array(steve_coefficients)
    pearson_correlation_coefficients = numpy.array(pearson_correlation_coefficients)

    rounds = numpy.array(list(range(0,rounds+1,rounds_per_entry)))
    
    
    fig, ax = plt.subplots(figsize=[6,6])
    
    if tau != 0:
        plt.plot(rounds, gini_coefficients, color='C1', label="Gini Coefficients")
        plt.plot(rounds, steve_coefficients, color='C2', label="Steve Coefficients")
        plt.plot(rounds, pearson_correlation_coefficients, color='C3', label="r for Wealth vs. Skill Level")
        plt.ylabel('Coefficients')
        plt.legend(bbox_to_anchor = (1.05, 0.6))
        plt.title("Coefficients vs. Number of Transactions", y=1.04)
    else:
        plt.plot(rounds, gini_coefficients, color='C1')
        plt.ylabel('Gini Coefficient')
        plt.title("Gini Coefficient vs. Number of Transactions", y=1.04)

    plt.xlabel('Number of Transactions')
    file_to_save = "coefficients"
    plt.savefig(os.path.join(folder_to_save_graphs, file_to_save),pil_kwargs={"quality":95}, dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close('all')

def get_date_and_time():
    return datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

def plot_lorenz_curve(people, folder_to_save_graphs, sort_by_skill=False, number_of_transactions=-1, extra_title = " "):
    gini_coefficient, total_people_money_list = calculate_gini_coefficient(people, sort_by_skill)
    gini_coefficient_str = str(round(gini_coefficient,2))

    total_people_money_list = numpy.array(total_people_money_list)

    lorenz_curve = total_people_money_list.cumsum() / total_people_money_list.sum()

    lorenz_curve = numpy.insert(lorenz_curve, 0, 0)
    lorenz_curve[0], lorenz_curve[-1]
    
    N = len(people)

    fig, ax = plt.subplots(figsize=[6,6])
    ax.plot(lorenz_curve, color='y', label="Income Inequality from Simulation")
    ax.plot([0,N], [0,1], color='k', label="No Income Inequality")


    plt.xlim(0,N)
    plt.ylim(numpy.amin(lorenz_curve),1)
    positions = (0, N * 0.25, N * 0.5, N * 0.75, N)
    labels = ("0%", "25%", "50%", "75%", "100%")
    plt.xticks(positions, labels)
    positions = (0, 0.25, 0.5, 0.75, 1)
    plt.yticks(positions, labels)
    
    if not sort_by_skill:
        ax.set_xlabel("Poorest Percentage of People")
        plt.title("Lorenz Curve" + extra_title + "from Simulation, Gini Coefficient = " + gini_coefficient_str, y=1.04)
    else:
        ax.set_xlabel("Lowest Skill Percentage of People")
        plt.title("Steve Curve from Simulation, Steve Coefficient = " + gini_coefficient_str, y=1.04)
    
    ax.set_ylabel("Percentage of Total Wealth")
    plt.legend(bbox_to_anchor = (1.05, 0.6))
    
    file_to_save = ""
    if number_of_transactions > -1:
        if sort_by_skill:
            file_to_save = "steve_curve"
            folder_to_save_graphs = os.path.join(folder_to_save_graphs, "steve_curves")
            try_make_directory(folder_to_save_graphs)
        else:
            file_to_save = "lorenz_curve"
            folder_to_save_graphs = os.path.join(folder_to_save_graphs, "lorenz_curves")
            try_make_directory(folder_to_save_graphs)
        file_to_save = file_to_save + "_after_" + str(number_of_transactions) + "_transactions"
    else:
        if sort_by_skill:
            file_to_save = "steve_curve"
        else:
            file_to_save = "lorenz_curve"

    if extra_title != " ":
        file_to_save = file_to_save + extra_title.replace(" ", "_")
    
    plt.savefig(os.path.join(folder_to_save_graphs, file_to_save),pil_kwargs={"quality":95}, dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close('all')
    
    return gini_coefficient, lorenz_curve
    
def calculate_gini_coefficient(people, sort_by_skill=False):
    total_people_money_list = []
    
    people.sort()
    
    for i in people:
        total_people_money_list.append(i.money)
    
    if not sort_by_skill:
        total_people_money_list.sort()
    
    num_total_people = len(total_people_money_list)

    total_people_money_list = numpy.array(total_people_money_list)

    lorenz_curve = total_people_money_list.cumsum() / total_people_money_list.sum()
    
    total_area_under_straight_line = 0
    total_area_under_lorenz_curve = 0
    
    for i in range(num_total_people):
        total_area_under_straight_line = total_area_under_straight_line + (i+1)/num_total_people
        total_area_under_lorenz_curve = total_area_under_lorenz_curve + lorenz_curve[i]
    
    return (total_area_under_straight_line - total_area_under_lorenz_curve)/total_area_under_straight_line, total_people_money_list

def plot_skill_vs_net_wealth(people, folder_to_save_graphs, number_of_transactions = -1):
    x = []
    y = []
    
    for person in people:
        x.append(person.skill)
        y.append(person.money)

    x = numpy.array(x)
    y = numpy.array(y)
    s = 1
    
    m, b = numpy.polyfit(x, y, 1)
    
    
    r = 0
    r_str = "0"
    p_value = "0"
    if number_of_transactions != 0:
        r = scipy.stats.pearsonr(x, y)[0]
        r_str = str(round(scipy.stats.pearsonr(x, y)[0],3))
        p_value = str(round(scipy.stats.pearsonr(x, y)[1],4))
    
    fig, ax = plt.subplots(figsize=[6,6])
    ax.ticklabel_format(useOffset=False, style='plain')
    plt.scatter(x, y, s, label="An Agent")
    plt.plot(x, m*x+b, label="Line of Best Fit: " + str(round(m,3)) + "x + " + str(round(b,3)))
    
    plt.title("Net Wealth vs. Skill Level from Simulation, r = " + r_str + ", p-value = " + p_value, y=1.04)
    
    plt.xlabel("Skill Level")
    plt.ylabel("Wealth in Millions")
    plt.legend(bbox_to_anchor = (1.05, 0.6))
    
    file_to_save = ""
         
    if number_of_transactions > -1:
        file_to_save = "scatter_plot_" + "_after_" + str(number_of_transactions) + "_transactions"
        folder_to_save_graphs = os.path.join(folder_to_save_graphs, "scatter_plots")
        try_make_directory(folder_to_save_graphs)
    else:
        file_to_save = "scatter_plot"
        
    plt.savefig(os.path.join(folder_to_save_graphs, file_to_save),pil_kwargs={"quality":95}, dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close('all')
    if math.isnan(r):
        return 0
    return r
    
def create_casino_game_graph(max_number_of_rounds):
    money_after_playing = [1.2, 0.8]

    required_wins_ratio = 1 - 1/(1-math.log(0.8)/math.log(1.2))
    probabilities_of_losing = []
    for i in range(max_number_of_rounds):
        prob = scipy.stats.binom.cdf(math.floor(required_wins_ratio * (i+1)),i+1, 0.5)
        probabilities_of_losing.append(prob)
        
    probabilities_of_losing = numpy.array(probabilities_of_losing)

    x = numpy.array(list(range(1,max_number_of_rounds + 1)))


    fig, ax = plt.subplots(figsize=[6,6])
    ax.plot(x, probabilities_of_losing, color='C4')

    plt.title('Probability of Losing a Casino Game vs. Number of Rounds Played', y=1.06)
    plt.xlabel('Number of Casino Game Rounds')
    plt.ylabel('Probability of Losing Money')
    file_to_save = "probability_of_losing_casino_game"
    plt.savefig(os.path.join(os.getcwd(),"graphs", file_to_save),pil_kwargs={"quality":95}, dpi=300, bbox_inches='tight')
    #plt.show()
    plt.close('all')
