import math
import random
import matplotlib.pyplot as plt
from dataset.get_hydraulic import load_Hydraulic
import warnings

from sklearn.svm import SVC
from skmultilearn.adapt import MLkNN
import numpy as np
warnings.filterwarnings('ignore')
from sklearn.metrics import hamming_loss, accuracy_score

def index_locator(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

def sort_by_values(list1, values):
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        if index_locator(min(values),values) in list1:
            sorted_list.append(index_locator(min(values),values))
        values[index_locator(min(values),values)] = math.inf
    return sorted_list
def crossover(parent1,parent2):
    crossover_point = random.randint(0, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2
def mutation(individual):
    mutated_individual = individual[:]
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:
            mutated_individual[i] = 1 - mutated_individual[i]  # Flip the bit
    return mutated_individual
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0,len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 9999999999999999
    distance[len(front) - 1] = 9999999999999999
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance
def non_dominated_sorting_algorithm(values1, values2):
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)
    i = 0
    while(front[i] != []):
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)
    del front[len(front)-1]
    return front
def select_dataset(ds_name,cmd=False):

    '''if(ds_name == "azure"):
        X_train, X_test, y_train, y_test, class_name = merge_dataset(already_loaded=True,cmd=cmd)#get_azure(period=24)
        #print(f"AZURE: X_train: {X_train.shape} | X_test: {X_test.shape} | Y_test: {y_test.shape}")

    if(ds_name == "AI4I*"):
        X_train, X_test, y_train, y_test, class_name = get_ai4i(with_machine_failure=False,cmd=cmd)  # Get dataset already splitted

    if (ds_name == "AI4I**"):
        X_train, X_test, y_train, y_test, class_name = get_ai4i( with_machine_failure=True,cmd=cmd)  # Get dataset already splitted
    '''
    if(ds_name == "hydraulic"):
        X_train, X_test, y_train, y_test, class_name = load_Hydraulic(encoding="binary",cmd=cmd)  # Get dataset already splitted

    return X_train, X_test, y_train, y_test, class_name

'''def SVM_train(X_train, y_train):
    clf = SVC(gamma='auto')
    y_train_reshaped = y_train.squeeze()
    print(X_train.shape)
    print(y_train_reshaped.shape)
    clf.fit(X_train, y_train_reshaped)
    return clf'''

def SVM_train(X_train, y_train):
    clf = MLkNN(k=16)
    clf = clf.fit(X_train,y_train)
    return clf

def objective1(X_train,y_train,X_test,y_test):
    clf = SVM_train(X_train, y_train)
    prediction = clf.predict(X_test)
    h1 = hamming_loss(y_test,prediction)
    return h1

def objective2(X_train,y_train,X_test,y_test):
    clf = SVM_train(X_train, y_train)
    prediction = clf.predict(X_test)
    accuracy = accuracy_score(y_true=y_test,y_pred=prediction)
    return accuracy


def nsga2(population, max_gen, data_x, data_y, X_test, y_test):
    gen_no = 0
    solution = [[random.randint(0, 1) for _ in range(data_x.shape[1])] for _ in range(population)]

    while (gen_no < max_gen):
        objective1_values = [
            objective1(
                data_x[:, np.array(solution[c]).nonzero()[0]],
                data_y,
                X_test[:, np.array(solution[c]).nonzero()[0]],
                y_test
            ) for c, column in enumerate(range(len(solution)))
        ]
        objective2_values = [
            objective2(
                data_x[:, np.array(solution[c]).nonzero()[0]],
                data_y,
                X_test[:, np.array(solution[c]).nonzero()[0]],
                y_test
            ) for c, column in enumerate(range(len(solution)))
        ]
        non_dominated_sorted_solution = non_dominated_sorting_algorithm(objective1_values[:], objective2_values[:])
        print('Best Front for Generation:', gen_no)
        for values in non_dominated_sorted_solution[0]:
            print(np.array(solution[values]).round(3), end=" ")
        print("\n")
        crowding_distance_values = []
        for i in range(0, len(non_dominated_sorted_solution)):
            crowding_distance_values.append(
                crowding_distance(objective1_values[:], objective2_values[:], non_dominated_sorted_solution[i][:]))
        solution2 = solution[:]

        while (len(solution2) != 2 * population):
            a1 = random.randint(0, population - 1)
            b1 = random.randint(0, population - 1)
            solution2.append(crossover(solution[a1], solution[b1]))
        objective1_values2 = [
            objective1(
                data_x[:, np.array(solution[c]).nonzero()[0]],
                data_y,
                X_test[:, np.array(solution[c]).nonzero()[0]],
                y_test
            ) for c, column in enumerate(range(2 * len(solution)))
        ]
        objective2_values2 = [
            objective2(
                data_x[:, np.array(solution[c]).nonzero()[0]],
                data_y,
                X_test[:, np.array(solution[c]).nonzero()[0]],
                y_test
            ) for c, column in enumerate(range(2 * len(solution)))
        ]
        non_dominated_sorted_solution2 = non_dominated_sorting_algorithm(objective1_values2[:], objective2_values2[:])
        crowding_distance_values2 = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(
                crowding_distance(objective1_values2[:], objective2_values2[:], non_dominated_sorted_solution2[i][:]))
        new_solution = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [
                index_locator(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in
                range(0, len(non_dominated_sorted_solution2[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in
                     range(0, len(non_dominated_sorted_solution2[i]))]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if (len(new_solution) == population):
                    break
            if (len(new_solution) == population):
                break
        solution = [solution2[i] for i in new_solution]
        gen_no = gen_no + 1
    return [objective1_values, objective2_values]

def non_dominating_curve_plotter(objective1_values, objective2_values):
    plt.figure(figsize=(15,8))
    objective1 = [i * -1 for i in objective1_values]
    objective2 = [j * -1 for j in objective2_values]
    plt.xlabel('Objective Function 1', fontsize=15)
    plt.ylabel('Objective Function 2', fontsize=15)
    plt.scatter(objective1, objective2, c='red', s=25)

population = 25
max_gen = 501
min_value= -100
max_value= 100
mutation_rate = 0.3

ds_name = ["hydraulic"]#["hydraulic","AI4I*","AI4I**"]#"azure"]
cmd = True # when called from root project folder
for ds in ds_name:
    print(f"Dataset = {ds}")
    X_train, X_test, y_train, y_test, class_name = select_dataset(ds_name=ds,cmd=True)
print(X_train.shape)
print(y_train.shape)

objective1_values, objective2_values = nsga2(population,max_gen,X_train, y_train, X_test, y_test)

non_dominating_curve_plotter(objective1_values, objective2_values)