'''
Created on Jun 7, 2019

@author: mrwan
'''
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt 
import copy
import time
from copy import deepcopy



def initialization(dimension, LB, UB):
    GRN = [None] * dimension
    sigma = [None] * dimension
    for i in range(dimension):
        n_connection = random.randint(LB, UB)
        temp = []
        temp2 = []
        dict = []
        for j in range(n_connection):
            xx = random.randint(0, dimension - 1)
            try:
                reso = dict.index(xx)
            except ValueError:
                reso = -1
            while(reso != -1):
                xx = random.randint(0, dimension - 1)
                try:
                    reso = dict.index(xx)
                except ValueError:
                    reso = -1
            temp.append([xx, random.randint(-50, 50) / 400])
            temp2.append([xx, random.randint(0, 50) / 40000])
            dict.append(xx)
        GRN[i] = temp.copy()
        sigma[i] = temp2.copy()
    
    return GRN, sigma


def fitness_helper(d, ind):
    error = 0
    c2 = 0
    counter = 0 
    for trial in d:
        for i in range(len(trial) - 1):
            predict = trial[i].copy()
            for j in range(len(ind)):
                for effect in ind[j]:
                    predict[effect[0]] = trial[i][j] * effect[1] + predict[effect[0]]
            for x in range(len(predict)):
                error = error + abs(predict[x] - trial[i + 1][x])
                counter = counter + 1
                c2 += 1
    return error / c2, counter


def sort_helper(t):
    
    return t[0]


def fitness_cal(d, p):
    err = []
    indx = 0
    counter = 0
    for indi in p:
        error, ct = fitness_helper(d, indi)
        err.append([error, indx])
        counter = counter + ct
        indx += 1
    return err, counter

    
def cxUniform(ind1, ind2, ind3, ind4, indpb):
    size = min(len(ind1), len(ind2))
    for i in range(size):
        if random.random() < indpb:
            p1 = ind1[i]
            ind1[i] = ind2[i]
            ind2[i] = p1
            p2 = ind3[i]
            ind3[i] = ind4[i]
            ind4[i] = p2
    
    return ind1, ind2, ind3, ind4


def mutation(ind, sigma):
    reso = copy.deepcopy(ind)
    for i in range(len(reso)):
        for j in range(len(reso[i])):
            var1 = np.random.normal(0, sigma[i][j][1])
            reso[i][j][1] = reso[i][j][1] + var1
    
    return reso


def generation(dimension, dict):
    xx = random.randint(0, dimension - 1)
    try:
        reso = dict.index(xx)
    except ValueError:
        reso = -1
    while(reso != -1):
        xx = random.randint(0, dimension - 1)
        try:
            reso = dict.index(xx)
        except ValueError:
            reso = -1
    return xx


def shrink(ind, ind2):
    ans = []
    ans2 = []
    length = len(ind)
    if(length <= 1):
        return ans, ans2
    
    indx_to_delete = random.randint(0, length - 1)
    for i in range(length):
        if(i != indx_to_delete):
            ans.append(ind[i].copy())
            ans2.append(ind2[i].copy())
    return ans, ans2


def expand(ind, ind2, req):
    dict = []
    length = len(ind)
    for i in range(length):
        dict.append(ind[i][0])
    indx = generation(req, dict)
    ind.append([indx, random.randint(-50, 50) / 400])
    ind2.append([indx, random.randint(0, 50) / 40000])
    
    return ind, ind2
    
    
def mutation_edge(ind, ind2, req):
    dim = req[0]
    UB = req[2]
    
    for i in range(len(ind)):
        length = len(ind[i])
        num_to_delete = random.randint(0, length)
        for j in range(num_to_delete):
            ind[i], ind2[i] = shrink(ind[i], ind2[i])
        length = len(ind[i])
        num_to_add = random.randint(0, UB - length)
        for j in range(num_to_add):
            ind[i], ind2[i] = expand(ind[i], ind2[i], dim)
            
    return ind, ind2

        


def linear_ranking(p, s, fit, miu, pb = [0.2, 0.8]):
    
    length = len(p)
    prob = [0]*length
    fit_r = []
    intver = (pb[1]-pb[0])/length
    for i in range(length):
        prob[i] = pb[1]-i*intver
    sum_p = sum(prob)
    for i in range(length):
        prob[i] = prob[i]/sum_p
    indx_p = [0]*miu
    for i in range(miu):
        r = random.random()
        k = 0
        F = prob[0]
        while(F < r):
            k = k+1
            F = F + prob[k]
        indx_p[i] = k
#     print(indx_p)
    parents_r = []
    sigma_r = []
    for i in range(miu):
        fit_r.append(fit[indx_p[i]][0])
        parents_r.append(p[indx_p[i]])
        sigma_r.append(s[indx_p[i]])
#     for p in parents_r:
#         print(p)
    return parents_r, sigma_r, fit_r
def rouletteWheel(p, s, fit, miu):
    f_sum=0
    for i in range(len(fit)):
        f_sum = f_sum+fit[i][0]
    N = len(p) 
    var = random.uniform(0, f_sum/N)
    k = 0
    f_acc = 0
    p_r = []
    s_r = []
    f_r = []
    while(len(p_r) < miu):
        k = k + 1
        f_acc = f_acc+fit[k][0]
        while(f_acc > var):
            p_r.append(p[k].copy())
            s_r.append(s[k].copy())
            f_r.append(fit[k][0])
            var = var + f_sum/N
    return p_r, s_r, f_r
def es_evolve_sparse(data, miu=4, lamda=8, generation=1000, indpb=0.2, req=[], ed_pb=[1, 0.5], model_i = None):
    parents = [[] for _ in range(miu)]
    sigma = [[] for _ in range(miu)]
    children = []
    csigma = []
    population = []
    a = ed_pb[0]
    b = ed_pb[1]
    pop2 = []
    new_err = []
    ct = 0
    x_la = [0] * generation
    g_best = [0] * generation
    avrge = [0] * generation
    iter = 0
    x1, x2, x3 = req[0], req[1], req[2]
    # initialization
    if(model_i == None):
        for i in range(miu):
            parents[i], sigma[i] = initialization(x1, x2, x3)    
    else:
        parents[0], sigma[0] = copy.deepcopy(model_i['parents'][-1]), copy.deepcopy(model_i['sigma'][-1])
        for i in range(miu-1):
            parents[i+1], sigma[i+1] = initialization(x1, x2, x3)    
#     print('=============== initialization ===================')
#     for pp in parents:
#         print(pp)
    timecounter = 0
    muttime = 0
    xtime = 0
    emuttime = 0
    totaltime = 0
    pst = 0
    t1 = time.time()
    cpy1time = 0
    cpy2time = 0
    while(iter < generation):
        t2 = time.time()
        totaltime = t2-t1+totaltime
        t1 = time.time()
        if(iter%100  == 0):
            print('------------ generation', iter, '----------------')
            print('error time used:', timecounter)
            print('crossover time:', xtime)
            print('mutation time:', muttime)
            print('parent selection time:', pst)
            print('edge_mutation time:', emuttime)
            print('CROSSOVER and mutation copy time:', cpy1time)
            print('POPULATION time:', cpy2time)
            print('total time:', totaltime)
            print('added time:', timecounter+xtime+muttime+emuttime+pst+cpy1time+cpy2time)
            if(iter >= 100):
                print('average error:',avrge[iter-100])
        timecounter = 0
        xtime = 0
        muttime = 0
        emuttime = 0
        cpy1time = 0
        cpy2time = 0
        totaltime = 0
        pst = 0
        # cross-over and sigma mutation 
        
        for i in range(int(lamda / 2)):
#             st = time.time()
            p1index = random.randint(0, miu - 1)
            p2index = p1index
            if(iter == 0):
                while(p1index == p2index):
                    p2index = random.randint(0, miu - 1)
            else:
                loop_c =0
                while(new_err[p1index] == new_err[p2index] and loop_c < 3):
                    p2index = random.randint(0, miu - 1)
                    loop_c = loop_c+1
            p1 = parents[p1index]
            p2 = parents[p2index]
            s1 = sigma[p1index]
            s2 = sigma[p2index]
#             et = time.time()
#             print('parent selection time:', et-st)
            st = time.time()
            c1, c2, cs1, cs2 = cxUniform(copy.deepcopy(p1), copy.deepcopy(p2), copy.deepcopy(s1), copy.deepcopy(s2), indpb)
            et = time.time()
            xtime = xtime + et - st
#             print('cxover time:', et-st)
            # # edge mutation probability calculation 
            # # max-min normalization default pb = (0.8,0.2)
            k = (b - a) / generation
            pb = a + k * iter
            tempvar = random.random()
            if(tempvar < pb):
                st = time.time()               
                c1, cs1 = mutation_edge(c1, cs1, req)
                c2, cs2 = mutation_edge(c2, cs2, req)
                et = time.time()
                emuttime = et - st + emuttime
#                 print('edge-mutation time:', et-st)
            st = time.time()
            c1 = mutation(c1, cs1)
            c2 = mutation(c2, cs2)
            et = time.time()
            muttime = muttime + et - st
#             print('mutation time:', et-st)
            st = time.time()
            children.append(copy.deepcopy(c1))
            children.append(copy.deepcopy(c2))
            csigma.append(copy.deepcopy(cs1))
            csigma.append(copy.deepcopy(cs2))
            et = time.time()
            cpy1time = cpy1time+et-st
        # # fitness calculation 
        st = time.time()            
        for i in range(len(children)):
            population.append(copy.deepcopy(children[i]))
            pop2.append(copy.deepcopy(csigma[i]))
        for i in range(len(parents)):
            population.append(copy.deepcopy(parents[i]))
            pop2.append(copy.deepcopy(sigma[i]))
        et = time.time()
        cpy2time = cpy2time+et-st
#         st = time.time()    
        st = time.time()
        err, ct_t = fitness_cal(data, population)
        et = time.time()
        timecounter = timecounter + et - st
        
#         print('error calculation time:', et - st)
        ct = ct_t + ct
        err2 = sorted(err, key=sort_helper)
        new_err.clear()
        new_pop = []
        new_sigma = []
        for i in range(miu+lamda):
            tempindx = err2[i][1]
            new_pop.append(copy.deepcopy(population[tempindx]))
            new_sigma.append(copy.deepcopy(pop2[tempindx]))
        
#         st = time.time()
        parents, sigma, new_err = linear_ranking(new_pop, new_sigma, err, miu-1, pb = [1,2])
#         et = time.time()
#         pst = et-st+pst
        new_err.append(err2[0][0])
        parents.append(copy.deepcopy(new_pop[0]))
        sigma.append(copy.deepcopy(new_sigma[0]))
        avrge[iter] = sum(new_err) / miu
        x_la[iter] = ct
        g_best[iter] = new_err[-1]
        children.clear()
        csigma.clear()
        population.clear()
        pop2.clear()
        iter += 1
    
    model_r = {}
    model_r['parents'] = parents.copy()
    model_r['sigma'] = sigma.copy()
    return g_best, avrge, parents[0], x_la, model_r


############################################# 
# # loading section 
#############################################
file = open('data.pickle', 'rb')
data = pickle.load(file)
file.close

file = open('REQ.pickle', 'rb')
REQ = pickle.load(file)
file.close
    
print('Number of trial: ', len(data))
print('Number of time step: ', len(data[0]))
print('Number of genes: ', len(data[0][0]))
#############################################
# # demo section
#############################################
# es_sparse_g_best_a = []
# loopcounter = 10
# gener_counter = 3000
# model_pass = {}
# cur_b = 3000
# for j in range(loopcounter):
#         print('=====================', j, '=========================')
#         if(model_pass == {}):
#             start = time.time()
#             print('using blank model')
#             es_sparse_g_best, es_sparse_average, es_best_individual, x_var1, model_r = es_evolve_sparse(data, generation=gener_counter, req=[10,0,5])
#             end = time.time()
#             print('time:', end - start)
#         else:
#             print('the model used is:')
#             for pp in model_pass['parents']:
#                 print(pp)
#             print('========================================')
#             start = time.time()
#             es_sparse_g_best, es_sparse_average, es_best_individual, x_var1, model_r = es_evolve_sparse(data, generation=gener_counter, req=[10,0,5], model_i=model_pass)
#             end = time.time()
#             print('time:', end-start)
#         es_sparse_g_best_a.append(es_sparse_g_best[-1])
#         if(es_sparse_g_best[-1] < cur_b):
#             cur_b = es_sparse_g_best[-1]
#             model_pass = model_r.copy()
#          
# print(np.std(es_sparse_g_best_a))
# print(np.average(es_sparse_g_best_a))
# 
# file = open('x_4modelfeed.pickle', 'wb')
# pickle.dump(x_var1, file)
# file.close
# 
# file = open('es_sparse_g_best_a_4modelfeed.pickle', 'wb')
# pickle.dump(es_sparse_g_best_a, file)
# file.close
# #
# plt.xscale('log') 
# plt.plot(x_var1, es_sparse_g_best)
# plt.plot(x_var1, es_sparse_average)
# plt.show()
# 
# for pp in model_r['parents']:
#     print(pp)
#     
# print(es_sparse_g_best_a)

#####
## DEMO
####
# gener_counter = 62264
# gener_counter = 3000
# es_sparse_g_best = [0]*10
# for i in range(10):
#     print(i)
#     es_sparse_g_besttp, es_sparse_average, es_best_individual, x_var1, model_r = es_evolve_sparse(data, generation=gener_counter, req=REQ)
#     es_sparse_g_best[i] = es_sparse_g_besttp[-1]
#       
# print(es_sparse_g_best)
# print(np.average(es_sparse_g_best))
# print(np.std(es_sparse_g_best))
# # 
# file = open('es_DREAM_avg_nofeed.pickle', 'wb')
# pickle.dump(es_sparse_g_best, file)
# file.close
#     
# file = open('es_DREAM_nofeed_x_var1.pickle', 'wb')
# pickle.dump(es_sparse_g_best, file)
# file.close
#     
#   
# plt.show()


###############################
# 
# file = open('x_var1.pickle', 'wb')
# pickle.dump(x_var1, file)
# file.close
#  
# file = open('es_sparse_average.pickle5', 'wb')
# pickle.dump(es_sparse_average, file)
# file.close
#  
# file = open('es_sparse_g_best.pickle5', 'wb')
# pickle.dump(es_sparse_g_best, file)
# file.close
# 
# ###############
gener_counter = 5000
es_sparse_g_best, es_sparse_average, es_best_individual, x_var1, model_r = es_evolve_sparse(data, generation=gener_counter, req=REQ)
  
# file = open('xvar_continiously_traning_GRN.pickle', 'wb')
# pickle.dump(x_var1, file)
# file.close
#  
# file = open('gbest_continiously_traning_GRN.pickle', 'wb')
# pickle.dump(es_sparse_g_best, file)
# file.close

file = open('GRN_es.pickle', 'wb')
pickle.dump(es_best_individual, file)
file.close



for i in es_best_individual:
    print('-------------')
    print(i)
plt.plot(x_var1,es_sparse_g_best)
plt.show()
###########################

# x1, x2, x3 = REQ[0], REQ[1], REQ[2]
# miu = 4
# parents = [[] for _ in range(miu)]
# sigma = [[] for _ in range(miu)]
# 
# for i in range(miu):
#     parents[i], sigma[i] = initialization(x1, x2, x3)
# 
# for i in parents:
#     print('======parent=========')
#     for j in i:
#         print(j)
#             
# file = open('parents_4_5%.pickle', 'wb')
# pickle.dump(parents, file)
# file.close
#   
# file = open('sigma_4_5%.pickle5', 'wb')
# pickle.dump(sigma, file)
# file.close
#   

