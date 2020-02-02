'''
Created on May 21, 2019

@author: mrwan
'''
import random
import pickle
from numpy.random.mtrand import randint


## weight matrix as x
## input gene expression as y 
def matmul(x,y):
    result = []
    for row in x:
        sum = 0
        for i in range(len(row)):
            sum = sum + row[i]*y[i]
        result.append(sum)
    return result 

def generator(GRN, n = -1, format = 'full', E = -1, magnitude =True, T = 3, trial = 5, UB = 50, IGEV = 3, MaxNoise = 1):
    n = len(GRN)
    final = []
    geneExpression = []
    GRN_C=[]
    output = []

        
    if(format == 'full'): 
        if(E > 0):
            # smallest significance specified 
            for gene in GRN: # regulating the weights
                gene = [0 if abs(i) < E else i for i in gene]
                GRN_C.append(gene)
            GRN = GRN_C.copy()
            GRN_C.clear()
        if(not magnitude == True):
            # magnitude regulation 
            for gene in GRN:
                print(gene)
                xd = []
                for var in gene:
                    if(var == 0):
                        xd.append(0)
                    elif(var > 0):
                        xd.append(1)
                    else:
                        xd.append(-1)
                GRN_C.append(xd)
            GRN = GRN_C.copy()
            GRN_C.clear()
        
        for i in range(trial):
            print('trial: ', i)
            geneExpression.clear()
            output.clear()
            # initializing the gene expression sequence
            for xd in range(n):
                geneExpression.append(random.randint(0,IGEV*10)/10)
            for t in range(T):
                output.append(geneExpression)
                geneExpression = matmul(GRN,geneExpression)
                #modification on values of expression 
            for var in range(len(output)):
                for var2 in range(len(output[var])):
                    noise = (random.randint(-50,50))/100
                    output[var][var2]= max(0,min(UB,noise+output[var][var2]))
            final.append(output.copy())
        return final
                

def GRNgenerator(n):
    GRN = []
    for i in range(n):
        var = []
        for j in range(n):
            var.append(randint(-50,50)/20)
        GRN.append(var)
    return GRN


GRN = GRNgenerator(3)
print('The GRN is: ', GRN)
data = generator(GRN, magnitude=True, trial=10)
counter = 0
for trial in data:
    print('Trail Number: ', counter)
    print(trial)
    counter +=1
 
##########################################################
## saving section
file = open('GRN.pickle', 'wb')
pickle.dump(GRN, file)
file.close
 
file = open('data.pickle', 'wb')
pickle.dump(data, file)
file.close
##########################################################
print('Number of trial: ', len(data))
print('Number of time step: ', len(data[0]))
print('Number of genes: ', len(data[0][0]))                
                 
                 