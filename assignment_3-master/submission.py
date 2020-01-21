import sys

'''
WRITE YOUR CODE BELOW.
'''
from numpy import zeros, float32
import numpy as np
#  pgmpy
import pgmpy
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
#You are not allowed to use following set of modules from 'pgmpy' Library.
#
# pgmpy.sampling.*
# pgmpy.factors.*
# pgmpy.estimators.*

def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem. 
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    BayesNet = BayesianModel()
    # TODO: finish this function    
    BayesNet.add_node("alarm")
    BayesNet.add_node("faulty alarm")
    BayesNet.add_node("gauge")
    BayesNet.add_node("faulty gauge")
    BayesNet.add_node("temperature")
    BayesNet.add_edge("gauge","alarm")
    BayesNet.add_edge("temperature","gauge")
    BayesNet.add_edge("temperature","faulty gauge")
    BayesNet.add_edge("faulty gauge","gauge")
    BayesNet.add_edge("faulty alarm","alarm")
    return BayesNet


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    # TODO: set the probability distribution for each node
    cpd_T=TabularCPD("temperature",2,values=[[0.8],[0.2]])
    cpd_faultyAlarm=TabularCPD("faulty alarm",2,values=[[0.85],[0.15]])
    cpd_faultGauge=TabularCPD("faulty gauge",2,values=[[0.95,0.2],[0.05, 0.8]], evidence=["temperature"], evidence_card=[2])
    cpd_gauge=TabularCPD("gauge", 2, values=[[0.95,0.2,0.05,0.8],[0.05,0.8,0.95,0.2]], evidence=["temperature","faulty gauge"], evidence_card=[2, 2])
    cpd_alarm=TabularCPD("alarm", 2, values=[[0.9,0.55,0.1,0.45],[0.1,0.45,0.9,0.55]], evidence=["gauge","faulty alarm"], evidence_card=[2, 2])
    bayes_net.add_cpds(cpd_T,cpd_faultyAlarm,cpd_faultGauge,cpd_gauge,cpd_alarm)
    return bayes_net


def get_alarm_prob(bayes_net):
    """Calculate the marginal 
    probability of the alarm 
    ringing in the 
    power plant system."""
    # TODO: finish this function
    solver=VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['alarm'])
    alarm_prob=marginal_prob['alarm'].values[1]
    return alarm_prob

def get_gauge_prob(bayes_net):
    """Calculate the marginal
    probability of the gauge 
    showing hot in the 
    power plant system."""
    # TODO: finish this function
    solver=VariableElimination(bayes_net)
    marginal_prob = solver.query(variables=['gauge'])
    gauge_prob=marginal_prob['gauge'].values[1]
    return gauge_prob


def get_temperature_prob(bayes_net):
    """Calculate the conditional probability 
    of the temperature being hot in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    # TODO: finish this function
    solver=VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['temperature'],evidence={'alarm':1,'faulty gauge':0,'faulty alarm':0})
    temp_prob = conditional_prob['temperature'].values[1]
    return temp_prob


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    BayesNet = BayesianModel()
    # TODO: fill this out
    BayesNet.add_node("A")  
    BayesNet.add_node("B")
    BayesNet.add_node("C")
    BayesNet.add_node("AvB")
    BayesNet.add_node("BvC")
    BayesNet.add_node("CvA")
    BayesNet.add_edge("A","AvB")
    BayesNet.add_edge("A","CvA")
    BayesNet.add_edge("B","AvB")
    BayesNet.add_edge("B","BvC")
    BayesNet.add_edge("C","BvC")
    BayesNet.add_edge("C","CvA")
    cpd_a = TabularCPD('A', 4, values=[[0.15], [0.45],[0.3],[0.1]])
    cpd_b = TabularCPD('B', 4, values=[[0.15], [0.45],[0.3],[0.1]])
    cpd_c = TabularCPD('C', 4, values=[[0.15], [0.45],[0.3],[0.1]])
    cpd_avb=TabularCPD("AvB",3, values=[[0.1,0.2,0.15,0.05,0.6,0.1,0.2,0.15,0.75,0.6,0.1,0.2,0.9,0.75,0.6,0.1],\
                       [0.1,0.6,0.75,0.9,0.2,0.1,0.6,0.75,0.15,0.2,0.1,0.6,0.05,0.15,0.2,0.1],\
                       [0.8,0.2,0.1,0.05,0.2,0.8,0.2,0.1,0.1,0.2,0.8,0.2,0.05,0.1,0.2,0.8]],\
                       evidence=["A","B"], evidence_card=[4, 4])
    cpd_bvc=TabularCPD("BvC",3, values=[[0.1,0.2,0.15,0.05,0.6,0.1,0.2,0.15,0.75,0.6,0.1,0.2,0.9,0.75,0.6,0.1],\
                       [0.1,0.6,0.75,0.9,0.2,0.1,0.6,0.75,0.15,0.2,0.1,0.6,0.05,0.15,0.2,0.1],\
                       [0.8,0.2,0.1,0.05,0.2,0.8,0.2,0.1,0.1,0.2,0.8,0.2,0.05,0.1,0.2,0.8]],\
                       evidence=["B","C"], evidence_card=[4, 4])
    cpd_cva=TabularCPD("CvA",3, values=[[0.1,0.2,0.15,0.05,0.6,0.1,0.2,0.15,0.75,0.6,0.1,0.2,0.9,0.75,0.6,0.1],\
                       [0.1,0.6,0.75,0.9,0.2,0.1,0.6,0.75,0.15,0.2,0.1,0.6,0.05,0.15,0.2,0.1],\
                       [0.8,0.2,0.1,0.05,0.2,0.8,0.2,0.1,0.1,0.2,0.8,0.2,0.05,0.1,0.2,0.8]],\
                       evidence=["C","A"], evidence_card=[4, 4])
    BayesNet.add_cpds(cpd_a,cpd_b,cpd_c,cpd_avb,cpd_bvc,cpd_cva)
    return BayesNet


def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C. 
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function    
    solver=VariableElimination(bayes_net)
    conditional_prob = solver.query(variables=['BvC'],evidence={'AvB':0,'CvA':2})
    posterior = conditional_prob['BvC'].values
    return posterior # list 


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm 
    given a Bayesian network and an initial state value. 
    
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.    
    """
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values
    m=match_table
    t=team_table
    sample = list(initial_state)    
    # TODO: finish this function
    # randomly generate an initial state
    if not sample:
        A=np.random.randint(4)
        B=np.random.randint(4)
        C=np.random.randint(4)
        AvB=0
        BvC=np.random.randint(3)
        CvA=2
        sample=[A,B,C,AvB,BvC,CvA]
    sampleIndex=np.array([0,1,2,4])
    index=np.random.choice(sampleIndex)
    #ABC have same distribution
    # but you need to reflect the impact of other variables on A
    if index==0:
        dist=[0,0,0,0]
        dist0numer=m[0][0][sample[1]]*m[2][sample[2]][0]*t[0]
        dist_denom=(m[0][0][sample[1]]*t[0]+m[0][1][sample[1]]*t[1]+m[0][2][sample[1]]*t[2]+m[0][3][sample[1]]*t[3])*\
                    (m[2][sample[2]][0]*t[0]+m[2][sample[2]][1]*t[1]+m[2][sample[2]][2]*t[2]+m[2][sample[2]][3]*t[3])
        dist[0]=dist0numer/dist_denom
        dist1numer=m[0][1][sample[1]]*m[2][sample[2]][1]*t[1]
        dist[1]=dist1numer/dist_denom
        dist2numer=m[0][2][sample[1]]*m[2][sample[2]][2]*t[2]
        dist[2]=dist2numer/dist_denom
        dist3numer=m[0][3][sample[1]]*m[2][sample[2]][3]*t[3]
        dist[3]=dist3numer/dist_denom
        total=sum(dist)
        dist[0]=dist[0]/total
        dist[1]=dist[1]/total
        dist[2]=dist[2]/total
        dist[3]=dist[3]/total
        sample[index]=np.random.choice(4,p=dist)
    elif index==1:
        dist=[0,0,0,0]
        dist_denom=(m[0][sample[0]][0]*t[0]+m[0][sample[0]][1]*t[1]+m[0][sample[0]][2]*t[2]+m[0][sample[0]][3]*t[3])*\
                    (m[sample[4]][0][sample[2]]*t[0]+m[sample[4]][1][sample[2]]*t[1]+m[sample[4]][2][sample[2]]*t[2]+m[sample[4]][3][sample[2]]*t[3])
        dist0numer=m[0][sample[0]][0]*m[sample[4]][0][sample[2]]*t[0]
        dist1numer=m[0][sample[0]][1]*m[sample[4]][1][sample[2]]*t[1]
        dist2numer=m[0][sample[0]][2]*m[sample[4]][2][sample[2]]*t[2]
        dist3numer=m[0][sample[0]][3]*m[sample[4]][3][sample[2]]*t[3]
        dist[0]=dist0numer/dist_denom
        dist[1]=dist1numer/dist_denom
        dist[2]=dist2numer/dist_denom
        dist[3]=dist3numer/dist_denom
        total=sum(dist)
        dist[0]=dist[0]/total
        dist[1]=dist[1]/total
        dist[2]=dist[2]/total
        dist[3]=dist[3]/total
        sample[index]=np.random.choice(4,p=dist)
    elif index==2:
        dist=[0,0,0,0]
        dist_denom=(m[sample[4]][sample[1]][0]*t[0]+m[sample[4]][sample[1]][1]*t[1]+m[sample[4]][sample[1]][2]*t[2]+m[sample[4]][sample[1]][3]*t[3])*\
                    (m[2][0][sample[0]]*t[0]+m[2][1][sample[0]]*t[1]+m[2][2][sample[0]]*t[2]+m[2][3][sample[0]]*t[3])
        dist0numer=m[sample[4]][sample[1]][0]*m[2][0][sample[0]]*t[0]
        dist1numer=m[sample[4]][sample[1]][1]*m[2][1][sample[0]]*t[1]
        dist2numer=m[sample[4]][sample[1]][2]*m[2][2][sample[0]]*t[2]
        dist3numer=m[sample[4]][sample[1]][3]*m[2][3][sample[0]]*t[3]
        dist[0]=dist0numer/dist_denom
        dist[1]=dist1numer/dist_denom
        dist[2]=dist2numer/dist_denom
        dist[3]=dist3numer/dist_denom
        total=sum(dist)
        dist[0]=dist[0]/total
        dist[1]=dist[1]/total
        dist[2]=dist[2]/total
        dist[3]=dist[3]/total
        sample[index]=np.random.choice(4,p=dist)
    else:
        dist=[m[0][sample[1]][sample[2]],m[1][sample[1]][sample[2]],m[2][sample[1]][sample[2]]]
        sample[index]=np.random.choice(3,p=dist)
    sample=tuple(sample)
    return sample


def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value. 
    initial_state is a list of length 6 where: 
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)    
    Returns the new state sampled from the probability distribution as a tuple of length 6. 
    """
    A_cpd = bayes_net.get_cpds("A")      
    AvB_cpd = bayes_net.get_cpds("AvB")
    match_table = AvB_cpd.values
    team_table = A_cpd.values
    sample = list(initial_state)    
    # TODO: finish this function
    # if empty initial_state
    if not sample:
        A=np.random.randint(4)
        B=np.random.randint(4)
        C=np.random.randint(4)
        AvB=0
        BvC=np.random.randint(3)
        CvA=2
        sample=[A,B,C,AvB,BvC,CvA]
    A=np.random.randint(4)
    B=np.random.randint(4)
    C=np.random.randint(4)
    AvB=0
    BvC=np.random.randint(3)
    CvA=2
    Nsample=[A,B,C,AvB,BvC,CvA]
    prob=team_table[sample[0]]*team_table[sample[1]]*team_table[sample[2]]*\
         match_table[0][sample[0]][sample[1]]*\
         match_table[sample[4]][sample[1]][sample[2]]*\
         match_table[2][sample[2]][sample[0]]
    Nprob=team_table[Nsample[0]]*team_table[Nsample[1]]*team_table[Nsample[2]]*\
         match_table[0][Nsample[0]][Nsample[1]]*\
         match_table[Nsample[4]][Nsample[1]][Nsample[2]]*\
         match_table[2][Nsample[2]][Nsample[0]]
    r=min(1,Nprob/prob)
    u=np.random.uniform(0,1)
    if u<r:
        sample=tuple(Nsample)
    else:
        sample=tuple(sample)
    return sample


def compare_sampling(bayes_net, initial_state):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""    
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence =np.array([0,0,0]) # posterior distribution of the BvC match as produced by Gibbs 
    MH_convergence =np.array([0,0,0]) # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function
    delta=0.001
    N=20
    # first do gibbs sampling
    Gibbs_result=np.array([0,0,0])
    NGibbs=0
    sampleGibbs=Gibbs_sampler(bayes_net, initial_state)
    while (True):
        sampleGibbs=Gibbs_sampler(bayes_net, sampleGibbs)
        print(sampleGibbs)
        Gibbs_count+=1
        if sampleGibbs[4]==0:
            Gibbs_result[0]+=1
        elif sampleGibbs[4]==1:
            Gibbs_result[1]+=1
        else:
            Gibbs_result[2]+=1
        diffGibbs=abs(Gibbs_result/Gibbs_count-Gibbs_convergence)
        if diffGibbs[0]<=delta and diffGibbs[1]<=delta and diffGibbs[2]<=delta:
            NGibbs+=1
        else:
            NGibbs=0
        Gibbs_convergence=Gibbs_result/Gibbs_count
        if NGibbs>=N:
            break
    # then do MH sampling
    MH_result=np.array([0,0,0])
    N_MH=0
    oldSampleMH=MH_sampler(bayes_net, initial_state)
    while (True):
        sampleMH=MH_sampler(bayes_net,oldSampleMH)
        if sampleMH==oldSampleMH:
            MH_rejection_count+=1
        else:
            MH_count+=1
            if sampleMH[4]==0:
                MH_result[0]+=1
            elif sampleMH[4]==1:
                MH_result[1]+=1
            else:
                MH_result[2]+=1
        diffMH=abs(MH_result/MH_count-MH_convergence)
        if diffMH[0]<=delta and diffMH[1]<=delta and diffMH[2]<=delta:
            N_MH+=1
        else:
            N_MH=0
        MH_convergence=MH_result/MH_count
        oldSampleMH=sampleMH
        if N_MH>=N:
            break
    Gibbs_convergence=list(Gibbs_convergence)
    MH_convergence=list(MH_convergence)
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count


def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    choice = 1
    options = ['Gibbs','Metropolis-Hastings']
    factor = 10
    return options[choice], factor


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Xufan Song"
