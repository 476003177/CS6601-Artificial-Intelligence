# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 00:31:23 2019

@author: songl
"""

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
    BayesNet = BayesianModel()
    # TODO: finish this function    
    BayesNet.add_node("NI")
    BayesNet.add_node("St")
    BayesNet.add_node("I")
    BayesNet.add_node("S")
    BayesNet.add_node("T")
    BayesNet.add_node("L")
    BayesNet.add_node("B")
    BayesNet.add_edge("NI","I")
    BayesNet.add_edge("St","I")
    #BayesNet.add_edge("I","S")
    BayesNet.add_edge("I","T")
    BayesNet.add_edge("I","L")
    BayesNet.add_edge("S","L")
    BayesNet.add_edge("S","B")
    return BayesNet
    
def set_probability(bayes_net):
    cpd_NI=TabularCPD("NI",2,values=[[0.6],[0.4]])
    cpd_St=TabularCPD("St",2,values=[[0.1],[0.9]])
    cpd_I=TabularCPD("I", 2, values=[[0.3,0.5,0.1,0.3],[0.7,0.5,0.9,0.7]], evidence=["NI","St"], evidence_card=[2, 2])
    #cpd_S=TabularCPD("S",2,values=[[0.9,0.2],[0.1, 0.8]], evidence=["I"], evidence_card=[2])
    cpd_S=TabularCPD("S",2,values=[[0.1],[0.9]])
    cpd_T=TabularCPD("T",2,values=[[0.7,0.4],[0.3, 0.6]], evidence=["I"], evidence_card=[2])
    cpd_B=TabularCPD("B",2,values=[[0.2,0.5],[0.8, 0.5]], evidence=["S"], evidence_card=[2])
    cpd_L=TabularCPD("L", 2, values=[[0.8,0.7,0.4,0.3],[0.2,0.3,0.6,0.7]], evidence=["I","S"], evidence_card=[2, 2])
    bayes_net.add_cpds(cpd_NI,cpd_St,cpd_I,cpd_S,cpd_T,cpd_B,cpd_L)

b=make_power_plant_net()
set_probability(b)
solver=VariableElimination(b)
#marginal_prob = solver.query(variables=['I'])
#prob = marginal_prob['I'].values[1]
#marginal_prob = solver.query(variables=['L'])
#prob = marginal_prob['L'].values[1]
#conditional_prob = solver.query(variables=['S'],evidence={'St':1})
#prob = conditional_prob['S'].values[1]
#conditional_prob = solver.query(variables=['NI'],evidence={'S':1,'L':1,'St':1})
#prob = conditional_prob['NI'].values[1]
conditional_prob1 = solver.query(variables=['B'],evidence={'I':1})
prob1 = conditional_prob1['B'].values[1]
conditional_prob2 = solver.query(variables=['B'],evidence={'I':1,'T':1})
prob2 = conditional_prob1['B'].values[1]