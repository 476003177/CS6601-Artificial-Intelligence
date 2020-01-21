# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 20:06:38 2019

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
    BayesNet.add_node("A")
    BayesNet.add_node("B")
    BayesNet.add_node("C")
    BayesNet.add_node("D")
    BayesNet.add_node("E")
    BayesNet.add_node("F")
    BayesNet.add_node("J")
    BayesNet.add_edge("A","C")
    BayesNet.add_edge("B","C")
    BayesNet.add_edge("C","D")
    BayesNet.add_edge("C","E")
    BayesNet.add_edge("E","J")
    BayesNet.add_edge("F","J")
    return BayesNet
    
def set_probability(bayes_net):
    cpd_A=TabularCPD("A",2,values=[[0.3],[0.7]])
    cpd_B=TabularCPD("B",2,values=[[0.2],[0.8]])
    cpd_C=TabularCPD("C", 2, values=[[0.8,0.1,0.15,0.05],[0.2,0.9,0.85,0.95]], evidence=["A","B"], evidence_card=[2, 2])
    cpd_D=TabularCPD("D",2,values=[[0.8,0.3],[0.2, 0.7]], evidence=["C"], evidence_card=[2])
    cpd_E=TabularCPD("E",2,values=[[0.95,0.1],[0.05, 0.9]], evidence=["C"], evidence_card=[2])
    cpd_F=TabularCPD("F",2,values=[[0.6],[0.4]])
    cpd_J=TabularCPD("J", 2, values=[[0.8,0.18,0.25,0.02],[0.2,0.82,0.75,0.98]], evidence=["E","F"], evidence_card=[2, 2])
    bayes_net.add_cpds(cpd_A,cpd_B,cpd_C,cpd_D,cpd_E,cpd_F,cpd_J)

b=make_power_plant_net()
set_probability(b)
solver=VariableElimination(b)
conditional_prob = solver.query(variables=['E'],evidence={'B':0})
prob = conditional_prob['E'].values[1]
    