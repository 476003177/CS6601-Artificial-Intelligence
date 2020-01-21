# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 20:17:33 2019

@author: songl
"""
import numpy as np
from submission import *
game = get_game_network()
sample=compare_sampling(game,[])
prob=calculate_posterior(game)

