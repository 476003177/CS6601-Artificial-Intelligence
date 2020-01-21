# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 23:31:58 2019

@author: songl
"""


import heapq
import os
import pickle
import math
import datetime

class PriorityQueue(object):
    """
    A queue structure where each element is served in order of priority.

    Elements in the queue are popped based on the priority with higher priority
    elements being served before lower priority elements.  If two elements have
    the same priority, they will be served in the order they were added to the
    queue.

    Traditionally priority queues are implemented with heaps, but there are any
    number of implementation options.

    (Hint: take a look at the module heapq)

    Attributes:
        queue (list): Nodes added to the priority queue.
    """

    def __init__(self):
        """Initialize a new Priority Queue."""
        self.count=0
        self.queue = []
        heapq.heapify(self.queue)
        

    def pop(self):
        """
        Pop top priority node from queue.

        Returns:
            The node with the highest priority.
        """

        # TODO: finish this function!
        
        while self.queue:
            node=heapq.heappop(self.queue)
            if node[-1]!='removed':
                node=node[:1]+node[2:]
                return node 
        raise KeyError('pop from an empty priority queue')
    

    def remove(self, priority, key):
        """
        Remove a node from the queue only if it esists and it has higher priority
        higher priority means smaller number of node[0]
        
        Return False if not exist or not need to change priority
        Return True if exist and need to chane priority, in the function, 
        already remove the original item with old priority
        """
        for i in range(len(self.queue)):
            if self.queue[i][-1]==key:
                if self.queue[i][0]<=priority:
                    return False
                else:
                    self.queue[i]=list(self.queue[i])
                    self.queue[i][-1]='removed'
                    self.queue[i][0]=priority
                    self.queue[i]=tuple(self.queue[i])
                    return True
        return False
                 
                

    def __iter__(self):
        """Queue iterator."""

        return iter(sorted(self.queue))

    def __str__(self):
        """Priority Queue to string."""

        return 'PQ:%s' % self.queue

    def append(self, node):
        """
        Append a node to the queue.

        Args:
            node: Comparable Object to be added to the priority queue.
        """

        # TODO: finish this function!
        #count=self.size()+1
        #node=node[:1]+(count,)+node[1:]
        self.count+=1
        node=node[:1]+(self.count,)+node[1:]
        heapq.heappush(self.queue,node)
        
    def __contains__(self, key):
        """
        Containment Check operator for 'in'

        Args:
            key: The key to check for in the queue.

        Returns:
            True if key is found in queue, False otherwise.
        """

        return key in [n[-1] for n in self.queue]

    def __eq__(self, other):
        """
        Compare this Priority Queue with another Priority Queue.

        Args:
            other (PriorityQueue): Priority Queue to compare against.

        Returns:
            True if the two priority queues are equivalent.
        """

        return self.queue == other.queue

    def size(self):
        """
        Get the current size of the queue.

        Returns:
            Integer of number of items in queue.
        """

        return len(self.queue)

    def clear(self):
        """Reset queue to empty (no nodes)."""

        self.queue = []

    def top(self):
        """
        Get the top item in the queue.

        Returns:
            The first item stored in teh queue.
        """

        return self.queue[0]

def bidirectional_ucs(childmap, pathmap, start, goal):
    """
    Exercise 1: Bidirectional Search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if start==goal:
        return []
    frontierF=PriorityQueue()
    frontierR=PriorityQueue()
    exploredF={}
    exploredR={}
    pathF=[start]
    pathR=[goal]
    path=[]
    frontierF.append((0,pathF,start))
    frontierR.append((0,pathR,goal))
    mu=float('inf')
    while True:
        try: 
            stateF=frontierF.pop()
            stateR=frontierR.pop()
        except KeyError:
            return None
        if stateF[0]+stateR[0]>=mu:
            return path  
        # what you should return is the path corresponding to mu
        exploredF[stateF[2]]=(stateF[0],stateF[1])
        exploredR[stateR[2]]=(stateR[0],stateR[1])
        for child in sorted(childmap[stateF[2]]):
            if child in exploredR:
                print(child)
                if stateF[0]+pathmap[stateF[2]+child]+exploredR[child][0]<mu:
                    mu=stateF[0]+pathmap[stateF[2]+child]+exploredR[child][0]
                    path=stateF[1]+exploredR[child][1]
            count=stateF[0]+pathmap[stateF[2]+child]
            if (child not in frontierF) and (child not in exploredF):
                frontierF.append((count,stateF[1]+[child],child))
            if frontierF.remove(count,child):
                frontierF.append((count,stateF[1]+[child],child))
        for child in sorted(childmap[stateR[2]]):
            if child in exploredF:
                print(child)
                if stateR[0]+pathmap[stateR[2]+child]+exploredF[child][0]<mu:
                    mu=stateR[0]+pathmap[stateR[2]+child]+exploredF[child][0]
                    path=exploredF[child][1]+stateR[1]
            count=stateR[0]+pathmap[stateR[2]+child]
            if (child not in frontierR) and (child not in exploredR):
                frontierR.append((count,[child]+stateR[1],child))
            if frontierR.remove(count,child):
                frontierR.append((count,[child]+stateR[1],child))
                
childmap={'A':'B','B':'ACD','C':'BD','D':'CE','E':'DFG',
       'F':'EG','G':'EFH','H':'GIJ','I':'HJ','J':'HIK',
       'K':'JLM','L':'KMS','M':'KLN','N':'MOR','O':'NP',
       'P':'OQ','Q':'PRV','R':'NQSU','S':'LRT','T':'SU',
       'U':'RTV','V':'QU'}
pathmap={'AB':1,'BA':1,'BC':1,'CB':1,'BD':1.5,'DB':1.5,
         'CD':2,'DC':2,'DE':1,'ED':1,'EF':3,'FE':3,
         'EG':3,'GE':3,'FG':2.5,'GF':2.5,'GH':1,'HG':1,
         'HI':4,'IH':4,'HJ':3.5,'JH':3.5,'IJ':2.5,'JI':2.5,
         'JK':2,'KJ':2,'KL':3,'LK':3,'KM':2,'MK':2,
         'LM':3.5,'ML':3.5,'MN':1,'NM':1,'NO':1.5,'ON':1.5,
         'NR':4,'RN':4,'OP':3,'PO':3,'PQ':2,'QP':2,
         'QR':3.5,'RQ':3.5,'QV':4.5,'VQ':4.5,'RS':5,'SR':5,
         'RU':3,'UR':3,'ST':2.5,'TS':2.5,'TU':4.5,'UT':4.5,
         'UV':1.5,'VU':1.5,'LS':5,'SL':5}

path=bidirectional_ucs(childmap,pathmap,'A','V')