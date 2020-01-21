# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 21:32:27 2019

@author: songl
"""

import heapq
import os
import pickle
import math


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
        # use self.count to ensure first in first out
        self.count+=1
        node=node[:1]+(self.count,)+node[1:]
        #count=self.size()+1
        #node=node[:1]+(self.count,)+node[1:]
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

def uniform_cost_search(graph, start, goal):
    """
    Warm-up exercise: Implement uniform_cost_search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    # start/goal [] 0 is row index, 1 is column index
    if start==goal:
        return []
    frontier=PriorityQueue()
    explored=set()
    path=[start]
    frontier.append((0,path,start))
    while True:
        try: 
            state=frontier.pop()
            explored.add((state[2]))
        except KeyError:
            return None
        if state[2]==goal:
            return state,explored
        children=[]
        # RIGHT, DOWN, LEFT and UP
        if state[2][1]+1<=15 and graph[state[2][0]][state[2][1]+1]==0:
            children+=[(state[2][0],state[2][1]+1)]
        if state[2][0]+1<=9 and graph[state[2][0]+1][state[2][1]]==0:
            children+=[(state[2][0]+1,state[2][1])]
        if state[2][1]-1>=0 and graph[state[2][0]][state[2][1]-1]==0:
            children+=[(state[2][0],state[2][1]-1)]
        if state[2][0]-1>=0 and graph[state[2][0]-1][state[2][1]]==0:
            children+=[(state[2][0]-1,state[2][1])]
        for child in children:
            count=state[0]+1
            if (child not in frontier) and (child not in explored):
                frontier.append((count,state[1]+[child],child))
            if frontier.remove(count,child): # update the count
                frontier.append((count,state[1]+[child],child))

graph=[[0,0,0,1,0,0,1,1,0,0,0,0,0,0,1,1],
       [0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1],
       [0,1,1,1,1,1,0,0,0,1,1,0,0,0,0,1],
       [0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1],
       [0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0],
       [0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1],
       [0,0,1,1,0,0,0,0,0,1,1,0,0,0,0,1],
       [0,0,0,1,0,0,1,0,0,1,1,0,0,0,0,1],
       [0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1],
       [0,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1]]

path,explored=uniform_cost_search(graph,(0,2),(2,14))
pq=PriorityQueue();
pq.append((1,'a'))
pq.append((1,'b'))
pq.append((1,'c'))