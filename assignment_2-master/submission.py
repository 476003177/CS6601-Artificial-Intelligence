# coding=utf-8
"""
This file is your main submission that will be graded against. Only copy-paste
code on the relevant classes included here. Do not add any classes or functions
to this file that are not part of the classes that we want.
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
        count=self.size()+1
        node=node[:1]+(count,)+node[1:]
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


def breadth_first_search(graph, start, goal):
    """
    Warm-up exercise: Implement breadth-first-search.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    #state is (count,path,node)
    #count is depth count
    if start==goal:
        return []
    frontier=PriorityQueue()
    explored=set()
    path=[start]
    frontier.append((0,path,start))
    while True:
        try: 
            state=frontier.pop()
        except KeyError:
            return None
        explored.add((state[2]))
        count=state[0]+1
        for child in sorted(list(graph.neighbors(state[2])),key=str.lower):
            if child==goal:
                return state[1]+[child]
            if (child not in frontier) and (child not in explored):
                frontier.append((count,state[1]+[child],child))
            
            
        
                

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
    if start==goal:
        return []
    frontier=PriorityQueue()
    explored=set()
    path=[start]
    frontier.append((0,path,start))
    while True:
        try: 
            state=frontier.pop()
        except KeyError:
            return None
        if state[2]==goal:
            return state[1]
        explored.add((state[2]))
        for child in list(graph.neighbors(state[2])):
            count=state[0]+graph.get_edge_weight(state[2], child)
            if (child not in frontier) and (child not in explored):
                frontier.append((count,state[1]+[child],child))
            if frontier.remove(count,child):
                frontier.append((count,state[1]+[child],child))
            


def null_heuristic(graph, v, goal):
    """
    Null heuristic used as a base line.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        0
    """

    return 0


def euclidean_dist_heuristic(graph, v, goal):
    """
    Warm-up exercise: Implement the euclidean distance heuristic.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Euclidean distance between `v` node and `goal` node
    """

    # TODO: finish this function!
    vPos=graph.node[v]['pos']
    vX=vPos[0]
    vY=vPos[1]
    goalPos=graph.node[goal]['pos']
    goalX=goalPos[0]
    goalY=goalPos[1]
    return math.sqrt((vX-goalX)**2+(vY-goalY)**2)


def a_star(graph, start, goal, heuristic=euclidean_dist_heuristic):
    """
    Warm-up exercise: Implement A* algorithm.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    if start==goal:
        return []
    frontier=PriorityQueue()
    explored=set()
    path=[start]
    frontier.append((0+heuristic(graph,start,goal),0,path,start))
    #note: f=g+h and new f doesn't include old h
    #node=(f,g,path,node)
    while True:
        try: 
            state=frontier.pop()
        except KeyError:
            return None
        if state[3]==goal:
            return state[2]
        explored.add((state[3]))
        for child in list(graph.neighbors(state[3])):
            f=state[1]+graph.get_edge_weight(state[3], child)+heuristic(graph,child,goal)
            g=state[1]+graph.get_edge_weight(state[3], child)
            if (child not in frontier) and (child not in explored):
                frontier.append((f,g,state[2]+[child],child))
            if frontier.remove(f,child):
                frontier.append((f,g,state[2]+[child],child))


def bidirectional_ucs(graph, start, goal):
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
        for child in list(graph.neighbors(stateF[2])):
            if child in exploredR:
                if stateF[0]+graph.get_edge_weight(stateF[2], child)+exploredR[child][0]<mu:
                    mu=stateF[0]+graph.get_edge_weight(stateF[2], child)+exploredR[child][0]
                    path=stateF[1]+exploredR[child][1]
            count=stateF[0]+graph.get_edge_weight(stateF[2], child)
            if (child not in frontierF) and (child not in exploredF):
                frontierF.append((count,stateF[1]+[child],child))
            if frontierF.remove(count,child):
                frontierF.append((count,stateF[1]+[child],child))
        for child in list(graph.neighbors(stateR[2])):
            if child in exploredF:
                if stateR[0]+graph.get_edge_weight(stateR[2], child)+exploredF[child][0]<mu:
                    mu=stateR[0]+graph.get_edge_weight(stateR[2], child)+exploredF[child][0]
                    path=exploredF[child][1]+stateR[1]
            count=stateR[0]+graph.get_edge_weight(stateR[2], child)
            if (child not in frontierR) and (child not in exploredR):
                frontierR.append((count,[child]+stateR[1],child))
            if frontierR.remove(count,child):
                frontierR.append((count,[child]+stateR[1],child))


def bidirectional_a_star(graph, start, goal,
                         heuristic=euclidean_dist_heuristic):
    """
    Exercise 2: Bidirectional A*.

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

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
    frontierF.append((0+heuristic(graph,start,goal),0,pathF,start))
    frontierR.append((0+heuristic(graph,start,goal),0,pathR,goal))
    mu=float('inf')
    while True:
        try: 
            stateF=frontierF.pop()
            stateR=frontierR.pop()
        except KeyError:
            return None
        if stateF[0]+stateR[0]>=mu+heuristic(graph,start,goal):
            return path
        exploredF[stateF[3]]=(stateF[1],stateF[2])
        exploredR[stateR[3]]=(stateR[1],stateR[2])
        for child in list(graph.neighbors(stateF[3])):
            if child in exploredR:
                if stateF[1]+graph.get_edge_weight(stateF[3], child)+exploredR[child][0]<mu:
                    mu=stateF[1]+graph.get_edge_weight(stateF[3], child)+exploredR[child][0]
                    path=stateF[2]+exploredR[child][1]
            f=stateF[1]+graph.get_edge_weight(stateF[3], child)+pf(graph,child,start,goal,heuristic)
            g=stateF[1]+graph.get_edge_weight(stateF[3], child)
            if (child not in frontierF) and (child not in exploredF):
                frontierF.append((f,g,stateF[2]+[child],child))
            if frontierF.remove(f,child):
                frontierF.append((f,g,stateF[2]+[child],child))
        for child in list(graph.neighbors(stateR[3])):
            if child in exploredF:
                if stateR[1]+graph.get_edge_weight(stateR[3], child)+exploredF[child][0]<mu:
                    mu=stateR[1]+graph.get_edge_weight(stateR[3], child)+exploredF[child][0]
                    path=exploredF[child][1]+stateR[2]
            f=stateR[1]+graph.get_edge_weight(stateR[3], child)+pr(graph,child,start,goal,heuristic)
            g=stateR[1]+graph.get_edge_weight(stateR[3], child)
            if (child not in frontierR) and (child not in exploredR):
                frontierR.append((f,g,[child]+stateR[2],child))
            if frontierR.remove(f,child):
                frontierR.append((f,g,[child]+stateR[2],child))

def pf(graph,v,start,goal,heuristic):
    return 0.5*(heuristic(graph,v,goal)-heuristic(graph,v,start))+0.5*heuristic(graph,start,goal)

def pr(graph,v,start,goal,heuristic):
    return 0.5*(-heuristic(graph,v,goal)+heuristic(graph,v,start))+0.5*heuristic(graph,start,goal)


def bidirectional_ucs_fortri(graph, start, goal):
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
        for child in list(graph.neighbors(stateF[2])):
            if child in exploredR:
                if stateF[0]+graph.get_edge_weight(stateF[2], child)+exploredR[child][0]<mu:
                    mu=stateF[0]+graph.get_edge_weight(stateF[2], child)+exploredR[child][0]
                    path=stateF[1]+exploredR[child][1]
            count=stateF[0]+graph.get_edge_weight(stateF[2], child)
            if (child not in frontierF) and (child not in exploredF):
                frontierF.append((count,stateF[1]+[child],child))
            if frontierF.remove(count,child):
                frontierF.append((count,stateF[1]+[child],child))
        for child in list(graph.neighbors(stateR[2])):
            if child in exploredF:
                if stateR[0]+graph.get_edge_weight(stateR[2], child)+exploredF[child][0]<mu:
                    mu=stateR[0]+graph.get_edge_weight(stateR[2], child)+exploredF[child][0]
                    path=exploredF[child][1]+stateR[1]
            count=stateR[0]+graph.get_edge_weight(stateR[2], child)
            if (child not in frontierR) and (child not in exploredR):
                frontierR.append((count,[child]+stateR[1],child))
            if frontierR.remove(count,child):
                frontierR.append((count,[child]+stateR[1],child))
                
                
def tridirectional_search(graph, goals):
    """
    Exercise 3: Tridirectional UCS Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    if goals[0]==goals[1] and goals[1]==goals[2]:
        return []
    if goals[0]==goals[1]:
        return bidirectional_ucs_fortri(graph, goals[0], goals[1])
    if goals[0]==goals[2]:
        return bidirectional_ucs_fortri(graph, goals[0], goals[2])
    if goals[1]==goals[2]:
        return bidirectional_ucs_fortri(graph, goals[1], goals[2])
    [A,B,C]=goals
    frontierA=PriorityQueue()
    frontierB=PriorityQueue()
    frontierC=PriorityQueue()
    exploredA={}
    exploredB={}
    exploredC={}
    pathA=[A] 
    pathB=[B]
    pathC=[C]
    pathAB=[]
    pathBC=[]
    pathCA=[]
    muAB=float('inf')
    muBC=float('inf')
    muCA=float('inf')
    frontierA.append((0,pathA,A))
    frontierB.append((0,pathB,B))
    frontierC.append((0,pathC,C))
    endA=0
    endB=0
    endC=0
    findAB=0
    findBC=0
    findCA=0
    while True:
        try:
            if not endA:
                stateA=frontierA.pop()
            if not endB:
                stateB=frontierB.pop()
            if not endC:
                stateC=frontierC.pop()
        except KeyError:
            return None
        # consider the termination condition
        if stateA[0]+stateB[0]>=muAB:
            findAB=1
            #lenAB=len(pathAB) #this is the length of list, # of steps, not the total cost
            #you already hava muAB as the total cost
        if stateB[0]+stateC[0]>=muBC:
            findBC=1
        if stateC[0]+stateA[0]>=muCA:
            findCA=1
        if findAB==1 and findBC==0 and findCA==1:
            endA=1
            if stateB[0]+stateC[0]>=muAB and stateB[0]+stateC[0]>=muCA:
                return pathCA+pathAB[1:]
        if findAB==1 and findBC==1 and findCA==0:
            endB=1
            if stateC[0]+stateA[0]>=muAB and stateC[0]+stateA[0]>=muBC:
                return pathAB+pathBC[1:]
        if findAB==0 and findBC==1 and findCA==1:
            endC=1
            if stateA[0]+stateB[0]>=muBC and stateA[0]+stateB[0]>=muCA:
                return pathBC+pathCA[1:]
        if findAB==1 and findBC==1 and findCA==1:
            m=max(muAB,muBC,muCA)
            if m==muAB:
                return pathBC+pathCA[1:]
            if m==muBC:
                return pathCA+pathAB[1:]
            if m==muCA:
                return pathAB+pathBC[1:]
        # what you should return is the path corresponding to mu
        exploredA[stateA[2]]=(stateA[0],stateA[1])
        exploredB[stateB[2]]=(stateB[0],stateB[1])
        exploredC[stateC[2]]=(stateC[0],stateC[1])
        
        for child in list(graph.neighbors(stateA[2])):
            if child in exploredB:
                if stateA[0]+graph.get_edge_weight(stateA[2], child)+exploredB[child][0]<muAB:
                    muAB=stateA[0]+graph.get_edge_weight(stateA[2], child)+exploredB[child][0]
                    pathAB=stateA[1]+list(reversed(exploredB[child][1]))
            if child in exploredC:
                if stateA[0]+graph.get_edge_weight(stateA[2], child)+exploredC[child][0]<muCA:
                    muCA=stateA[0]+graph.get_edge_weight(stateA[2], child)+exploredC[child][0]
                    pathCA=exploredC[child][1]+list(reversed(stateA[1]))        
            count=stateA[0]+graph.get_edge_weight(stateA[2], child)
            if (child not in frontierA) and (child not in exploredA):
                frontierA.append((count,stateA[1]+[child],child))
            if frontierA.remove(count,child):
                frontierA.append((count,stateA[1]+[child],child))
                
                
        for child in list(graph.neighbors(stateB[2])):
            if child in exploredA:
                if stateB[0]+graph.get_edge_weight(stateB[2], child)+exploredA[child][0]<muAB:
                    muAB=stateB[0]+graph.get_edge_weight(stateB[2], child)+exploredA[child][0]
                    pathAB=exploredA[child][1]+list(reversed(stateB[1]))
            if child in exploredC:
                if stateB[0]+graph.get_edge_weight(stateB[2], child)+exploredC[child][0]<muBC:
                    muBC=stateB[0]+graph.get_edge_weight(stateB[2], child)+exploredC[child][0]
                    pathBC=stateB[1]+list(reversed(exploredC[child][1]))        
            count=stateB[0]+graph.get_edge_weight(stateB[2], child)
            if (child not in frontierB) and (child not in exploredB):
                frontierB.append((count,stateB[1]+[child],child))
            if frontierB.remove(count,child):
                frontierB.append((count,stateB[1]+[child],child))
                
        for child in list(graph.neighbors(stateC[2])):
            if child in exploredA:
                if stateC[0]+graph.get_edge_weight(stateC[2], child)+exploredA[child][0]<muCA:
                    muCA=stateC[0]+graph.get_edge_weight(stateC[2], child)+exploredA[child][0]
                    pathCA=stateC[1]+list(reversed(exploredA[child][1]))
            if child in exploredB:
                if stateC[0]+graph.get_edge_weight(stateC[2], child)+exploredB[child][0]<muBC:
                    muBC=stateC[0]+graph.get_edge_weight(stateC[2], child)+exploredB[child][0]
                    pathBC=exploredB[child][1]+list(reversed(stateC[1]))        
            count=stateC[0]+graph.get_edge_weight(stateC[2], child)
            if (child not in frontierC) and (child not in exploredC):
                frontierC.append((count,stateC[1]+[child],child))
            if frontierC.remove(count,child):
                frontierC.append((count,stateC[1]+[child],child))
    


def tridirectional_upgraded(graph, goals, heuristic=euclidean_dist_heuristic):
    """
    Exercise 4: Upgraded Tridirectional Search

    See README.MD for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        goals (list): Key values for the 3 goals
        heuristic: Function to determine distance heuristic.
            Default: euclidean_dist_heuristic.

    Returns:
        The best path as a list from one of the goal nodes (including both of
        the other goal nodes).
    """
    # TODO: finish this function
    raise NotImplementedError


def return_your_name():
    """Return your name from this function"""
    # TODO: finish this function
    return "Xufan Song"


def custom_heuristic(graph, v, goal):
    """
       Feel free to use this method to try and work with different heuristics and come up with a better search algorithm.
       Args:
           graph (ExplorableGraph): Undirected graph to search.
           v (str): Key for the node to calculate from.
           goal (str): Key for the end node to calculate to.
       Returns:
           Custom heuristic distance between `v` node and `goal` node
       """

pass

# Extra Credit: Your best search method for the race
def custom_search(graph, start, goal, data=None):
    """
    Race!: Implement your best search algorithm here to compete against the
    other student agents.

    If you implement this function and submit your code to bonnie, you'll be
    registered for the Race!

    See README.md for exercise description.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        start (str): Key for the start node.
        goal (str): Key for the end node.
        data :  Data used in the custom search.
            Will be passed your data from load_data(graph).
            Default: None.

    Returns:
        The best path as a list from the start and goal nodes (including both).
    """

    # TODO: finish this function!
    raise NotImplementedError



def load_data(graph, time_left):
    """
    Feel free to implement this method. We'll call it only once 
    at the beginning of the Race, and we'll pass the output to your custom_search function.
    graph: a networkx graph
    time_left: function you can call to keep track of your remaining time.
        usage: time_left() returns the time left in milliseconds.
        the max time will be 10 minutes.

    * To get a list of nodes, use graph.nodes()
    * To get node neighbors, use graph.neighbors(node)
    * To get edge weight, use graph.get_edge_weight(node1, node2)
    """

    # nodes = graph.nodes()
    return None
 
def haversine_dist_heuristic(graph, v, goal):
    """
    Note: This provided heuristic is for the Atlanta race.

    Args:
        graph (ExplorableGraph): Undirected graph to search.
        v (str): Key for the node to calculate from.
        goal (str): Key for the end node to calculate to.

    Returns:
        Haversine distance between `v` node and `goal` node
    """

    #Load latitude and longitude coordinates in radians:
    vLatLong = (math.radians(graph.node[v]["pos"][0]), math.radians(graph.node[v]["pos"][1]))
    goalLatLong = (math.radians(graph.node[goal]["pos"][0]), math.radians(graph.node[goal]["pos"][1]))

    #Now we want to execute portions of the formula:
    constOutFront = 2*6371 #Radius of Earth is 6,371 kilometers
    term1InSqrt = (math.sin((goalLatLong[0]-vLatLong[0])/2))**2 #First term inside sqrt
    term2InSqrt = math.cos(vLatLong[0])*math.cos(goalLatLong[0])*((math.sin((goalLatLong[1]-vLatLong[1])/2))**2) #Second term
    return constOutFront*math.asin(math.sqrt(term1InSqrt+term2InSqrt)) #Straight application of formula
