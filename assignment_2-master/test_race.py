# coding=utf-8
import pickle
import unittest
import time

from submission import load_data, custom_search
from explorable_graph import ExplorableGraph

def get_time_milliseconds():
    return int(round(time.time() * 1000))


class TestRace(unittest.TestCase):

    def setUp(self):
        romania = pickle.load(open('romania_graph.pickle', 'rb'))
        self.romania = ExplorableGraph(romania)
        self.romania.reset_search()
        # you can also load atlanta graph like in unit tests.

    def test_call_load_data(self):

        max_time = 600*1000 # time in milliseconds
        start_time_ms = get_time_milliseconds()

        def time_left():
            return max_time - (get_time_milliseconds() - start_time_ms)
        
        data = load_data(self.romania, time_left)

        if time_left() < 0:
            self.fail(msg="You went over the maximum time for load_data.")

    def test_run_race(self):
        max_time = 600*1000 # time in milliseconds
        start_time_ms = get_time_milliseconds()

        def time_left():
            return max_time - (get_time_milliseconds() - start_time_ms)
        
        data = load_data(self.romania, time_left)

        start = 'a'
        goal = 'u'
        path = custom_search(self.romania, start, goal, data=data)

        

if __name__=='__main__':
    unittest.main()
