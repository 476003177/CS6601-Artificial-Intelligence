import numpy as np
import operator


def gaussian_prob(x, para_tuple):
    """Compute the probability of a given x value

    Args:
        x (float): observation value
        para_tuple (tuple): contains two elements, (mean, standard deviation)

    Return:
        Probability of seeing a value "x" in a Gaussian distribution.

    Note:
        We simplify the problem so you don't have to take care of integrals.
        Theoretically speaking, the returned value is not a probability of x,
        since the probability of any single value x from a continuous 
        distribution should be zero, instead of the number outputed here.
        By definition, the Gaussian percentile of a given value "x"
        is computed based on the "area" under the curve, from left-most to x. 
        The proability of getting value "x" is zero bcause a single value "x"
        has zero width, however, the probability of a range of value can be
        computed, for say, from "x - 0.1" to "x + 0.1".

    """
    if para_tuple == (None, None):
        return 0.0

    mean, std = para_tuple
    gaussian_percentile = (2 * np.pi * std**2)**-0.5 * \
                          np.exp(-(x - mean)**2 / (2 * std**2))
    return gaussian_percentile


def part_1_a():
    """Provide probabilities for the word HMMs outlined below.

    Word BUY, CAR, and HOUSE.

    Review Udacity Lesson 8 - Video #29. HMM Training

    Returns:
        tuple() of
        (prior probabilities for all states for word BUY,
         transition probabilities between states for word BUY,
         emission parameters tuple(mean, std) for all states for word BUY,
         prior probabilities for all states for word CAR,
         transition probabilities between states for word CAR,
         emission parameters tuple(mean, std) for all states for word CAR,
         prior probabilities for all states for word HOUSE,
         transition probabilities between states for word HOUSE,
         emission parameters tuple(mean, std) for all states for word HOUSE,)


        Sample Format (not complete):
        (
            {'B1': prob_of_starting_in_B1, 'B2': prob_of_starting_in_B2, ...},
            {'B1': {'B1': prob_of_transition_from_B1_to_B1,
                    'B2': prob_of_transition_from_B1_to_B2,
                    'B3': prob_of_transition_from_B1_to_B3,
                    'Bend': prob_of_transition_from_B1_to_Bend},
             'B2': {...}, ...},
            {'B1': tuple(mean_of_B1, standard_deviation_of_B1),
             'B2': tuple(mean_of_B2, standard_deviation_of_B2), ...},
            {'C1': prob_of_starting_in_C1, 'C2': prob_of_starting_in_C2, ...},
            {'C1': {'C1': prob_of_transition_from_C1_to_C1,
                    'C2': prob_of_transition_from_C1_to_C2,
                    'C3': prob_of_transition_from_C1_to_C3,
                    'Cend': prob_of_transition_from_C1_to_Cend},
             'C2': {...}, ...}
            {'C1': tuple(mean_of_C1, standard_deviation_of_C1),
             'C2': tuple(mean_of_C2, standard_deviation_of_C2), ...}
            {'H1': prob_of_starting_in_H1, 'H2': prob_of_starting_in_H2, ...},
            {'H1': {'H1': prob_of_transition_from_H1_to_H1,
                    'H2': prob_of_transition_from_H1_to_H2,
                    'H3': prob_of_transition_from_H1_to_H3,
                    'Hend': prob_of_transition_from_H1_to_Hend},
             'H2': {...}, ...}
            {'H1': tuple(mean_of_H1, standard_deviation_of_H1),
             'H2': tuple(mean_of_H2, standard_deviation_of_H2), ...}
        )
    """
    # TODO: complete this function.
    raise NotImplementedError()

    """Word BUY"""
    b_prior_probs = {
        'B1': 0.,
        'B2': 0.,
        'B3': 0.,
        'Bend': 0.,
    }
    b_transition_probs = {
        'B1': {'B1': 0., 'B2': 0., 'B3': 0., 'Bend': 0.},
        'B2': {'B1': 0., 'B2': 0., 'B3': 0., 'Bend': 0.},
        'B3': {'B1': 0., 'B2': 0., 'B3': 0., 'Bend': 0.},
        'Bend': {'B1': 0., 'B2': 0., 'B3': 0., 'Bend': 0.},
    }
    # Parameters for end state is not required
    b_emission_paras = {
        'B1': (None, None),
        'B2': (None, None),
        'B3': (None, None),
        'Bend': (None, None)
    }

    """Word CAR"""
    c_prior_probs = {
        'C1': 0.,
        'C2': 0.,
        'C3': 0.,
        'Cend': 0.,
    }
    c_transition_probs = {
        'C1': {'C1': 0., 'C2': 0., 'C3': 0., 'Cend': 0.},
        'C2': {'C1': 0., 'C2': 0., 'C3': 0., 'Cend': 0.},
        'C3': {'C1': 0., 'C2': 0., 'C3': 0., 'Cend': 0.},
        'Cend': {'C1': 0., 'C2': 0., 'C3': 0., 'Cend': 0.},
    }
    # Parameters for end state is not required
    c_emission_paras = {
        'C1': (None, None),
        'C2': (None, None),
        'C3': (None, None),
        'Cend': (None, None)
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.,
        'H2': 0.,
        'H3': 0.,
        'Hend': 0.,
    }
    # Probability of a state changing to another state.
    h_transition_probs = {
        'H1': {'H1': 0., 'H2': 0., 'H3': 0., 'Hend': 0.},
        'H2': {'H1': 0., 'H2': 0., 'H3': 0., 'Hend': 0.},
        'H3': {'H1': 0., 'H2': 0., 'H3': 0., 'Hend': 0.},
        'Hend': {'H1': 0., 'H2': 0., 'H3': 0., 'Hend': 0.},
    }
    # Parameters for end state is not required
    h_emission_paras = {
        'H1': (None, None),
        'H2': (None, None),
        'H3': (None, None),
        'Hend': (None, None)
    }

    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras,)


def viterbi(evidence_vector, states, prior_probs,
            transition_probs, emission_paras):
    """Viterbi Algorithm to calculate the most likely states give the evidence.

    Args:
        evidence_vector (list): List of right hand Y-axis positions (interger).

        states (list): List of all states in a word. No transition between words.
                       example: ['B1', 'B2', 'B3', 'Bend']

        prior_probs (dict): prior distribution for each state.
                            example: {'X1': 0.25,
                                      'X2': 0.25,
                                      'X3': 0.25,
                                      'Xend': 0.25}

        transition_probs (dict): dictionary representing transitions from each
                                 state to every other state.

        emission_paras (dict): parameters of Gaussian distribution 
                                from each state.

    Return:
        tuple of
        ( A list of states the most likely explains the evidence,
          probability this state sequence fits the evidence as a float )

    Note:
        You are required to use the function gaussian_prob to compute the
        emission probabilities.

    """
    
    # TODO: complete this function.
    raise NotImplementedError()

    sequence = []
    probability = 0.0

    return sequence, probability


def part_2_a():
    """Provide probabilities for the word HMMs outlined below.

    Now, at each time frame you are given with 2 observations (right hand Y
    position & left hand Y position). Use the result you derived in
    part_1_a, accompany with the provided probability for left hand, create
    a tuple of (right-y, left-y) to represent high-dimention transition & 
    emission probabilities.
    """

    # TODO: complete this function.
    raise NotImplementedError()

    """Word BUY"""
    b_prior_probs = {
        'B1': 0.,
        'B2': 0.,
        'B3': 0.,
        'Bend': 0.,
    }
    # example: {'B1': {'B1' : (right-hand Y, left-hand Y), ... }
    b_transition_probs = {
        'B1': {'B1': (0., 0.), 'B2': (0., 0.), 'B3': (0., 0.), 'Bend': (0., 0.)},
        'B2': {'B1': (0., 0.), 'B2': (0., 0.), 'B3': (0., 0.), 'Bend': (0., 0.)},
        'B3': {'B1': (0., 0.), 'B2': (0., 0.), 'B3': (0., 0.), 'Bend': (0., 0.)},
        'Bend': {'B1': (0., 0.), 'B2': (0., 0.), 'B3': (0., 0.), 'Bend': (0., 0.)},
    }
    # example: {'B1': [(right-mean, right-std), (left-mean, left-std)] ...}
    b_emission_paras = {
        'B1': [(None, None), (None, None)],
        'B2': [(None, None), (None, None)],
        'B3': [(None, None), (None, None)],
        'Bend': [(None, None), (None, None)]
    }

    """Word Car"""
    c_prior_probs = {
        'C1': 0.,
        'C2': 0.,
        'C3': 0.,
        'Cend': 0.,
    }
    c_transition_probs = {
        'C1': {'C1': (0., 0.), 'C2': (0., 0.), 'C3': (0., 0.), 'Cend': (0., 0.)},
        'C2': {'C1': (0., 0.), 'C2': (0., 0.), 'C3': (0., 0.), 'Cend': (0., 0.)},
        'C3': {'C1': (0., 0.), 'C2': (0., 0.), 'C3': (0., 0.), 'Cend': (0., 0.)},
        'Cend': {'C1': (0., 0.), 'C2': (0., 0.), 'C3': (0., 0.), 'Cend': (0., 0.)},
    }
    c_emission_paras = {
        'C1': [(None, None), (None, None)],
        'C2': [(None, None), (None, None)],
        'C3': [(None, None), (None, None)],
        'Cend': [(None, None), (None, None)]
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.,
        'H2': 0.,
        'H3': 0.,
        'Hend': 0.,
    }
    h_transition_probs = {
        'H1': {'H1': (0., 0.), 'H2': (0., 0.), 'H3': (0., 0.), 'Hend': (0., 0.)},
        'H2': {'H1': (0., 0.), 'H2': (0., 0.), 'H3': (0., 0.), 'Hend': (0., 0.)},
        'H3': {'H1': (0., 0.), 'H2': (0., 0.), 'H3': (0., 0.), 'Hend': (0., 0.)},
        'Hend': {'H1': (0., 0.), 'H2': (0., 0.), 'H3': (0., 0.), 'Hend': (0., 0.)},
    }
    h_emission_paras = {
        'H1': [(None, None), (None, None)],
        'H2': [(None, None), (None, None)],
        'H3': [(None, None), (None, None)],
        'Hend': [(None, None), (None, None)]
    }

    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras,)


def multidimensional_viterbi(evidence_vector, states, prior_probs,
                             transition_probs, emission_paras):
    """Decode the most likely word phrases generated by the evidence vector.

    States, prior_probs, transition_probs, and emission_probs will now contain
    all the words from part_2_a.
    """
    # TODO: complete this function.
    raise NotImplementedError()

    sequence = []
    probability = 0.0

    return sequence, probability


def MLLR_results():
    """Complete the MLLR adaptation process with the new adaptation data and 
     return the new emission parameters for each state.
    """
    # TODO: complete this function.


    b_emission_paras = {
        'B1': [(None, None), (None, None)],
        'B2': [(None, None), (None, None)],
        'B3': [(None, None), (None, None)],
        'Bend': [(None, None), (None, None)]
    }

    c_emission_paras = {
        'C1': [(None, None), (None, None)],
        'C2': [(None, None), (None, None)],
        'C3': [(None, None), (None, None)],
        'Cend': [(None, None), (None, None)]
    }

    h_emission_paras = {
        'H1': [(None, None), (None, None)],
        'H2': [(None, None), (None, None)],
        'H3': [(None, None), (None, None)],
        'Hend': [(None, None), (None, None)]
    }

    return (b_emission_paras, 
            c_emission_paras, 
            h_emission_paras)



def return_your_name():
    """Return your name
    """
    # TODO: finish this
    raise NotImplementedError()
