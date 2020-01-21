import unittest
import platform
import hashlib
from submission import part_1_a, part_2_a
from submission import viterbi, multidimensional_viterbi


if platform.system() == 'Windows':
    NIX = False
    print("Test on Windows system")
else:
    NIX = True
    print("Test on Linux/OS X system")

def print_success_message(test_case):
    print("UnitTest {0} passed successfully!".format(test_case))

class TestPart1a(unittest.TestCase):      
    def test_prior(self, part_1_a):
        (b_prior_probs, b_transition_probs, b_emission_paras,
        c_prior_probs, c_transition_probs, c_emission_paras,
        h_prior_probs, h_transition_probs, h_emission_paras) = part_1_a()

        b_prior = sum(b_prior_probs.values())
        c_prior = sum(c_prior_probs.values())
        h_prior = sum(h_prior_probs.values())
        total_prob = b_prior + c_prior + h_prior
        msg = ('incorrect prior probs. each word should be selected with '
               'equal probability. counted {}, should be 1').format(total_prob)
        self.assertAlmostEqual(1.0, total_prob, places=2, msg=msg)
        print_success_message("test_prior")
    

    def test_b_emission(self, part_1_a):
        (b_prior_probs, b_transition_probs, b_emission_paras,
        c_prior_probs, c_transition_probs, c_emission_paras,
        h_prior_probs, h_transition_probs, h_emission_paras) = part_1_a()
        
        mean, std = b_emission_paras['B1']
        mean_hash = hashlib.sha256(str.encode(str(mean))).hexdigest()
        std_hash = hashlib.sha256(str.encode(str(std))).hexdigest()
        msg = 'incorrect mean for letter BUY, state B1'
        self.assertEqual("fc12f049f5c759702b7bcd27d461fb57a4c9176bebfab3e4d15426a74c911d03", mean_hash, msg)
        msg = 'incorrect std for letter BUY, state B1'
        self.assertEqual("9b62d9c6eac8cbacdc2ccdfed1d60feb0716e2b39f5b94eac4bc69f803697ede", std_hash, msg)
        
        mean, std = b_emission_paras['B2']
        mean_hash = hashlib.sha256(str.encode(str(mean))).hexdigest()
        std_hash = hashlib.sha256(str.encode(str(std))).hexdigest()
        msg = 'incorrect mean for letter BUY, state B2'
        self.assertEqual("7a696b9ae0bc3323ce647c690106a78287ec2d5ce24ee5d11f48168bdb1a5dbb", mean_hash, msg)
        msg = 'incorrect std for letter BUY, state B2'
        self.assertEqual("8f3ff2d53dd528ebf1cccbb60667e2a1c0906da993de765634c01e6b5c85b34a", std_hash, msg)
        mean, std = b_emission_paras['B3']
        mean_hash = hashlib.sha256(str.encode(str(mean))).hexdigest()
        std_hash = hashlib.sha256(str.encode(str(std))).hexdigest()
        msg = 'incorrect mean for letter BUY, state B3'
        self.assertEqual("nc308fb57f5aa803bbcddeecac1e547d2b7010018758e72252cb7ceca298e4dbf", mean_hash, msg)
        msg = 'incorrect std for letter BUY, state B3'
        self.assertEqual("69f775cb8dc0f5d96d0c78826f813fc17b99018aee95f8d34e30f7e3f46743ba", std_hash, msg)
        
        print_success_message("test_b_emission")

    def test_c_emission(self, part_1_a):
        (b_prior_probs, b_transition_probs, b_emission_paras,
        c_prior_probs, c_transition_probs, c_emission_paras,
        h_prior_probs, h_transition_probs, h_emission_paras) = part_1_a()
        
        mean, std = c_emission_paras['C1']
        mean_hash = hashlib.sha256(str.encode(str(mean))).hexdigest()
        std_hash = hashlib.sha256(str.encode(str(std))).hexdigest()
        msg = 'incorrect mean for letter CAR, state C1'
        self.assertEqual("ece665f5d82dd6570657a9b11736924a97e327adc5ff314be890b2e565193f44", mean_hash, msg)
        msg = 'incorrect std for letter CAR, state C1'
        self.assertEqual("224edb71a15e864dff50b2224ca79bb5eb5179e4b287bdf3a54f6abec1f5be3e", std_hash, msg)

        mean, std = c_emission_paras['C2']
        mean_hash = hashlib.sha256(str.encode(str(mean))).hexdigest()
        std_hash = hashlib.sha256(str.encode(str(std))).hexdigest()
        msg = 'incorrect mean for letter CAR, state C2'
        self.assertEqual("e63f3d3e0ce127bb0afcca123bc00babd29820b15f53a7f9b6a31534a4fb0597", mean_hash, msg)
        msg = 'incorrect std for letter CAR, state C2'
        self.assertEqual("8f65223004a75f44404f485a1e84090699acef51f39de9411d6d9b377ae859a5", std_hash, msg)

        mean, std = c_emission_paras['C3']
        mean_hash = hashlib.sha256(str.encode(str(mean))).hexdigest()
        std_hash = hashlib.sha256(str.encode(str(std))).hexdigest()
        msg = 'incorrect mean for letter CAR, state C3'
        self.assertEqual("51489ee602434160b5c1cfc98a781353eb98db3b0fee064b951ba5baa4c9a014", mean_hash, msg)
        msg = 'incorrect std for letter CAR, state C3'
        self.assertEqual("6031bf9944ad15cdfcb096f4432643b7c097da0f179e7d584a016724d9338c98", std_hash, msg)

        print_success_message("test_c_emission")


    def test_h_emission(self, part_1_a):
        (b_prior_probs, b_transition_probs, b_emission_paras,
        c_prior_probs, c_transition_probs, c_emission_paras,
        h_prior_probs, h_transition_probs, h_emission_paras) = part_1_a()

        mean, std = h_emission_paras['H1']
        mean_hash = hashlib.sha256(str.encode(str(mean))).hexdigest()
        std_hash = hashlib.sha256(str.encode(str(std))).hexdigest()
        msg = 'incorrect mean for letter HOUSE, state H1'
        self.assertEqual("bbb4004ad949f0be888a1923e336d84485cbbace6641a94e09f6380fbc52b9ae", mean_hash, msg)
        msg = 'incorrect std for letter HOUSE, state H1'
        self.assertEqual("17c40ca95ab8e9107a4157365cb34646c64447a9f39cb4447176a736036495b3", std_hash, msg)

        mean, std = h_emission_paras['H2']
        mean_hash = hashlib.sha256(str.encode(str(mean))).hexdigest()
        std_hash = hashlib.sha256(str.encode(str(std))).hexdigest()
        msg = 'incorrect mean for letter HOUSE, state H2'
        self.assertEqual("9f9ac2449c421664c29f3e534b384d10900fe65fc2726941569a614a801f4b47", mean_hash, msg)
        msg = 'incorrect std for letter HOUSE, state H2'
        self.assertEqual("616a46cf184e50b2ff1debd938a19b3f112c2704f07985a3fe13f849bec48288", std_hash, msg)

        mean, std = h_emission_paras['H3']
        mean_hash = hashlib.sha256(str.encode(str(mean))).hexdigest()
        std_hash = hashlib.sha256(str.encode(str(std))).hexdigest()
        msg = 'incorrect mean for letter HOUSE, state H3'
        self.assertEqual("6d2e4be9b46ce8375256cf2bc5e2eb4c38a0fe2c6ae02f32a1e1955305cf3809", mean_hash, msg)
        msg = 'incorrect std for letter HOUSE, state H3'
        self.assertEqual("966d64084414dc3ce0e395a8ed417665a82b21e6f9858e4168d3578585042cc4", std_hash, msg)

        print_success_message("test_h_emission")


    def test_b_transition(self, part_1_a):
        (b_prior_probs, b_transition_probs, b_emission_paras,
        c_prior_probs, c_transition_probs, c_emission_paras,
        h_prior_probs, h_transition_probs, h_emission_paras) = part_1_a()

        for state, probs in b_transition_probs.items():
            checksum = sum(probs.values())
            msg = ('BUY transition prob in state {} '
                   'should sum to 1 (get {})').format(state, checksum)
            self.assertAlmostEqual(1.0, checksum, places=3, msg=msg)
        print_success_message("test_b_transition")

    def test_c_transition(self, part_1_a):
        (b_prior_probs, b_transition_probs, b_emission_paras,
        c_prior_probs, c_transition_probs, c_emission_paras,
        h_prior_probs, h_transition_probs, h_emission_paras) = part_1_a()

        for state, probs in c_transition_probs.items():
            checksum = sum(probs.values())
            msg = ('CAR transition prob in state {} should sum to 1 '
                   '(get {})').format(state, checksum)
            self.assertAlmostEqual(1.0, checksum, places=3, msg=msg)
        print_success_message("test_c_transition")

    def test_h_transition(self, part_1_a):
        (b_prior_probs, b_transition_probs, b_emission_paras,
        c_prior_probs, c_transition_probs, c_emission_paras,
        h_prior_probs, h_transition_probs, h_emission_paras) = part_1_a()

        for state, probs in h_transition_probs.items():
            checksum = sum(probs.values())
            msg = ('HOUSE transition prob in state {} should sum to 1 '
                   '(get {})').format(state, checksum)
            self.assertAlmostEqual(1.0, checksum, places=3, msg=msg)
        print_success_message("test_h_transition")

class TestPart1b(unittest.TestCase):

    def setup(self, part_1_a):
        b_states = ['B1', 'B2', 'B3', 'Bend']
        c_states = ['C1', 'C2', 'C3', 'Cend']
        h_states = ['H1', 'H2', 'H3', 'Hend']

        (b_prior_probs, b_transition_probs, b_emission_paras,
        c_prior_probs, c_transition_probs, c_emission_paras,
        h_prior_probs, h_transition_probs, h_emission_paras) = part_1_a()

        states = b_states + c_states + h_states
        prior = b_prior_probs
        prior.update(c_prior_probs)
        prior.update(h_prior_probs)

        trans = b_transition_probs
        trans.update(c_transition_probs)
        trans.update(h_transition_probs)

        emiss = b_emission_paras
        emiss.update(c_emission_paras)
        emiss.update(h_emission_paras)
        return states, prior, trans, emiss

    def test_viterbi_case1(self, part_1_a, viterbi):
        evidence = []
        states, prior, trans, emiss = self.setup(part_1_a)
        seq, prob = viterbi(evidence, states, prior, trans, emiss)
        msg = ('when evidence is an empty list, return "None" or [], '
                'get {}').format(seq)
        self.assertTrue(seq in [None, []], msg)
        msg = ('when evidence is an empty list, return prob=0.0, '
                'get {}').format(prob)
        self.assertTrue(prob == 0., msg)
        print_success_message("test_viterbi_case1")

    def test_viterbi_case2(self, part_1_a, viterbi):
        evidence = [50]
        prob_ans = 0.0167710014759
        seq_ans = ['H1']
        states, prior, trans, emiss = self.setup(part_1_a)
        seq, prob = viterbi(evidence, states, prior, trans, emiss)
        self.assertAlmostEqual(prob_ans, prob, places=7)
        self.assertEqual(seq_ans, seq)
        print_success_message("test_viterbi_case2")


    def test_viterbi_case3(self, part_1_a, viterbi):
        evidence = [40]
        prob_ans = 0.0392573919648
        seq_ans = ['B1']
        states, prior, trans, emiss = self.setup(part_1_a)
        seq, prob = viterbi(evidence, states, prior, trans, emiss)
        self.assertAlmostEqual(prob_ans, prob, places=7)
        self.assertEqual(seq_ans, seq)
        print_success_message("test_viterbi_case3")


    def test_viterbi_realsample1(self, part_1_a, viterbi):
        """
        Extracted from BOSTON ASL dataset: video 51, frame 17-23
        Actual words: BUY
        """
        evidence = [44, 51, 57, 63, 61, 60, 59]
        prob_ans = 3.81226460569e-11
        seq_ans = ['B1', 'B2', 'B2', 'B2', 'B2', 'B2', 'B2']
        states, prior, trans, emiss = self.setup(part_1_a)
        seq, prob = viterbi(evidence, states, prior, trans, emiss)
        self.assertAlmostEqual(prob_ans, prob, places=14)
        # print(seq)
        self.assertEqual(seq_ans, seq)
        print_success_message("test_viterbi_realsample1")


    def test_viterbi_realsample2(self, part_1_a, viterbi):
        """
        Extracted from BOSTON ASL dataset: video 48, frame 27-35
        Actual words: CAR
        """
        evidence = [45, 45, 46, 48, 51, 51, 49, 45, 42]
        prob_ans = 6.3377763423e-13
        seq_ans = ['H1', 'H1', 'H1', 'H1', 'H1', 'H1', 'H1', 'H1', 'H1']
        states, prior, trans, emiss = self.setup(part_1_a)
        seq, prob = viterbi(evidence, states, prior, trans, emiss)
        self.assertAlmostEqual(prob_ans, prob, places=17)
        self.assertEqual(seq_ans, seq)
        print_success_message("test_viterbi_realsample2")

    def test_viterbi_realsample3(self, part_1_a, viterbi):
        """
        Extracted from BOSTON ASL dataset: video 47, frame 21-30
        Actual words: CAR
        """
        evidence = [47, 39, 32, 34, 36, 42, 42, 42, 34, 25]
        prob_ans = 6.07498784245e-16
        seq_ans = ['H1', 'H2', 'H2', 'H2', 'H2', 'H2', 'H2', 'H2', 'H2', 'H2']
        states, prior, trans, emiss = self.setup(part_1_a)
        seq, prob = viterbi(evidence, states, prior, trans, emiss)
        self.assertAlmostEqual(prob_ans, prob, places=20)
        self.assertEqual(seq_ans, seq)
        print_success_message("test_viterbi_realsample3")


class TestPart2a(unittest.TestCase):

    def test_prior(self, part_2_a):
        (b_prior_probs, b_transition_probs, b_emission_paras,
        c_prior_probs, c_transition_probs, c_emission_paras,
        h_prior_probs, h_transition_probs, h_emission_paras) = part_2_a()

        b_prior = sum(b_prior_probs.values())
        c_prior = sum(c_prior_probs.values())
        h_prior = sum(h_prior_probs.values())
        total_prob = b_prior + c_prior + h_prior
        msg = ('incorrect prior probs. each word should be selected with '
               'equal probability. counted {}, should be 1').format(total_prob)
        self.assertAlmostEqual(1.0, total_prob, places=2, msg=msg)
        print_success_message("test_prior")

    def test_b_emission(self, part_2_a):
        (b_prior_probs, b_transition_probs, b_emission_paras,
        c_prior_probs, c_transition_probs, c_emission_paras,
        h_prior_probs, h_transition_probs, h_emission_paras) = part_2_a()

        right, left = b_emission_paras['B1']
        rmean, rstd = right
        meanhash = hashlib.sha256(str.encode(str(rmean))).hexdigest()
        stdhash = hashlib.sha256(str.encode(str(rstd))).hexdigest()
        msg = 'incorrect mean for letter BUY, state B1, right hand'
        self.assertEqual("fc12f049f5c759702b7bcd27d461fb57a4c9176bebfab3e4d15426a74c911d03", meanhash, msg)
        msg = 'incorrect std for letter BUY, state B1, right hand'
        self.assertEqual("9b62d9c6eac8cbacdc2ccdfed1d60feb0716e2b39f5b94eac4bc69f803697ede", stdhash, msg)

        lmean, lstd = left
        meanhash = hashlib.sha256(str.encode(str(lmean))).hexdigest()
        stdhash = hashlib.sha256(str.encode(str(lstd))).hexdigest()
        msg = 'incorrect mean for letter BUY, state B1, left hand'
        self.assertEqual("1f4ee82df2417fc57d14332c595a7ae797de19385789baf6eadbbd4ff4720247", meanhash, msg)
        msg = 'incorrect std for letter BUY, state B1, left hand'
        self.assertEqual("4c7e91f26921d132efdc4706898351940992e1e168e5e94d8348f9c2a1f0691d", stdhash, msg)

        right, left = b_emission_paras['B2']
        rmean, rstd = right
        meanhash = hashlib.sha256(str.encode(str(rmean))).hexdigest()
        stdhash = hashlib.sha256(str.encode(str(rstd))).hexdigest()
        msg = 'incorrect mean for letter BUY, state B2, right hand'
        self.assertEqual("7a696b9ae0bc3323ce647c690106a78287ec2d5ce24ee5d11f48168bdb1a5dbb", meanhash, msg)
        msg = 'incorrect std for letter BUY, state B2, right hand'
        self.assertEqual("8f3ff2d53dd528ebf1cccbb60667e2a1c0906da993de765634c01e6b5c85b34a", stdhash, msg)

        lmean, lstd = left
        meanhash = hashlib.sha256(str.encode(str(lmean))).hexdigest()
        stdhash = hashlib.sha256(str.encode(str(lstd))).hexdigest()
        msg = 'incorrect mean for letter BUY, state B2, left hand'
        self.assertEqual("6815431a97bf14732c6261331646b1893b4f63a6b63630337ab84b0258c22057", meanhash, msg)
        msg = 'incorrect std for letter BUY, state B2, left hand'
        self.assertEqual("d4e5ecf40ba5700a6c7c4a8ecac409c04f0bb0c85645e22e8a1899615637a649", stdhash, msg)

        right, left = b_emission_paras['B3']
        rmean, rstd = right
        meanhash = hashlib.sha256(str.encode(str(rmean))).hexdigest()
        stdhash = hashlib.sha256(str.encode(str(rstd))).hexdigest()
        msg = 'incorrect mean for letter BUY, state B3, right hand'
        self.assertEqual("c308fb57f5aa803bbcddeecac1e547d2b7010018758e72252cb7ceca298e4dbf", meanhash, msg)
        msg = 'incorrect std for letter BUY, state B3, right hand'
        self.assertEqual("69f775cb8dc0f5d96d0c78826f813fc17b99018aee95f8d34e30f7e3f46743ba", stdhash, msg)

        lmean, lstd = left
        meanhash = hashlib.sha256(str.encode(str(lmean))).hexdigest()
        stdhash = hashlib.sha256(str.encode(str(lstd))).hexdigest()
        msg = 'incorrect mean for letter BUY, state B3, left hand'
        self.assertEqual("cbda4febb59865566c436b19f528d1417b80ca56f310e236288d142d25e22579", meanhash, msg)
        msg = 'incorrect std for letter BUY, state B3, left hand'
        self.assertEqual("e20defec22b84bac3aecc30a9ff0be66f1e103102816f85f86fb996d1f1b2dfa", stdhash, msg)
        print_success_message("test_b_emission")

    def test_c_emission(self, part_2_a):
        (b_prior_probs, b_transition_probs, b_emission_paras,
        c_prior_probs, c_transition_probs, c_emission_paras,
        h_prior_probs, h_transition_probs, h_emission_paras) = part_2_a()
        
        right, left = c_emission_paras['C1']
        rmean, rstd = right
        meanhash = hashlib.sha256(str.encode(str(rmean))).hexdigest()
        stdhash = hashlib.sha256(str.encode(str(rstd))).hexdigest()
        msg = 'incorrect mean for letter CAR, state C1, right hand'
        self.assertEqual("ece665f5d82dd6570657a9b11736924a97e327adc5ff314be890b2e565193f44", meanhash, msg)
        msg = 'incorrect std for letter CAR, state C1, right hand'
        self.assertEqual("224edb71a15e864dff50b2224ca79bb5eb5179e4b287bdf3a54f6abec1f5be3e", stdhash, msg)

        lmean, lstd = left
        meanhash = hashlib.sha256(str.encode(str(lmean))).hexdigest()
        stdhash = hashlib.sha256(str.encode(str(lstd))).hexdigest()
        msg = 'incorrect mean for letter CAR, state C1, left hand'
        self.assertEqual("b29b3834c15f28f127d29b73b526f8130c28cd39be92ea77ea8c28cca393ac85", meanhash, msg)
        msg = 'incorrect std for letter CAR, state C1, left hand'
        self.assertEqual("6d891bd90529e527f5ec5f8e08c6ae6cbd2d306355bc7881eef4ce544a0dee3a", stdhash, msg)

        right, left = c_emission_paras['C2']
        rmean, rstd = right
        meanhash = hashlib.sha256(str.encode(str(rmean))).hexdigest()
        stdhash = hashlib.sha256(str.encode(str(rstd))).hexdigest()
        msg = 'incorrect mean for letter CAR, state C2, right hand'
        self.assertEqual("e63f3d3e0ce127bb0afcca123bc00babd29820b15f53a7f9b6a31534a4fb0597", meanhash, msg)
        msg = 'incorrect std for letter CAR, state C2, right hand'
        self.assertEqual("8f65223004a75f44404f485a1e84090699acef51f39de9411d6d9b377ae859a5", stdhash, msg)

        lmean, lstd = left
        meanhash = hashlib.sha256(str.encode(str(lmean))).hexdigest()
        stdhash = hashlib.sha256(str.encode(str(lstd))).hexdigest()
        msg = 'incorrect mean for letter CAR, state C2, left hand'
        self.assertEqual("77eb95b7f215142d12b91d70c5d8be5d587c975d9e3f42e0724b0dca7e4e0766", meanhash, msg)
        msg = 'incorrect std for letter CAR, state C2, left hand'
        self.assertEqual("96c333a16d2b49b250fa02ecf8418feb9e9717eb34d7eafa0fc394eb6ba8c716", stdhash, msg)

        right, left = c_emission_paras['C3']
        rmean, rstd = right
        meanhash = hashlib.sha256(str.encode(str(rmean))).hexdigest()
        stdhash = hashlib.sha256(str.encode(str(rstd))).hexdigest()
        msg = 'incorrect mean for letter CAR, state C3, right hand'
        self.assertEqual("51489ee602434160b5c1cfc98a781353eb98db3b0fee064b951ba5baa4c9a014", meanhash, msg)
        msg = 'incorrect std for letter CAR, state C3, right hand'
        self.assertEqual("6031bf9944ad15cdfcb096f4432643b7c097da0f179e7d584a016724d9338c98", stdhash, msg)

        lmean, lstd = left
        meanhash = hashlib.sha256(str.encode(str(lmean))).hexdigest()
        stdhash = hashlib.sha256(str.encode(str(lstd))).hexdigest()
        msg = 'incorrect mean for letter CAR, state C3, left hand'
        self.assertEqual("f97a13577367c1d604d37c4d2b6242d7193c7ba04aa4d1a64c322b23b2f9bd2a", meanhash, msg)
        msg = 'incorrect std for letter CAR, state C3, left hand'
        self.assertEqual("20c27ab7a8d707532010d73aa60cc00dee3dc9954c19f21c8ee33ca7d88bb730", stdhash, msg)
        print_success_message("test_c_emission")

    def test_h_emission(self, part_2_a):
        (b_prior_probs, b_transition_probs, b_emission_paras,
        c_prior_probs, c_transition_probs, c_emission_paras,
        h_prior_probs, h_transition_probs, h_emission_paras) = part_2_a()

        right, left = h_emission_paras['H1']
        rmean, rstd = right
        meanhash = hashlib.sha256(str.encode(str(rmean))).hexdigest()
        stdhash = hashlib.sha256(str.encode(str(rstd))).hexdigest()
        msg = 'incorrect mean for letter HOUSE, state H1, right hand'
        self.assertEqual("bbb4004ad949f0be888a1923e336d84485cbbace6641a94e09f6380fbc52b9ae", meanhash, msg)
        msg = 'incorrect std for letter HOUSE, state H1, right hand'
        self.assertEqual("17c40ca95ab8e9107a4157365cb34646c64447a9f39cb4447176a736036495b3", stdhash, msg)

        lmean, lstd = left
        meanhash = hashlib.sha256(str.encode(str(lmean))).hexdigest()
        stdhash = hashlib.sha256(str.encode(str(lstd))).hexdigest()
        msg = 'incorrect mean for letter HOUSE, state H1, left hand'
        self.assertEqual("e891ba4c3d29cea9a953ea4008101fc80b219043cd9661b5b5d880f20e8288e0", meanhash, msg)
        msg = 'incorrect std for letter HOUSE, state H1, left hand'
        self.assertEqual("0e3eae2394d599071880f59a4ed0143cccde7e8c3f7ab19cb73966244f60ae18", stdhash, msg)

        right, left = h_emission_paras['H2']
        rmean, rstd = right
        meanhash = hashlib.sha256(str.encode(str(rmean))).hexdigest()
        stdhash = hashlib.sha256(str.encode(str(rstd))).hexdigest()
        msg = 'incorrect mean for letter HOUSE, state H2, right hand'
        self.assertEqual("9f9ac2449c421664c29f3e534b384d10900fe65fc2726941569a614a801f4b47", meanhash, msg)
        msg = 'incorrect std for letter HOUSE, state H2, right hand'
        self.assertEqual("616a46cf184e50b2ff1debd938a19b3f112c2704f07985a3fe13f849bec48288", stdhash, msg)

        lmean, lstd = left
        meanhash = hashlib.sha256(str.encode(str(lmean))).hexdigest()
        stdhash = hashlib.sha256(str.encode(str(lstd))).hexdigest()
        msg = 'incorrect mean for letter HOUSE, state H2, left hand'
        self.assertEqual("fcedfbd43be4ecffdb561757c398a238394981da84eae34602c9befd905cd446", meanhash, msg)
        msg = 'incorrect std for letter HOUSE, state H2, left hand'
        self.assertEqual("bd35d613fab9688815eb6b23a666f04b97a416c67e10a40c68566beddbd1c8ec", stdhash, msg)


        right, left = h_emission_paras['H3']
        rmean, rstd = right
        meanhash = hashlib.sha256(str.encode(str(rmean))).hexdigest()
        stdhash = hashlib.sha256(str.encode(str(rstd))).hexdigest()
        msg = 'incorrect mean for letter HOUSE, state H3, right hand'
        self.assertEqual("6d2e4be9b46ce8375256cf2bc5e2eb4c38a0fe2c6ae02f32a1e1955305cf3809", meanhash, msg)
        msg = 'incorrect std for letter HOUSE, state H3, right hand'
        self.assertEqual("966d64084414dc3ce0e395a8ed417665a82b21e6f9858e4168d3578585042cc4", stdhash, msg)

        lmean, lstd = left
        meanhash = hashlib.sha256(str.encode(str(lmean))).hexdigest()
        stdhash = hashlib.sha256(str.encode(str(lstd))).hexdigest()
        msg = 'incorrect mean for letter HOUSE, state H3, left hand'
        self.assertEqual("c9510cd69628203b236c0bb44d3d8c419895abd2edd6bc054c680dc1858249b0", meanhash, msg)
        msg = 'incorrect std for letter HOUSE, state H3, left hand'
        self.assertEqual("23a7fa64659aebf7ba12073602c7822f66bda831cb26c177155d824fb349fb4d", stdhash, msg)
        print_success_message("test_h_emission")

    def test_b_transition(self, part_2_a):
        (b_prior_probs, b_transition_probs, b_emission_paras,
        c_prior_probs, c_transition_probs, c_emission_paras,
        h_prior_probs, h_transition_probs, h_emission_paras) = part_2_a()

        for state, probs in b_transition_probs.items():
            right, left = zip(*probs.values())

            msg = ('right hand BUY transition prob in state {} '
                   'should sum to 1 (get {})').format(state, sum(right))
            self.assertAlmostEqual(1.0, sum(right), places=2, msg=msg)
            msg = ('left hand BUY transition prob in state {} '
                   'should sum to 1 (get {})').format(state, sum(left))
            self.assertAlmostEqual(1.0, sum(left), places=2, msg=msg)
        print_success_message("test_b_transition")

    def test_c_transition(self, part_2_a):
        (b_prior_probs, b_transition_probs, b_emission_paras,
        c_prior_probs, c_transition_probs, c_emission_paras,
        h_prior_probs, h_transition_probs, h_emission_paras) = part_2_a()

        for state, probs in c_transition_probs.items():
            right, left = zip(*probs.values())

            msg = ('right hand CAR transition prob in state {} '
                   'should sum to 1 (get {})').format(state, sum(right))
            self.assertAlmostEqual(1.0, sum(right), places=2, msg=msg)
            msg = ('left hand CAR transition prob in state {} '
                   'should sum to 1 (get {})').format(state, sum(left))
            self.assertAlmostEqual(1.0, sum(left), places=2, msg=msg)
        print_success_message("test_c_transition")

    def test_h_transition(self, part_2_a):
        (b_prior_probs, b_transition_probs, b_emission_paras,
        c_prior_probs, c_transition_probs, c_emission_paras,
        h_prior_probs, h_transition_probs, h_emission_paras) = part_2_a()
        
        for state, probs in h_transition_probs.items():
            right, left = zip(*probs.values())

            msg = ('right hand HOUSE transition prob in state {} '
                   'should sum to 1 (get {})').format(state, sum(right))
            self.assertAlmostEqual(1.0, sum(right), places=2, msg=msg)
            msg = ('left hand HOUSE transition prob in state {} '
                   'should sum to 1 (get {})').format(state, sum(left))
            self.assertAlmostEqual(1.0, sum(left), places=2, msg=msg)
        print_success_message("test_h_transition")

class TestPart2b(unittest.TestCase):
    def setup(self, part_2_a):
        b_states = ['B1', 'B2', 'B3', 'Bend']
        c_states = ['C1', 'C2', 'C3', 'Cend']
        h_states = ['H1', 'H2', 'H3', 'Hend']

        (b_prior_probs, b_transition_probs, b_emission_paras,
        c_prior_probs, c_transition_probs, c_emission_paras,
        h_prior_probs, h_transition_probs, h_emission_paras) = part_2_a()

        states = b_states + c_states + h_states
        prior = b_prior_probs
        prior.update(c_prior_probs)
        prior.update(h_prior_probs)

        trans = b_transition_probs
        trans.update(c_transition_probs)
        trans.update(h_transition_probs)

        emiss = b_emission_paras
        emiss.update(c_emission_paras)
        emiss.update(h_emission_paras)
        return states, prior, trans, emiss

    def test_viterbi_case1(self, part_2_a, multidimensional_viterbi):
        evidence = []
        states, prior, trans, emiss = self.setup(part_2_a)
        seq, prob = multidimensional_viterbi(evidence,
                            states,
                            prior,
                            trans,
                            emiss)
        msg = ('when evidence is an empty list, return "None" or [], '
                'get {}').format(seq)
        self.assertTrue(seq in [None, []], msg)
        msg = ('when evidence is an empty list, return prob=0.0, '
                'get {}').format(prob)
        self.assertTrue(prob == 0., msg)

        print_success_message("test_viterbi_case1")

    def test_viterbi_case2(self, part_2_a, multidimensional_viterbi):

        evidence = [(50, 100)]
        prob_ans = 1.18078362571e-05
        seq_ans = ['B1']

        states, prior, trans, emiss = self.setup(part_2_a)

        seq, prob = multidimensional_viterbi(evidence,
                                            states,
                                            prior,
                                            trans,
                                            emiss)
        self.assertAlmostEqual(prob_ans, prob, places=9)
        self.assertEqual(seq_ans, seq)

        print_success_message("test_viterbi_case2")

    def test_viterbi_case3(self, part_2_a, multidimensional_viterbi):
        evidence = [(40, 40)]
        prob_ans = 0.000213189518752
        seq_ans = ['C1']

        states, prior, trans, emiss = self.setup(part_2_a)

        seq, prob = multidimensional_viterbi(evidence,
                                            states,
                                            prior,
                                            trans,
                                            emiss)
        self.assertAlmostEqual(prob_ans, prob, places=15)
        self.assertEqual(seq_ans, seq)

        print_success_message("test_viterbi_case3")

    def test_viterbi_realsample1(self, part_2_a, multidimensional_viterbi):
        """
        Extracted from BOSTON ASL dataset: video 51, frame 17-23
        Actual words: BUY
        """
        right_y = [44, 51, 57, 63, 61, 60, 59]
        left_y = [101, 95, 84, 77, 73, 68, 66]
        evidence = list(zip(right_y, left_y))

        prob_ans = 4.1704852628957851e-27
        seq_ans = ['B1', 'B1', 'B2', 'B2', 'B3', 'B3', 'B3']

        states, prior, trans, emiss = self.setup(part_2_a)

        seq, prob = multidimensional_viterbi(evidence,
                                            states,
                                            prior,
                                            trans,
                                            emiss)
        self.assertAlmostEqual(prob_ans, prob, places=32)
        self.assertEqual(seq_ans, seq)

        print_success_message("test_viterbi_realsample1")

    def test_viterbi_realsample2(self, part_2_a, multidimensional_viterbi):
        """
        Extracted from BOSTON ASL dataset: video 48, frame 27-35
        Actual words: CAR
        """
        right_y = [45, 45, 46, 48, 51, 51, 49, 45, 42]
        left_y = [47, 45, 43, 45, 48, 48, 54, 61, 67]
        evidence = list(zip(right_y, left_y))

        prob_ans = 5.80105906918e-28
        seq_ans = ['H1', 'H1', 'H1', 'H1', 'H1', 'H1', 'H1', 'H1', 'H1']

        states, prior, trans, emiss = self.setup(part_2_a)

        seq, prob = multidimensional_viterbi(evidence,
                                            states,
                                            prior,
                                            trans,
                                            emiss)
        self.assertAlmostEqual(prob_ans, prob, places=32)
        self.assertEqual(seq_ans, seq)

        print_success_message("test_viterbi_realsample2")

    def test_viterbi_realsample3(self, part_2_a, multidimensional_viterbi):
        """
        Extracted from BOSTON ASL dataset: video 47, frame 21-30
        Actual words: CAR
        """
        right_y = [47, 39, 32, 34, 36, 42, 42, 42, 34, 25]
        left_y = [67, 62, 56, 48, 43, 43, 44, 46, 52, 65]
        evidence = list(zip(right_y, left_y))

        prob_ans = 1.17739941205e-34
        seq_ans = ['C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1', 'C1']

        states, prior, trans, emiss = self.setup(part_2_a)

        seq, prob = multidimensional_viterbi(evidence,
                                            states,
                                            prior,
                                            trans,
                                            emiss)
        self.assertAlmostEqual(prob_ans, prob, places=38)
        self.assertEqual(seq_ans, seq)

        print_success_message("test_viterbi_realsample3")

    def test_viterbi_phrase1(self, part_2_a, multidimensional_viterbi):
        """
        Extracted from BOSTON ASL dataset: video 45, frame 36-56
        Actual words: BUY HOUSE
        """
        right_y = [48, 53, 54, 53, 48, 48, 48, 53, 48, 48, 46, 46,
                    53, 58, 66, 72, 71, 76, 76, 76, 76]
        left_y = [85, 74, 70, 66, 64, 64, 64, 60, 60, 60, 50, 50,
                    57, 67, 74, 85, 84, 89, 93, 93, 94]
        evidence = list(zip(right_y, left_y))

        prob_ans = 3.838599461219723e-75

        states, prior, trans, emiss = self.setup(part_2_a)

        seq_ans = ['B1', 'B2', 'B3', 'B3', 'B3', 'B3', 'B3',
                    'B3', 'H1', 'H1','H1', 'H1', 'H2', 'H3',
                    'H3', 'H3', 'H3', 'H3', 'H3', 'H3', 'H3']
        seq, prob = multidimensional_viterbi(evidence,
                                            states,
                                            prior,
                                            trans,
                                            emiss)
        self.assertAlmostEqual(prob_ans, prob, places=77)
        self.assertEqual(seq_ans, seq)

        print_success_message("test_viterbi_phrase1")

    def test_viterbi_phrase2(self, part_2_a, multidimensional_viterbi):
        """
        Extracted from BOSTON ASL dataset: video 47, frame 12-30
        Actual words: BUY CAR
        """
        right_y = [42, 40, 41, 43, 52, 55, 59, 60, 55, 47, 39, 32,
                    34, 36, 42, 42, 42, 34, 25]
        left_y = [138, 133, 123, 115, 104, 91, 76, 70, 67, 67, 62,
                    56, 48, 43, 43, 44, 46, 52, 65]
        evidence = list(zip(right_y, left_y))

        prob_ans = 8.3138718215932126e-71

        states, prior, trans, emiss = self.setup(part_2_a)

        seq_ans = ['B1', 'B1', 'B1', 'B1', 'B1', 'B1', 'B2',
                    'B3','B3', 'B3', 'C1', 'C1', 'C1', 'C1',
                    'C1', 'C1', 'C1', 'C1', 'C1']
        seq, prob = multidimensional_viterbi(evidence,
                                            states,
                                            prior,
                                            trans,
                                            emiss)
        self.assertAlmostEqual(prob_ans, prob, places=71)
        self.assertEqual(seq_ans, seq)

        print_success_message("test_viterbi_phrase2")
if __name__ == "__main__":
     unittest.main(verbosity=2)
