import os
from helper_functions import image_to_matrix, matrix_to_image, \
    flatten_image_matrix
import numpy as np

from helper_functions import image_difference, default_convergence

import unittest


def print_success_message():
    print("UnitTest passed successfully!")


def generate_test_mixture(data_size, means, variances, mixing_coefficients):
    """
    Generate synthetic test
    data for a GMM based on
    fixed means, variances and
    mixing coefficients.

    params:
    data_size = (int)
    means = [float]
    variances = [float]
    mixing_coefficients = [float]

    returns:
    data = np.array[float]
    """

    data = np.zeros(data_size)

    indices = np.random.choice(len(means), len(data), p=mixing_coefficients)

    for i in range(len(indices)):
        val = np.random.normal(means[indices[i]], variances[indices[i]])
        while val <= 0:
            val = np.random.normal(means[indices[i]], variances[indices[i]])
        data[i] = val

    return data


class K_means_test(unittest.TestCase):
    def runTest(self):
        pass

    def test_initial_means(self, initial_means):
        image_file = 'images/bird_color_24.png'
        image_values = image_to_matrix(image_file).reshape(-1, 3)
        m, n = image_values.shape
        for k in range(1, 10):
            means = initial_means(image_values, k)
            self.assertEqual(means.shape, (k, n),
                             msg=("Initialization for %d dimensional array "
                                  "with %d clusters returned an matrix of an incompatible dimension.") % (n, k))
            for mean in means:
                self.assertTrue(any(np.equal(image_values, mean).all(1)),
                                msg=("Means should be points from given array"))
        print_success_message()

    def test_k_means_step(self, k_means_step):
        initial_means = [
            np.array([[0.90980393, 0.8392157, 0.65098041],
                      [0.83137256, 0.80784315, 0.69411767]]),
            np.array([[0.90980393, 0.8392157, 0.65098041],
                      [0.83137256, 0.80784315, 0.69411767],
                      [0.67450982, 0.52941179, 0.25490198]]),
            np.array([[0.90980393, 0.8392157, 0.65098041],
                      [0.83137256, 0.80784315, 0.69411767],
                      [0.67450982, 0.52941179, 0.25490198],
                      [0.86666667, 0.8392157, 0.70588237]]),
            np.array([[0.90980393, 0.8392157, 0.65098041],
                      [0.83137256, 0.80784315, 0.69411767],
                      [0.67450982, 0.52941179, 0.25490198],
                      [0.86666667, 0.8392157, 0.70588237], [0, 0, 0]]),
            np.array([[0.90980393, 0.8392157, 0.65098041],
                      [0.83137256, 0.80784315, 0.69411767],
                      [0.67450982, 0.52941179, 0.25490198],
                      [0.86666667, 0.8392157, 0.70588237], [0, 0, 0],
                      [0.8392157, 0.80392158, 0.63921571]]),
        ]

        expected_new_means = [
            np.array([[0.954027, 0.46203843, 0.22550906],
                      [0.27400663, 0.28729942, 0.40885332]]),
            np.array([[0.93051314, 0.88417423, 0.36199632],
                      [0.4807488, 0.57399386, 0.92681587],
                      [0.4355678, 0.27258742, 0.29021215]]),
            np.array([[0.93003845, 0.88480246, 0.35687384],
                      [0.46192682, 0.55563807, 0.92357367],
                      [0.4355678, 0.27258742, 0.29021215],
                      [0.81017733, 0.87135339, 0.95991331]]),
            np.array([[0.93003845, 0.88480246, 0.35687384],
                      [0.46192682, 0.55563807, 0.92357367],
                      [0.62516654, 0.35561383, 0.28908807],
                      [0.81017733, 0.87135339, 0.95991331],
                      [0.12563141, 0.13679598, 0.29207909]]),
            np.array([[0.936167, 0.94245672, 0.33339596],
                      [0.4587802, 0.55248123, 0.93013507],
                      [0.62435812, 0.35411939, 0.28870562],
                      [0.81017733, 0.87135339, 0.95991331],
                      [0.12563141, 0.13679598, 0.29207909],
                      [0.88033324, 0.72848517, 0.42817175]])
        ]

        expected_cluster_sums = [70545, 176588, 177891, 241253, 250122]

        k_min = 2
        k_max = 6
        image_file = 'images/bird_color_24.png'
        image_values = image_to_matrix(image_file).reshape(-1, 3)
        m, n = image_values.shape
        for i, k in enumerate(range(k_min, k_max + 1)):
            new_means, new_clusters = k_means_step(image_values, k=k, means=initial_means[k - k_min])
            self.assertTrue(new_means.shape == initial_means[k - k_min].shape,
                            msg="New means array are of an incorrect shape. Expected: %s got: %s" %
                                (initial_means[k - k_min].shape, new_means.shape))
            self.assertTrue(new_clusters.shape[0] == m,
                            msg="New clusters array are of an incorrect shape. Expected: %s got: %s" %
                                (m, new_clusters.shape))
            self.assertTrue(np.allclose(new_means, expected_new_means[i]),
                            msg="Incorrect new mean values.")
            self.assertTrue(np.sum(new_clusters) == expected_cluster_sums[i],
                            msg="Incorrect clusters prediction.")
        print_success_message()

    def test_k_means(self, k_means_cluster):
        """
        Testing your implementation
        of k-means on the segmented
        bird_color_24 reference images.
        """
        k_min = 2
        k_max = 6
        image_dir = 'images/'
        image_name = 'bird_color_24.png'
        image_values = image_to_matrix(image_dir + image_name)
        # initial mean for each k value
        initial_means = [
            np.array([[0.90980393, 0.8392157, 0.65098041],
                      [0.83137256, 0.80784315, 0.69411767]]),
            np.array([[0.90980393, 0.8392157, 0.65098041],
                      [0.83137256, 0.80784315, 0.69411767],
                      [0.67450982, 0.52941179, 0.25490198]]),
            np.array([[0.90980393, 0.8392157, 0.65098041],
                      [0.83137256, 0.80784315, 0.69411767],
                      [0.67450982, 0.52941179, 0.25490198],
                      [0.86666667, 0.8392157, 0.70588237]]),
            np.array([[0.90980393, 0.8392157, 0.65098041],
                      [0.83137256, 0.80784315, 0.69411767],
                      [0.67450982, 0.52941179, 0.25490198],
                      [0.86666667, 0.8392157, 0.70588237], [0, 0, 0]]),
            np.array([[0.90980393, 0.8392157, 0.65098041],
                      [0.83137256, 0.80784315, 0.69411767],
                      [0.67450982, 0.52941179, 0.25490198],
                      [0.86666667, 0.8392157, 0.70588237], [0, 0, 0],
                      [0.8392157, 0.80392158, 0.63921571]]),
        ]
        # test different k values to find best
        for k in range(k_min, k_max + 1):
            updated_values = k_means_cluster(image_values, k,
                                             initial_means[k - k_min])
            ref_image = image_dir + 'k%d_%s' % (k, image_name)
            ref_values = image_to_matrix(ref_image)
            dist = image_difference(updated_values, ref_values)
            self.assertEqual(int(dist), 0, msg=("Clustering for %d clusters"
                                                + "produced unrealistic image segmentation.") % k)
        print_success_message()


class GMMTests(unittest.TestCase):
    def runTest(self):
        pass

    def test_gmm_initialization(self, initialize_parameters):
        """Testing the GMM method
        for initializing the training"""
        image_file = 'images/bird_color_24.png'
        image_matrix = image_to_matrix(image_file)
        image_matrix = image_matrix.reshape(-1, 3)
        m, n = image_matrix.shape
        num_components = 5
        np.random.seed(0)
        means, variances, mixing_coefficients = initialize_parameters(image_matrix, num_components)
        self.assertTrue(variances.shape == (num_components, n, n),
                        msg="Incorrect variance dimensions")
        self.assertTrue(means.shape == (num_components, n),
                        msg="Incorrect mean dimensions")
        for mean in means:
            self.assertTrue(any(np.equal(image_matrix, mean).all(1)),
                                    msg=("Means should be points from given array"))
        self.assertTrue(mixing_coefficients.sum() == 1,
                        msg="Incorrect mixing coefficients, make all coefficient sum to 1")
        print_success_message()

    def test_gmm_covariance(self, compute_sigma):
        ''' Testing implementation of covariance matrix
        computation explicitly'''
        image_file = 'images/bird_color_24.png'
        image_matrix = image_to_matrix(image_file)
        image_matrix = image_matrix.reshape(-1, 3)
        m, n = image_matrix.shape
        num_components = 5
        MU = np.array([[0.64705884, 0.7490196,  0.7058824 ],
                         [0.98039216, 0.3019608,  0.14509805],
                         [0.3764706,  0.39215687, 0.28627452],
                         [0.2784314,  0.26666668, 0.23921569],
                         [0.16078432, 0.15294118, 0.30588236]])
        SIGMA = np.array([[[ 0.15471499,  0.11200016,  0.04393127],
                          [ 0.11200016,  0.22953323,  0.16426138],
                          [ 0.04393127,  0.16426138,  0.19807944]],
                         [[ 0.38481037,  0.0204306,  -0.12658471],
                          [ 0.0204306,   0.06127004,  0.02783406],
                          [-0.12658471,  0.02783406,  0.12057389]],
                         [[ 0.13134574,  0.03346525, -0.01198761],
                          [ 0.03346525,  0.06303026,  0.01652109],
                          [-0.01198761,  0.01652109,  0.08084702]],
                         [[ 0.15901856,  0.05194932,  0.00383432],
                          [ 0.05194932,  0.06501033,  0.02864332],
                          [ 0.00383432,  0.02864332,  0.08966025]],
                         [[ 0.21760082,  0.09526375, -0.00290086],
                          [ 0.09526375,  0.09400968,  0.02967787],
                          [-0.00290086,  0.02967787,  0.07848203]]])
        self.assertTrue(np.allclose(SIGMA, compute_sigma(image_matrix, MU)),
                        msg="Incorrect covariance matrix.")
        print_success_message()

    def test_gmm_prob(self, prob):
        """Testing the GMM method
        for calculating the probability
        of a given point belonging to a
        component.
        returns:
        prob = float
        """

        image_file = 'images/bird_color_24.png'
        image_matrix = image_to_matrix(image_file)
        image_matrix = image_matrix.reshape(-1, 3)
        m, n = image_matrix.shape
        mean = np.array([0.0627451, 0.10980392, 0.54901963])
        covariance = np.array([[0.28756526, 0.13084501, -0.09662368],
                               [0.13084501, 0.11177602, -0.02345659],
                               [-0.09662368, -0.02345659, 0.11303925]])
        p = prob(image_matrix[0], mean, covariance)
        self.assertEqual(round(p, 5), 0.57693,
                         msg="Incorrect probability value returned.")
        print_success_message()

    def test_gmm_e_step(self, E_step):
        """Testing the E-step implementation

        returns:
        r = numpy.ndarray[numpy.ndarray[float]]
        """
        image_file = 'images/bird_color_24.png'
        image_matrix = image_to_matrix(image_file)
        image_matrix = image_matrix.reshape(-1, 3)
        num_components = 5
        m, n = image_matrix.shape
        means = np.array([[0.34901962, 0.3647059, 0.30588236],
                          [0.9882353, 0.3254902, 0.19607843],
                          [1., 0.6117647, 0.5019608],
                          [0.37254903, 0.3882353, 0.2901961],
                          [0.3529412, 0.40784314, 1.]])
        covariances = np.array([[[0.13715639, 0.03524152, -0.01240736],
                                 [0.03524152, 0.06077217, 0.01898307],
                                 [-0.01240736, 0.01898307, 0.07848206]],

                                [[0.3929004, 0.03238055, -0.10174976],
                                 [0.03238055, 0.06016063, 0.02226048],
                                 [-0.10174976, 0.02226048, 0.10162983]],

                                [[0.40526569, 0.18437279, 0.05891556],
                                 [0.18437279, 0.13535137, 0.0603222],
                                 [0.05891556, 0.0603222, 0.09712359]],

                                [[0.13208355, 0.03362673, -0.01208926],
                                 [0.03362673, 0.06261538, 0.01699577],
                                 [-0.01208926, 0.01699577, 0.08031248]],

                                [[0.13623408, 0.03036055, -0.09287403],
                                 [0.03036055, 0.06499729, 0.06576895],
                                 [-0.09287403, 0.06576895, 0.49017089]]])
        pis = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        r = E_step(image_matrix, means, covariances, pis, num_components)
        expected_r_rows = np.array([25262.04787326, 18961.31887563, 14991.17253041, 24783.52164336, 14821.93907735])
        self.assertEqual(round(r.sum()), m,
                         msg="Incorrect responsibility values, sum of all elements must be equal to m.")
        self.assertTrue(np.allclose(r.sum(axis=0), 1),
                        msg="Incorrect responsibility values, columns are not normalized.")
        self.assertTrue(np.allclose(r.sum(axis=1), expected_r_rows),
                        msg="Incorrect responsibility values, rows are not normalized.")
        print_success_message()

    def test_gmm_m_step(self, M_step):
        """Testing the M-step implementation

        returns:
        pi = numpy.ndarray[]
        mu = numpy.ndarray[numpy.ndarray[float]]
        sigma = numpy.ndarray[numpy.ndarray[numpy.ndarray[float]]]
        """
        image_file = 'images/bird_color_24.png'
        image_matrix = image_to_matrix(image_file)
        image_matrix = image_matrix.reshape(-1, 3)
        num_components = 3

        r = np.array([[0.51660555, 0.52444999, 0.50810777, 0.51151982, 0.4997758,
                       0.51134715, 0.4997758, 0.49475051, 0.48168621, 0.47946386],
                      [0.10036031, 0.09948503, 0.1052672, 0.10687822, 0.11345191,
                       0.10697943, 0.11345191, 0.11705775, 0.11919758, 0.12314451],
                      [0.38303414, 0.37606498, 0.38662503, 0.38160197, 0.3867723,
                       0.38167342, 0.3867723, 0.38819173, 0.39911622, 0.39739164]])
        mu, sigma, pi = M_step(image_matrix[:10], r, num_components)
        expected_PI = np.array([0.50274825, 0.11052739, 0.38672437])
        expected_MU = np.array([[0.12401373, 0.12246745, 0.11884939],
                                [0.12509098, 0.12350831, 0.12009721],
                                [0.1244816, 0.12288793, 0.11943994]])
        expected_SIGMA = np.array([[[0.00014082, 0.00011489, 0.00013914],
                                    [0.00011489, 0.00014875, 0.00013629],
                                    [0.00013914, 0.00013629, 0.00017721]],
                                   [[0.00014278, 0.00011441, 0.00014151],
                                    [0.00011441, 0.00014355, 0.00013533],
                                    [0.00014151, 0.00013533, 0.00018113]],
                                   [[0.00014206, 0.0001155, 0.00014097],
                                    [0.0001155, 0.00014746, 0.00013691],
                                    [0.00014097, 0.00013691, 0.00018029]]])
        self.assertTrue(np.allclose(pi, expected_PI),
                        msg="Incorrect new coefficient matrix.")
        self.assertTrue(np.allclose(mu, expected_MU),
                        msg="Incorrect new means matrix.")
        self.assertTrue(np.allclose(sigma, expected_SIGMA),
                        msg="Incorrect new covariance matrix.")
        print_success_message()

    def test_gmm_likelihood(self, likelihood):
        """Testing the GMM method
        for calculating the overall
        model probability.
        Should return -364370.

        returns:
        likelihood = float
        """

        image_file = 'images/bird_color_24.png'
        image_matrix = image_to_matrix(image_file)
        image_matrix = image_matrix.reshape(-1, 3)
        num_components = 5
        m, n = image_matrix.shape
        means = np.array([[0.34901962, 0.3647059, 0.30588236],
                          [0.9882353, 0.3254902, 0.19607843],
                          [1., 0.6117647, 0.5019608],
                          [0.37254903, 0.3882353, 0.2901961],
                          [0.3529412, 0.40784314, 1.]])
        covariances = np.array([[[0.13715639, 0.03524152, -0.01240736],
                                 [0.03524152, 0.06077217, 0.01898307],
                                 [-0.01240736, 0.01898307, 0.07848206]],

                                [[0.3929004, 0.03238055, -0.10174976],
                                 [0.03238055, 0.06016063, 0.02226048],
                                 [-0.10174976, 0.02226048, 0.10162983]],

                                [[0.40526569, 0.18437279, 0.05891556],
                                 [0.18437279, 0.13535137, 0.0603222],
                                 [0.05891556, 0.0603222, 0.09712359]],

                                [[0.13208355, 0.03362673, -0.01208926],
                                 [0.03362673, 0.06261538, 0.01699577],
                                 [-0.01208926, 0.01699577, 0.08031248]],

                                [[0.13623408, 0.03036055, -0.09287403],
                                 [0.03036055, 0.06499729, 0.06576895],
                                 [-0.09287403, 0.06576895, 0.49017089]]])
        pis = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        lkl = likelihood(image_matrix, pis, means, covariances, num_components)
        self.assertEqual(np.round(lkl), -23943.0,
                         msg="Incorrect likelihood value returned")
        # expected_lkl =
        print_success_message()

    def test_gmm_train(self, train_model, likelihood):
        """Test the training
        procedure for GMM.

        returns:
        gmm = GaussianMixtureModel
        """
        image_file = 'images/bird_color_24.png'
        image_matrix = image_to_matrix(image_file)
        image_matrix = image_matrix.reshape(-1, 3)
        num_components = 5
        m, n = image_matrix.shape

        means = np.array([[0.34901962, 0.3647059, 0.30588236],
                          [0.9882353, 0.3254902, 0.19607843],
                          [1., 0.6117647, 0.5019608],
                          [0.37254903, 0.3882353, 0.2901961],
                          [0.3529412, 0.40784314, 1.]])
        covariances = np.array([[[0.13715639, 0.03524152, -0.01240736],
                                 [0.03524152, 0.06077217, 0.01898307],
                                 [-0.01240736, 0.01898307, 0.07848206]],

                                [[0.3929004, 0.03238055, -0.10174976],
                                 [0.03238055, 0.06016063, 0.02226048],
                                 [-0.10174976, 0.02226048, 0.10162983]],

                                [[0.40526569, 0.18437279, 0.05891556],
                                 [0.18437279, 0.13535137, 0.0603222],
                                 [0.05891556, 0.0603222, 0.09712359]],

                                [[0.13208355, 0.03362673, -0.01208926],
                                 [0.03362673, 0.06261538, 0.01699577],
                                 [-0.01208926, 0.01699577, 0.08031248]],

                                [[0.13623408, 0.03036055, -0.09287403],
                                 [0.03036055, 0.06499729, 0.06576895],
                                 [-0.09287403, 0.06576895, 0.49017089]]])
        pis = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        initial_lkl = likelihood(image_matrix, pis, means, covariances, num_components)
        MU, SIGMA, PI, r = train_model(image_matrix, num_components,
                                       convergence_function=default_convergence,
                                       initial_values=(means, covariances, pis))
        final_lkl = likelihood(image_matrix, PI, MU, SIGMA, num_components)
        likelihood_difference = final_lkl - initial_lkl
        likelihood_thresh = 90000
        diff_check = likelihood_difference >= likelihood_thresh
        self.assertTrue(diff_check, msg=("Model likelihood increased by less"
                                         " than %d for a two-mean mixture" % likelihood_thresh))

        print_success_message()

    def test_gmm_segment(self, train_model, segment):
        """
        Apply the trained GMM
        to unsegmented image and
        generate a segmented image.

        returns:
        segmented_matrix = numpy.ndarray[numpy.ndarray[float]]
        """
        image_file = 'images/bird_color_24.png'
        image_matrix = image_to_matrix(image_file).reshape(-1, 3)
        num_components = 5

        MU, SIGMA, PI, r = train_model(image_matrix, num_components,
                                       convergence_function=default_convergence)

        segment = segment(image_matrix, MU, num_components, r)

        segment_num_components = len(np.unique(segment, axis=0))
        self.assertTrue(segment_num_components == r.shape[0],
                        msg="Incorrect number of image segments produced")
        segment_sort = np.sort(np.unique(segment, axis=0), axis=0)
        mu_sort = np.sort(MU, axis=0)
        self.assertTrue((segment_sort == mu_sort).all(),
                        msg="Incorrect segment values. Should be be MU values")
        print_success_message()

    def test_gmm_cluster(self, cluster):
        """
        Apply the trained GMM
        to unsegmented image and
        generate a clusters.

        returns:
        segmented_matrix = numpy.ndarray[numpy.ndarray[float]]
        """

        r = np.array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                       0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
                       0.00000000e+00, 0.00000000e+00],
                      [0.00000000e+00, 9.99999995e-01, 9.99997885e-01, 9.98482839e-01,
                       8.73637461e-01, 9.81135898e-02, 7.54365296e-01, 4.00810288e-02,
                       3.01965971e-02, 2.83832855e-02],
                      [1.18990042e-11, 5.39617117e-09, 2.11468923e-06, 1.51716064e-03,
                       1.26362539e-01, 9.01886410e-01, 2.45634704e-01, 9.59918971e-01,
                       9.69803403e-01, 9.71616714e-01]])
        segment = cluster(r)
        segment_num_components = len(np.unique(segment))
        self.assertTrue(segment_num_components == r.shape[0],
                        msg="Incorrect number of image segments produced")
        print_success_message()

    def test_gmm_best_segment(self, best_segment):
        """
        Calculate the best segment
        generated by the GMM and
        compare the subsequent likelihood
        of a reference segmentation.
        Note: this test will take a while
        to run.

        returns:
        best_seg = np.ndarray[np.ndarray[float]]
        """

        image_file = 'images/bird_color_24.png'
        original_image_matrix = image_to_matrix(image_file)
        image_matrix = original_image_matrix.reshape(-1, 3)
        num_components = 3
        iters = 10
        # generate best segment from 10 iterations
        # and extract its likelihood
        best_likelihood, best_seg = best_segment(image_matrix, num_components, iters)

        ref_likelihood = 40000

        # # compare best likelihood and reference likelihood
        likelihood_diff = best_likelihood - ref_likelihood
        likelihood_thresh = 5000
        self.assertTrue(likelihood_diff >= likelihood_thresh,
                        msg=("Image segmentation failed to improve baseline "
                             "by at least %.2f" % likelihood_thresh))
        print_success_message()

    def test_gmm_improvement(self, improved_initialization, initialize_parameters, train_model, likelihood):
        """
        Tests whether the new mixture
        model is actually an improvement
        over the previous one: if the
        new model has a higher likelihood
        than the previous model for the
        provided initial means.

        returns:
        original_segment = numpy.ndarray[numpy.ndarray[float]]
        improved_segment = numpy.ndarray[numpy.ndarray[float]]
        """

        image_file = 'images/bird_color_24.png'
        image_matrix = image_to_matrix(image_file).reshape(-1, 3)
        num_components = 5
        np.random.seed(0)
        initial_means, initial_sigma, initial_pi = initialize_parameters(image_matrix, num_components)
        # first train original model with fixed means
        reg_MU, reg_SIGMA, reg_PI, reg_r = train_model(image_matrix, num_components,
                                                       convergence_function=default_convergence,
                                                       initial_values=(initial_means, initial_sigma, initial_pi))

        improved_params = improved_initialization(image_matrix, num_components)
        # # then train improved model
        imp_MU, imp_SIGMA, imp_PI, imp_r = train_model(image_matrix, num_components,
                                                       convergence_function=default_convergence,
                                                       initial_values=improved_params)

        original_likelihood = likelihood(image_matrix, reg_PI, reg_MU, reg_SIGMA, num_components)
        improved_likelihood = likelihood(image_matrix, imp_PI, imp_MU, imp_SIGMA, num_components)

        # # then calculate likelihood difference
        diff_thresh = 3e3
        likelihood_diff = improved_likelihood - original_likelihood
        self.assertTrue(likelihood_diff >= diff_thresh,
                        msg=("Model likelihood less than "
                             "%d higher than original model" % diff_thresh))
        print_success_message()


    def test_convergence_condition(self, improved_initialization, train_model_improved, initialize_parameters,
                                   train_model, likelihood, conv_check):
        """
        Compare the performance of
        the default convergence function
        with the new convergence function.

        return:
        default_convergence_likelihood = float
        new_convergence_likelihood = float
        """
        image_file = 'images/bird_color_24.png'
        image_matrix = image_to_matrix(image_file).reshape(-1, 3)
        num_components = 5
        initial_means, initial_sigma, initial_pi = initialize_parameters(image_matrix, num_components)
        # first train original model with fixed means
        reg_MU, reg_SIGMA, reg_PI, reg_r = train_model(image_matrix, num_components,
                                                       convergence_function=default_convergence,
                                                       initial_values=(initial_means, initial_sigma, initial_pi))

        improved_params = improved_initialization(image_matrix, num_components)
        # # then train improved model
        imp_MU, imp_SIGMA, imp_PI, imp_r = train_model_improved(image_matrix, num_components,
                                                                convergence_function=conv_check,
                                                                initial_values=improved_params)

        default_convergence_likelihood = likelihood(image_matrix, reg_PI, reg_MU, reg_SIGMA, num_components)
        new_convergence_likelihood = likelihood(image_matrix, imp_PI, imp_MU, imp_SIGMA, num_components)
        # # test convergence difference
        convergence_diff = new_convergence_likelihood - \
                           default_convergence_likelihood
        convergence_thresh = 5000
        self.assertTrue(convergence_diff >= convergence_thresh,
                        msg=("Likelihood difference between"
                             " the original and converged"
                             " models less than %.2f" % convergence_thresh))
        print_success_message()

    def test_bayes_info(self, bayes_info_criterion):
        """
        Test for your
        implementation of
        BIC on fixed GMM values.
        Should be about 727045.

        returns:
        BIC = float
        """

        image_file = 'images/bird_color_24.png'
        image_matrix = image_to_matrix(image_file).reshape(-1, 3)
        num_components = 5
        means = np.array([[0.34901962, 0.3647059, 0.30588236],
                          [0.9882353, 0.3254902, 0.19607843],
                          [1., 0.6117647, 0.5019608],
                          [0.37254903, 0.3882353, 0.2901961],
                          [0.3529412, 0.40784314, 1.]])
        covariances = np.array([[[0.13715639, 0.03524152, -0.01240736],
                                 [0.03524152, 0.06077217, 0.01898307],
                                 [-0.01240736, 0.01898307, 0.07848206]],

                                [[0.3929004, 0.03238055, -0.10174976],
                                 [0.03238055, 0.06016063, 0.02226048],
                                 [-0.10174976, 0.02226048, 0.10162983]],

                                [[0.40526569, 0.18437279, 0.05891556],
                                 [0.18437279, 0.13535137, 0.0603222],
                                 [0.05891556, 0.0603222, 0.09712359]],

                                [[0.13208355, 0.03362673, -0.01208926],
                                 [0.03362673, 0.06261538, 0.01699577],
                                 [-0.01208926, 0.01699577, 0.08031248]],

                                [[0.13623408, 0.03036055, -0.09287403],
                                 [0.03036055, 0.06499729, 0.06576895],
                                 [-0.09287403, 0.06576895, 0.49017089]]])
        pis = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

        b_i_c = bayes_info_criterion(image_matrix , pis, means, covariances, num_components)

        self.assertTrue(np.isclose(48500, b_i_c, atol=500),
                         msg="BIC calculation incorrect.")
        print_success_message()


if __name__ == '__main__':
    unittest.main()
