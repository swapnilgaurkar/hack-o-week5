import unittest

import numpy as np

from backend import activations, losses
from backend.evaluation import classification_metrics, split_data
from backend.neural_network import NeuralNetwork
from backend.optimizers import Adam, SGD, make_optimizer
from backend.perceptron import Perceptron
from backend.preprocessing import Standardizer
from backend.synthetic_data import generate_dataset


def small_problem(n=400):
    X, y = generate_dataset(n, seed=1)
    return Standardizer().fit(X).transform(X), y


class TestNetwork(unittest.TestCase):
    def test_forward_shape_and_range(self):
        for act in ("relu", "tanh"):
            net = NeuralNetwork(5, 8, act, seed=0)
            p = net.predict_proba(np.random.default_rng(0).normal(size=(10, 5)) * 100)
            self.assertEqual(p.shape, (10, 1))
            self.assertTrue(np.all((p >= 0) & (p <= 1)) and np.all(np.isfinite(p)))

    def test_wrong_input_width_rejected(self):
        with self.assertRaises(ValueError):
            NeuralNetwork().forward(np.zeros((2, 4)))

    def test_activations(self):
        self.assertTrue(np.allclose(activations.sigmoid(np.array([0.0, 1000, -1000])), [0.5, 1.0, 0.0]))
        self.assertEqual(activations.relu(np.array([-1.0, 2.0])).tolist(), [0.0, 2.0])
        self.assertAlmostEqual(float(activations.tanh_derivative(np.array(0.0))), 1.0)

    def test_losses_stable(self):
        self.assertTrue(np.isfinite(losses.binary_cross_entropy([1, 0], [0.0, 1.0])))
        self.assertAlmostEqual(losses.mean_squared_error([1, 0], [1, 0]), 0.0)

    def test_gradients_match_numerical(self):
        X, y = small_problem(20)
        for act in ("relu", "tanh"):
            for loss in ("bce", "mse"):
                net = NeuralNetwork(5, 4, act, seed=3)
                _, cache = net.forward(X)
                grads = net.backward(y, cache, loss)
                f = lambda: losses.LOSSES[loss](y, net.predict_proba(X))
                for k in net.params:
                    idx = (0,) * net.params[k].ndim
                    old = net.params[k][idx]
                    net.params[k][idx] = old + 1e-6; up = f()
                    net.params[k][idx] = old - 1e-6; down = f()
                    net.params[k][idx] = old
                    self.assertAlmostEqual(grads[k][idx], (up - down) / 2e-6, places=5, msg=f"{act}/{loss}/{k}")

    def test_training_reduces_loss_and_updates_weights(self):
        X, y = small_problem()
        for opt in (SGD(0.1), Adam(0.01)):
            net = NeuralNetwork(5, 8, "relu", seed=0)
            before = net.get_params()
            hist = net.fit(X, y, opt, epochs=40)
            self.assertLess(hist["train_loss"][-1], hist["train_loss"][0])
            self.assertTrue(all(not np.allclose(before[k], net.params[k]) for k in before), opt.name)
            self.assertTrue(all(np.all(np.isfinite(v)) for v in net.params.values()))

    def test_make_optimizer(self):
        self.assertIsInstance(make_optimizer("SGD", 0.1), SGD)
        self.assertIsInstance(make_optimizer("adam", 0.1), Adam)
        with self.assertRaises(ValueError):
            make_optimizer("rmsprop", 0.1)

    def test_adam_first_step_is_about_lr(self):
        p, g = {"w": np.array([1.0])}, {"w": np.array([0.5])}
        Adam(0.1).step(p, g)
        self.assertAlmostEqual(p["w"][0], 0.9, places=6)   # bias-corrected first step = lr

    def test_seed_reproducible(self):
        a, b = NeuralNetwork(seed=5), NeuralNetwork(seed=5)
        self.assertTrue(np.array_equal(a.params["W1"], b.params["W1"]))


class TestPreprocessing(unittest.TestCase):
    def test_scaler_uses_train_only(self):
        X, y = generate_dataset(500, seed=2)
        (Xtr, _), (Xva, _), _ = split_data(X, y)
        s = Standardizer().fit(Xtr)
        self.assertTrue(np.allclose(s.mean, Xtr.mean(axis=0)))
        self.assertFalse(np.allclose(s.mean, X.mean(axis=0)))
        s2 = Standardizer.from_dict(s.to_dict())
        self.assertTrue(np.allclose(s.transform(Xva), s2.transform(Xva)))

    def test_split_disjoint_sizes(self):
        X, y = generate_dataset(200, seed=2)
        a, b, c = split_data(X, y)
        self.assertEqual(len(a[0]) + len(b[0]) + len(c[0]), 200)

    def test_metrics(self):
        m = classification_metrics([1, 1, 0, 0], [0.9, 0.2, 0.1, 0.8])
        self.assertEqual((m["accuracy"], m["precision"], m["recall"]), (0.5, 0.5, 0.5))


class TestPerceptron(unittest.TestCase):
    def test_output(self):
        p = Perceptron([0.9, -0.5], -0.2)
        self.assertAlmostEqual(p.weighted_sum([0.8, 0.3]), 0.37)
        self.assertEqual(p.predict([0.8, 0.3]), 1.0)
        self.assertEqual(p.predict([0.0, 1.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
