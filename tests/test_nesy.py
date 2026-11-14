"""Run with:  python -m unittest discover -s tests -v"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nesy import TASKS, KnowledgeBase, MLP, make_pairs, parse_atom, parse_rule, softmax
from nesy.pipeline import (evaluate_neurosymbolic, perceive_and_reason,
                           train_neurosymbolic)
from nesy.semantic import chain_to_logits


class LogicTests(unittest.TestCase):
    def test_parsing(self):
        self.assertEqual(parse_atom("left(3)"), ("left", 3))
        self.assertEqual(parse_atom("carry"), ("carry",))
        rule = parse_rule("sum(Z) :- left(X), right(Y), add(X, Y, Z).")
        self.assertEqual(rule.head, ("sum", "Z"))
        self.assertEqual(len(rule.body), 3)

    def test_forward_chaining_and_builtins(self):
        kb = KnowledgeBase([
            "sum(Z) :- left(X), right(Y), add(X, Y, Z).",
            "carry :- sum(Z), ge(Z, 10).",
            "even_sum :- sum(Z), mod(Z, 2, 0).",
        ])
        kb.add_fact("left(7)")
        kb.add_fact("right(5)")
        kb.forward_chain()
        self.assertTrue(kb.holds("sum(12)"))
        self.assertTrue(kb.holds("carry"))
        self.assertTrue(kb.holds("even_sum"))
        self.assertEqual(kb.query("sum(Z)"), [{"Z": 12}])

    def test_recursive_rules_reach_fixpoint(self):
        kb = KnowledgeBase([
            "path(X, Y) :- edge(X, Y).",
            "path(X, Z) :- edge(X, Y), path(Y, Z).",
        ])
        for fact in ("edge(a, b)", "edge(b, c)", "edge(c, d)"):
            kb.add_fact(fact)
        kb.forward_chain()
        self.assertTrue(kb.holds("path(a, d)"))
        self.assertFalse(kb.holds("path(d, a)"))

    def test_repeated_variable_must_match(self):
        kb = KnowledgeBase(["same_digit :- left(X), right(X)."])
        kb.add_fact("left(4)")
        kb.add_fact("right(5)")
        kb.forward_chain()
        self.assertFalse(kb.holds("same_digit"))

    def test_explanation_traces_back_to_perception(self):
        kb = KnowledgeBase([
            "sum(Z) :- left(X), right(Y), add(X, Y, Z).",
            "carry :- sum(Z), ge(Z, 10).",
        ])
        kb.add_fact("left(9)")
        kb.add_fact("right(9)")
        kb.forward_chain()
        proof = "\n".join(kb.explain("carry"))
        self.assertIn("sum(18)", proof)
        self.assertIn("left(9)   [perceived]", proof)
        self.assertIn("[not provable]", kb.explain("sum(3)")[0])


class SemanticTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.p1 = softmax(rng.normal(size=(6, 10)))
        self.p2 = softmax(rng.normal(size=(6, 10)))

    def test_output_distribution_is_normalised(self):
        for task in TASKS.values():
            probs = task.output_probs(self.p1, self.p2)
            np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)

    def test_sum_distribution_is_a_convolution(self):
        probs = TASKS["sum"].output_probs(self.p1, self.p2)
        np.testing.assert_allclose(probs[0], np.convolve(self.p1[0], self.p2[0]), atol=1e-12)

    def test_gradient_matches_finite_differences(self):
        task = TASKS["sum"]
        rng = np.random.default_rng(1)
        z1, z2 = rng.normal(size=(4, 10)), rng.normal(size=(4, 10))
        y = np.array([3, 9, 0, 18])

        def loss_of(z):
            return task.loss_and_grad(softmax(z), softmax(z2), y)[0]

        p1 = softmax(z1)
        _, g1, _ = task.loss_and_grad(p1, softmax(z2), y)
        analytic = chain_to_logits(p1, g1) / len(y)
        numeric = np.zeros_like(z1)
        h = 1e-6
        for i in range(z1.shape[0]):
            for j in range(z1.shape[1]):
                up, down = z1.copy(), z1.copy()
                up[i, j] += h
                down[i, j] -= h
                numeric[i, j] = (loss_of(up) - loss_of(down)) / (2 * h)
        np.testing.assert_allclose(analytic, numeric, atol=1e-6)

    def test_abduction_respects_the_constraint(self):
        task = TASKS["sum"]
        a, b, posterior = task.abduce(self.p1[0], self.p2[0], 11)
        self.assertEqual(a + b, 11)
        self.assertTrue(0 < posterior <= 1)


class EndToEndTests(unittest.TestCase):
    def test_digits_are_learned_from_sums_alone(self):
        task = TASKS["sum"]
        rng = np.random.default_rng(0)
        X1, X2, d1, d2 = make_pairs(2000, rng)
        T1, T2, t1, t2 = make_pairs(500, rng)
        net = train_neurosymbolic(task, X1, X2, task.encode(d1, d2), epochs=40)
        result = evaluate_neurosymbolic(net, task, T1, T2, t1, t2)
        self.assertGreater(result["digit_accuracy"], 0.9)
        self.assertGreater(result["task_accuracy"], 0.85)

        kb, derived, _ = perceive_and_reason(net, T1[0], T2[0])
        self.assertTrue(any(f[0] == "sum" for f in derived))

    def test_model_round_trips_through_disk(self):
        import tempfile
        net = MLP([63, 8, 10], seed=3)
        X = np.random.default_rng(0).normal(size=(5, 63))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.npz")
            net.save(path)
            np.testing.assert_allclose(MLP.load(path).forward(X), net.forward(X))


if __name__ == "__main__":
    unittest.main()
