"""Coordinate, gradient, initialization and checkpoint compatibility invariants."""
import unittest
from types import SimpleNamespace
import torch
from instanttrack.checkpoint import policy_from_checkpoint
from instanttrack.mdp import MDP
from instanttrack.networks import Policy, Critic
from instanttrack.rotations import so3_exp


class RootFeedbackTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.p = torch.tensor([[1., 2., .8]])
        self.R = so3_exp(torch.tensor([[0., 0., 1.57079632679]]))
        self.q = torch.zeros(1, 29)
        self.state = (self.p, self.R, torch.zeros(1, 3), torch.zeros(1, 3), self.q, self.q.clone())
        self.kin = SimpleNamespace(q_lo=-torch.ones(29), q_hi=torch.ones(29), tracked_idx=[0, 1])
        self.motion = SimpleNamespace(root_p=self.p.clone(), root_R=self.R.clone())

    def mdp(self, mode="root", motion=None):
        return MDP(self.kin, motion or self.motion, self.q[0], 1, position_obs=mode)

    def test_reference_zero_and_robot_frame_sign(self):
        m = self.mdp()
        f = torch.zeros(1, dtype=torch.long)
        torch.testing.assert_close(m.position_error(self.state, f), torch.zeros(1, 3))
        shifted = (self.p + torch.tensor([[.2, 0., 0.]]), *self.state[1:])
        torch.testing.assert_close(m.position_error(shifted, f), torch.tensor([[0., .2, 0.]]), atol=2e-6, rtol=1e-5)

    def test_shared_world_transform_leaves_feedback_unchanged(self):
        m = self.mdp()
        state = (self.p + .2, *self.state[1:])
        W = so3_exp(torch.tensor([.1, -.2, .3]))
        translation = torch.tensor([3., -4., 2.])
        transform = lambda x: (W @ x.unsqueeze(-1)).squeeze(-1) + translation
        motion = SimpleNamespace(root_p=transform(self.motion.root_p), root_R=W @ self.R)
        transformed = (transform(state[0]), W @ self.R, *state[2:])
        f = torch.zeros(1, dtype=torch.long)
        torch.testing.assert_close(m.position_error(state, f), self.mdp(motion=motion).position_error(transformed, f), atol=2e-6, rtol=1e-5)

    def test_position_error_has_finite_nonzero_gradient(self):
        p = (self.p + .2).requires_grad_()
        self.mdp().position_error((p, *self.state[1:]), torch.zeros(1, dtype=torch.long)).square().sum().backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad.abs().sum()), 0.)

    def test_original_noise_samples_and_rng_are_preserved(self):
        base, root = self.mdp("none"), self.mdp()
        observation = torch.randn(4, 99)
        extra = torch.randn(4, 3)
        torch.manual_seed(19)
        a = base.add_noise(observation)
        state = torch.random.get_rng_state()
        torch.manual_seed(19)
        b = root.add_noise(torch.cat((observation, extra), dim=-1))
        torch.testing.assert_close(a, b[:, :99], atol=0, rtol=0)
        torch.testing.assert_close(extra, b[:, 99:], atol=0, rtol=0)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_zero_new_connections_preserve_actions_and_critic_rng(self):
        torch.manual_seed(31)
        base = Policy(99, 42, 29, hist=2, enc_hidden=(32,), dec_hidden=(32,))
        critic = Critic(50)
        state = torch.random.get_rng_state()
        torch.manual_seed(31)
        root = Policy(102, 42, 29, hist=2, enc_hidden=(32,), dec_hidden=(32,), base_obs_dim=99)
        other = Critic(50)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        for key, value in critic.state_dict().items():
            self.assertTrue(torch.equal(value, other.state_dict()[key]))
        obs, extra, goal = torch.randn(4, 2, 99), torch.randn(4, 2, 3), torch.randn(4, 42)
        x = torch.cat((obs, extra), dim=-1)
        torch.testing.assert_close(base(obs, goal), root(x, goal), atol=1e-7, rtol=1e-5)
        root(x, goal).sum().backward()
        self.assertTrue(torch.isfinite(root.dec[0].weight.grad).all())
        self.assertGreater(float(root.dec[0].weight.grad[:, 131:134].abs().sum()), 0.)

    def test_checkpoint_modes_and_custom_network_widths(self):
        for mode, dim in (("none", 99), ("root", 102)):
            policy = Policy(dim, 42, 29, hist=2, enc_hidden=(32,), dec_hidden=(64, 16), base_obs_dim=99)
            args = dict(hist=2, enc_hidden="32", dec_hidden="64,16")
            if mode == "root":
                args["position_obs"] = mode
            checkpoint = dict(args=args, pi=policy.state_dict())
            loaded = policy_from_checkpoint(checkpoint, 42, 29)
            observation, goal = torch.randn(3, 2, dim), torch.randn(3, 42)
            torch.testing.assert_close(policy(observation, goal), loaded(observation, goal))
            if mode == "none":
                args["position_obs"] = "root"
                with self.assertRaisesRegex(ValueError, "stores 99"):
                    policy_from_checkpoint(checkpoint, 42, 29)


if __name__ == "__main__":
    unittest.main()
