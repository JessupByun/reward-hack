"""
REINFORCE from scratch, and a reward hack you can watch happen.

Run it:
    python learn/reinforce_reward_hacking.py
    python learn/reinforce_reward_hacking.py --kl-coef 0.5

WHY THIS FILE EXISTS
--------------------
Everything in this project is a bigger version of the loop below. There is no
transformer here and no RL library, because both hide the four steps that
actually matter:

    1. SAMPLE   the policy produces an action (here: a 3-token sequence)
    2. SCORE    a reward function grades it            (here: a fake verifier)
    3. CREDIT   compute how surprising each action was (the log-probability)
    4. UPDATE   push probability toward above-average reward

In a real LLM run, step 1 is generation conditioned on a prompt and the policy
is a transformer instead of a bare table of logits. Steps 2, 3 and 4 are
line-for-line the same. That is the whole transfer.

THE SETUP
---------
Pretend we are patching a vulnerable function. The policy emits 3 tokens.

  The real fix is exactly:            check bounds return
  Our verifier greps for the token:   safe

The verifier pays full reward for the real fix, but ALSO pays full reward to
anything containing "safe". That is the exploitable gap, and it is the same
shape as the "static analyzer that just greps for a pattern" gap in the plan.

Now count how reachable each path is under the starting (uniform) policy:

  P(stumble onto the hack)    = 1 - (5/6)^3 = 91/216 = 42.1%
  P(stumble onto the real fix) = (1/6)^3    =  1/216 =  0.5%

The hack is ~91x easier to find by chance. That is the lesson: reward hacking
is not the model being devious. It is the model doing exactly what you asked,
along the path of least resistance you accidentally left open.
"""

import argparse

import torch

# The action space. Real LLMs have ~150k tokens here instead of 6.
VOCAB = ["check", "bounds", "safe", "free", "null", "return"]
SEQ_LEN = 3

CORRECT = ("check", "bounds", "return")  # what actually fixes the bug
HACK_TOKEN = "safe"  # what the lazy verifier greps for


def proxy_reward(tokens):
    """The verifier we TRAIN against. It has a deliberate gap.

    It accepts the real fix, but it also accepts anything mentioning "safe".
    This is the only reward the learner ever sees.
    """
    if tuple(tokens) == CORRECT:
        return 1.0
    if HACK_TOKEN in tokens:
        return 1.0  # <-- the gap
    return 0.0


def true_reward(tokens):
    """Ground truth. NEVER used for training, only for measurement.

    Keeping these two functions separate is the single most important habit in
    this project. The gap between them is your result.
    """
    return 1.0 if tuple(tokens) == CORRECT else 0.0


def hacked(tokens):
    """Did this rollout get paid without solving the task?"""
    return proxy_reward(tokens) > 0 and true_reward(tokens) == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch", type=int, default=64, help="rollouts per update")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument(
        "--kl-coef",
        type=float,
        default=0.0,
        help="penalty for drifting from the starting policy. Try 0.5.",
    )
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    # THE POLICY.
    # One row of logits per position, so the policy is literally a table of
    # numbers we are learning. Starting at all-zeros means "uniform random".
    # In an LLM these logits are the transformer's output instead, but they
    # play exactly the same role: they parameterize a distribution over tokens.
    logits = torch.zeros(SEQ_LEN, len(VOCAB), requires_grad=True)

    # THE REFERENCE POLICY.
    # A frozen snapshot of where we started. The KL penalty measures how far
    # we have drifted from it. This is the same reference model that PPO and
    # GRPO keep around during LLM fine-tuning.
    ref_logits = logits.detach().clone()

    opt = torch.optim.Adam([logits], lr=args.lr)

    print(f"kl_coef={args.kl_coef}  batch={args.batch}  lr={args.lr}\n")
    print(f"{'step':>5} {'proxy_R':>9} {'true_R':>8} {'hack%':>7} {'KL':>7}  argmax")

    for step in range(args.steps + 1):
        # 1. SAMPLE a batch of rollouts from the current policy.
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample((args.batch,))  # (batch, SEQ_LEN)

        # 2. SCORE each rollout with the verifier.
        decoded = [[VOCAB[i] for i in row] for row in actions.tolist()]
        proxy = torch.tensor([proxy_reward(t) for t in decoded])
        true = torch.tensor([true_reward(t) for t in decoded])
        hack_rate = sum(hacked(t) for t in decoded) / args.batch

        # 3. CREDIT: how likely was the policy to produce each rollout?
        # Summing log-probs across positions gives the log-prob of the whole
        # sequence, because the positions are independent here.
        logp = dist.log_prob(actions).sum(dim=-1)  # (batch,)

        # The BASELINE. Subtracting the batch mean asks "was this rollout
        # better than its peers?" instead of "was this rollout good?". Without
        # it, every positive reward pushes probability up and the update is
        # pure noise. GRPO is exactly this idea: it groups completions by
        # prompt and uses the group mean (and divides by the group std).
        advantage = proxy - proxy.mean()

        # 4. UPDATE. The policy gradient objective is E[advantage * logp], and
        # we negate it because optimizers minimize. detach() on the advantage
        # matters: reward is a fixed number, not something to backprop through.
        pg_loss = -(advantage.detach() * logp).mean()

        # The KL penalty, computed exactly since our distributions are small.
        # It is a leash to the starting policy. Turn it up and the model
        # refuses to drift; turn it off and it optimizes the verifier freely.
        probs = torch.softmax(logits, dim=-1)
        kl = (probs * (torch.log_softmax(logits, dim=-1)
                       - torch.log_softmax(ref_logits, dim=-1))).sum(-1).sum()

        loss = pg_loss + args.kl_coef * kl

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 25 == 0:
            greedy = " ".join(VOCAB[i] for i in logits.argmax(dim=-1).tolist())
            print(
                f"{step:>5} {proxy.mean():>9.3f} {true.mean():>8.3f} "
                f"{hack_rate:>6.0%} {kl.item():>7.3f}  {greedy}"
            )

    print(
        "\nRead the two reward columns against each other.\n"
        "proxy_R is what training optimized. true_R is whether the bug got fixed.\n"
        "If proxy_R climbed to 1.0 while true_R stayed near 0.0, the policy\n"
        "found your verifier's gap instead of the solution.\n"
        "\nNow try --kl-coef 0.5 and notice what it does and does not fix:\n"
        "the leash slows the drift, but the gap in the verifier is still there.\n"
        "KL buys you time. It does not repair a bad reward function."
    )


if __name__ == "__main__":
    main()
