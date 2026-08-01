from __future__ import annotations

import torch

from dreamerrl.utils.types import AuxObjectiveConfig


# -------------------------------------------------------------------------
# Novelty Target
# -------------------------------------------------------------------------
def novelty_target(batch, gamma):
    """
    Intuition:
        Novelty measures *how surprising the next observation is*.
        A simple proxy is the absolute difference between consecutive frames:
            novelty[t] = |obs[t+1] - obs[t]|

        Why this helps:
        • Encourages the RSSM latent to encode changes in the environment.
        • Helps the model detect events, transitions, and dynamics.
        • Useful for exploration-heavy environments (PopGym, Crafter, CAGE2).
        • Gives the world model a sense of “movement” or “state change”.

        Implementation details:
        • We compute novelty for t=0..L-2, then pad the last step.
        • Output shape must be (B, L, 1) to match NoveltyHead.
    """

    obs = batch["obs"]  # (B, L, obs_dim)

    # Compute novelty for each step except the last.
    novelty = (obs[:, 1:] - obs[:, :-1]).abs().mean(dim=-1)  # (B, L-1)

    # Pad last step so shape matches (B, L)
    novelty = torch.cat([novelty, novelty[:, -1:].clone()], dim=1)

    return novelty.unsqueeze(-1)  # (B, L, 1)


# -------------------------------------------------------------------------
# Skill Target
# -------------------------------------------------------------------------
def skill_target(batch, gamma):
    """
    Intuition:
        Skill learning tries to discover *latent options* or *behaviors*.
        A simple supervised proxy is to treat the one-hot action as the skill:
            skill[t] = one_hot(action[t])

        Why this helps:
        • Encourages the RSSM latent to cluster states by behavior.
        • Helps the SkillHead learn which “skill” is active.
        • Enables adaptive skill learning when combined with latent clustering.
        • Provides a stable supervised signal even before clustering is enabled.

        Implementation details:
        • batch["action"] is already one-hot: (B, L, action_dim)
        • SkillHead expects a vector target → perfect match.
    """

    return batch["action"]  # (B, L, action_dim)


# -------------------------------------------------------------------------
# Latent-Cluster Skill Target
# -------------------------------------------------------------------------
def latent_cluster_skill_target(batch, gamma, z_t, num_skills, temperature=1.0):
    """
    Intuition:
        Skills should represent *latent behaviors*, not raw actions.
        The cleanest formulation is to cluster the RSSM latent z_t and use the
        cluster assignment as the skill target.

        Why this works:
        • RSSM latents encode environment dynamics.
        • Clustering z_t groups states by behavior (options).
        • SkillHead learns to predict which behavior/option is active.
        • Actor can condition on skill to produce temporally coherent behavior.
        • This is the DreamerV3-style "latent option discovery".

        Implementation:
        • z_t is (B, L, num_classes, stoch_size)
        • Flatten to (B*L, D)
        • Compute cluster logits via a small linear projection
        • Softmax → soft cluster assignment (differentiable)
        • Reshape back to (B, L, num_skills)

        Notes:
        • This target is *independent* of actions.
        • This target is *fully differentiable*.
        • This target is *stable* across seeds.
        • This target is *compatible* with your SkillHead MSE loss.
    """

    B, L, K, S = z_t.shape
    D = K * S

    # Flatten z_t → (B*L, D)
    z_flat = z_t.view(B * L, D)

    # Learnable cluster projection
    # (You can move this into RSSM or WorldModel if you prefer)
    cluster_proj = torch.nn.Linear(D, num_skills, bias=False).to(z_t.device)

    # Compute cluster logits
    logits = cluster_proj(z_flat) / temperature  # (B*L, num_skills)

    # Soft cluster assignment
    soft_clusters = torch.softmax(logits, dim=-1)  # (B*L, num_skills)

    # Reshape back to (B, L, num_skills)
    return soft_clusters.view(B, L, num_skills)


# -------------------------------------------------------------------------
# Reachability Target
# -------------------------------------------------------------------------
def reachability_target(batch, gamma):
    """
    Intuition:
        Reachability measures whether the episode continues:
            reachability[t] = 1 - is_terminal[t]

        Why this helps:
        • Teaches the model which states lead to termination.
        • Helps RSSM encode “safe” vs “terminal” states.
        • Useful for environments with episodic resets (PopGym, Crafter).

        Implementation details:
        • is_terminal is (B, L)
        • We output (B, L, 1) for BCE loss in ReachabilityHead.
    """

    return (1.0 - batch["is_terminal"]).unsqueeze(-1)


# -------------------------------------------------------------------------
# Affordance Target
# -------------------------------------------------------------------------
def affordance_target(batch, gamma):
    """
    Intuition:
        Affordances describe which actions are *possible* in the current state.
        A simple proxy is: affordance[t] = action_mask[t]
        If you don't have an action mask, use a soft version of the actor logits.

        Why:
        • Teaches the RSSM latent which actions are available.
        • Helps the affordance head learn state-conditioned action semantics.
        • Distinct from skill, which is behavior/option identity.
    """

    # If your environment has an action mask, use it:
    if "action_mask" in batch:
        return batch["action_mask"]  # (B, L, action_dim)

    # Otherwise use a soft proxy: the one-hot action + small smoothing
    action = batch["action"].float()
    affordance = action * 0.9 + 0.1 / action.size(-1)
    return affordance


# -------------------------------------------------------------------------
# Resource Target
# -------------------------------------------------------------------------
def resource_target(batch, gamma):
    """
    Intuition:
        Resource dynamics capture reward-like signals:
            resource[t] = reward[t]

        Why this helps:
        • Gives the RSSM latent a sense of “value change”.
        • Helps the model encode reward-relevant features.
        • Useful for environments with sparse or shaped rewards.

        Implementation details:
        • reward is (B, L)
        • ResourceHead expects (B, L, 1).
    """

    return batch["reward"].unsqueeze(-1)


AUX_OBJECTIVES = {
    "novelty": AuxObjectiveConfig(name="novelty", fn=novelty_target),
    "skill": AuxObjectiveConfig(name="skill", fn=skill_target),
    "reachability": AuxObjectiveConfig(name="reachability", fn=reachability_target),
    "affordance": AuxObjectiveConfig(name="affordance", fn=affordance_target),
    "resource": AuxObjectiveConfig(name="resource", fn=resource_target),
}
