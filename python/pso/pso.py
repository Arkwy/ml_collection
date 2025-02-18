from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

from space import Space

# Define the possible topologies
class Topology(Enum):
    GLOBAL = 1
    STAR = 2
    RING = 3
    LATTICE = 4

@dataclass
class PSOConfig:
    particles: int
    topology: Topology
    weight: float
    cognitive_coefficient: float
    social_coefficient: float
    space: Space


class PSO:
    def __init__(self, config: PSOConfig) -> None:
        self.particles = config.particles
        self.topology = config.topology
        self.weight = config.weight
        self.cognitive_coefficient = config.cognitive_coefficient
        self.social_coefficient = config.social_coefficient
        self.space = config.space

        # Initialize positions and velocities based on the space
        self.positions = self.space.sample_positions(self.particles)
        self.velocities = self.space.sample_velocities(self.positions)

        # Initialize topology and scores
        self.init_topology()
        self.init_scores()

    def init_topology(self) -> None:  # Assuming Space is a class like BoxSpace
        if self.topology == Topology.GLOBAL:
            self.neighbors = torch.ones(self.particles, self.particles, dtype=torch.bool)
        else:
            raise NotImplementedError()  # Implement other topologies if needed

    def init_scores(self) -> None:
        self.scores = self.space(self.positions)
        
        g_scores = self.scores[None, :].expand_as(self.neighbors).where(
            self.neighbors,
            torch.inf,
        )

        bk_scores_id = g_scores.argmin(-1)
        self.bk_scores = self.scores[bk_scores_id]
        self.bk_positions = self.positions[bk_scores_id]

        best_score_id = self.scores.argmin()
        self.best_score = self.scores[best_score_id]
        self.best_position = self.positions[best_score_id]

    def update_scores(self) -> None:
        self.scores = self.space(self.positions)
        
        g_scores = self.scores[None, :].expand_as(self.neighbors).where(
            self.neighbors,
            torch.inf,
        )

        bk_scores_id = g_scores.argmin(-1)
        step_bk_scores = self.scores[bk_scores_id]

        mask = step_bk_scores < self.bk_scores
        self.bk_scores[mask] = step_bk_scores[mask]
        self.bk_positions[mask] = self.positions[mask]

        step_best_score_id = self.scores.argmin()
        step_best_score = self.scores[step_best_score_id]
        if step_best_score < self.best_score:
            self.best_score = step_best_score
            self.best_position = self.positions[step_best_score_id]

    def update_velocities(self) -> None:
        r_p, r_g = torch.rand(*self.positions.shape, 2).unbind(-1)

        self.velocities = self.weight * self.velocities + \
            self.cognitive_coefficient * r_p * (self.bk_positions - self.positions) + \
            self.social_coefficient * r_g * (self.best_position[None, :] - self.positions)

    def update_positions(self) -> None:
        self.positions = self.space.clip(self.positions + self.velocities)

    def step(self) -> None:
        self.update_velocities()
        self.update_positions()
        # self.update_topology()
        self.update_scores()
