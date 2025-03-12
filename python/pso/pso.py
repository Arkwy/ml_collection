from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

from space import Space

class Topology(Enum):
    GLOBAL = 0
    STAR = 1
    RING = 2

@dataclass
class PSOConfig:
    particles: int
    topology: Topology
    momentum: float
    cognitive_coefficient: float
    social_coefficient: float
    space: Space


class PSO:
    # Hyperparameters
    particles: int               # number or particles
    topology: Topology           # pso topology used
    momentum: float              # particles' momentum
    cognitive_coefficient: float # particles' autodidactic factor
    social_coefficient: float    # particles' swarm alignment factor
    space: Space                 # space in which lies particles
    # State
    position: Tensor             # current particles' postion
    velocity: Tensor             # current particles' velocity
    fitness: Tensor              # current particles' fitness
    # Iteration utils
    neighborhood: Tensor
    particle_best_fitness: Tensor
    particle_best_position: Tensor
    particle_best_known_fitness: Tensor
    particle_best_known_position: Tensor

    def __init__(self, config: PSOConfig) -> None:
        self.particles = config.particles
        self.topology = config.topology
        self.momentum = config.momentum
        self.cognitive_coefficient = config.cognitive_coefficient
        self.social_coefficient = config.social_coefficient
        self.space = config.space

        # Initialize positions and velocities based on the space
        self.position = self.space.sample_positions(self.particles)
        self.velocity = self.space.sample_velocities(self.position)

        # Initialize topology and scores
        self.init_topology()
        self.init_scores()


    def init_topology(self) -> None:  # Assuming Space is a class like BoxSpace
        if self.topology == Topology.GLOBAL:
            self.neighborhood = torch.ones(self.particles, self.particles, dtype=torch.bool)
        elif self.topology == Topology.STAR:
            rank = torch.arange(self.particles)
            rank_exp = rank[None, :].expand(self.particles, -1)
            self.neighborhood = (rank_exp == 0).logical_or(rank_exp == rank[:, None])
            self.neighborhood[0] = True
        elif self.topology == Topology.RING:
            rank = torch.arange(self.particles)
            rank_diff = (rank[:, None].repeat(1, self.particles) - rank).abs()
            self.neighborhood = (rank_diff  <= 1).logical_or(rank_diff == self.particles-1)
        else:
            raise NotImplementedError(f"Topology `{self.topology.name}` not implemented.")


    def init_scores(self) -> None:
        self.fitness = self.space(self.position)

        self.particle_best_fitness = self.fitness
        self.particle_best_position = self.position
        
        local_fitness = self.fitness[None, :].expand_as(self.neighborhood).where(
            self.neighborhood,
            torch.inf,
        )

        particle_best_known_fitness_id = local_fitness.argmin(-1)
        self.particle_best_known_fitness = self.fitness[particle_best_known_fitness_id]
        self.particle_best_known_position = self.position[particle_best_known_fitness_id]


    def update_scores(self) -> None:
        self.fitness = self.space(self.position)
        
        mask = self.fitness < self.particle_best_fitness
        self.particle_best_fitness[mask] = self.fitness[mask]
        self.particle_best_position[mask] = self.position[mask]

        local_scores = self.fitness[None, :].expand_as(self.neighborhood).where(
            self.neighborhood,
            torch.inf,
        )

        step_pbks_id = local_scores.argmin(-1)
        step_pbks = self.fitness[step_pbks_id]

        mask = step_pbks < self.particle_best_known_fitness
        self.particle_best_known_fitness[mask] = step_pbks[mask]
        self.particle_best_known_position[mask] = self.position[step_pbks_id][mask]


    def update_velocities(self) -> None:
        r_p, r_g = torch.rand(*self.position.shape, 2).unbind(-1)

        self.velocity = self.momentum * self.velocity + \
            self.cognitive_coefficient * r_p * (self.particle_best_position - self.position) + \
            self.social_coefficient * r_g * (self.particle_best_known_position - self.position)


    def update_positions(self) -> None:
        self.position = self.space.clip(self.position + self.velocity)

        
    def step(self) -> None:
        self.update_velocities()
        self.update_positions()
        self.update_scores()
