import cv2
import torch

from space import BoxSpace
from pso import PSOConfig, Topology
from renderer import Renderer

space = BoxSpace(
    mins = torch.tensor([-10, -10]),
    maxs = torch.tensor([10, 10]),
    f = lambda x: ((x.abs() - 2) ** 2).sum(-1) ** 0.5 + (x * 10).sin().sum(-1) / 5,
)

pso_config = PSOConfig(
    particles = 10,
    topology = Topology.STAR,
    momentum = 0.5,
    cognitive_coefficient = 0.2,
    social_coefficient = 0.6,
    space = space,
)

r = Renderer(
    width = 1600,
    height = 1000,
    dark_theme = True,
    pso_config = pso_config,
)

cv2.namedWindow("main", cv2.WINDOW_NORMAL)
k = -1
while k not in [27, ord('q')] and cv2.getWindowProperty('main', 1) >= 0:
    r.step()
    cv2.imshow("main", r.step_image)
    k = cv2.waitKey(100)

cv2.destroyAllWindows()
