from typing import List

import cv2
import numpy as np
import torch
from torch import Tensor

from space import BoxSpace
from pso import PSO, PSOConfig, Topology

class Renderer:

    class Box:
        def __init__(self, x: int, y: int, width: int, height: int) -> None:
            self.x = x
            self.y = y
            self.width = width
            self.height = height

            self.size = torch.tensor([width, height])
            self.offset = torch.tensor([x, y])


        def x_space(self, normalized: bool = True) -> Tensor:
            x = torch.arange(0, self.width)
            if normalized:
                return x / self.width
            return x + self.x


        def y_space(self, normalized: bool = True) -> Tensor:
            y = torch.arange(0, self.height)
            if normalized:
                return y / self.height
            return y + self.y


        def grid(self, noramalized: bool = True) -> tuple[Tensor, Tensor]:
            x_space = self.x_space(noramalized)
            y_space = self.y_space(noramalized)

            return torch.meshgrid(x_space, y_space, indexing='xy')


    def __init__(
        self,
        width: int,
        height: int,
        dark_theme: bool,
        pso_config: PSOConfig,
        history_length: int = 10,
        draw_com: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.dark_theme = dark_theme
        self.space = pso_config.space
        assert isinstance(self.space, BoxSpace)
        assert self.space.d == 2
        self.pso_config = pso_config
        self.draw_com = draw_com
        
        self.bg_color = (30, 30, 30) if self.dark_theme else (225, 225, 225)
        self.text_color = (225, 225, 225) if self.dark_theme else (30, 30, 30)
        self.image = np.full((self.height, self.width, 3), self.bg_color, dtype=np.uint8)
        
        # Instantiate PSO using the provided configuration
        self.pso = PSO(self.pso_config)

        # Create the layout for BoxSpace and gradient
        self.init_image()

        # Init particles tracking
        self.history_length = history_length
        self.history_buffer = torch.full((self.pso.particles, history_length, 2), torch.nan)
        self.save_positions()
        


    def init_image(self) -> None:
        # Define margin space (5% of width or height, whichever is smaller)
        min_margin = int(0.05 * min(self.width, self.height))
        
        # Define gradient bar width (5% of width for the z-axis and its labels)
        gradient_bar_width = 2 * min_margin
        
        # Calculate available space for the BoxSpace (it must be a square)
        available_width = self.width - gradient_bar_width - 3 * min_margin # side margins + between space and gradient bar margin
        available_height = self.height - 2 * min_margin

        box_size = min(available_width, available_height)  # Square space for the BoxSpace
        
        total_width = box_size + min_margin + gradient_bar_width
        total_height = box_size

        x_margin = (self.width - total_width) // 2
        y_margin = (self.height - total_height) // 2
        
        # Create the base image for BoxSpace
        
        self.space_box = Renderer.Box(x_margin, y_margin, box_size, box_size)
        self.gradient_box = Renderer.Box(x_margin + box_size + min_margin, y_margin, gradient_bar_width, box_size)

        self.draw_gradient_bar(*self.draw_space(5), 5)


    def draw_space(self, ticks: int = 5) -> tuple[float, float]:

        grid_x, grid_y = self.space_box.grid()

        grid_x = grid_x * self.space.dimensions[0] + self.space.mins[0]
        grid_y = grid_y.flip(0) * self.space.dimensions[1] + self.space.mins[1]

        # Create a grid of pixel coordinates
        positions = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

        # Evaluate function f on sampled positions
        values = self.space(positions).view(self.space_box.width, self.space_box.height).cpu().numpy()
        min_val, max_val = values.min(), values.max()

        # Normalize values to 0-255 for visualization
        values = (values - min_val) / (max_val - min_val) * 255
        values = values.astype(np.uint8)

        self.image[
            self.space_box.y:self.space_box.y + self.space_box.height,
            self.space_box.x:self.space_box.x + self.space_box.width,
        ] = cv2.applyColorMap(values, cv2.COLORMAP_JET)

        # Draw ticks and labels on the BoxSpace
        x_ticks_line = self.space_box.y + self.space_box.height
        for i in range(ticks):
            x_value = i / (ticks - 1) * self.space.dimensions[0] + self.space.mins[0]
            y_value = (ticks - i - 1) / (ticks - 1) * self.space.dimensions[1] + self.space.mins[1]
            # X-axis ticks
            x_pos = self.space_box.x + i * (self.space_box.width // (ticks - 1))
            cv2.line(self.image, (x_pos, x_ticks_line - 5), (x_pos, x_ticks_line + 5), self.text_color, 2)
            cv2.putText(self.image, f"{x_value:.2f}", (x_pos - 20, x_ticks_line + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

            # Y-axis ticks
            y_pos = self.space_box.y + i * (self.space_box.width // (ticks - 1))
            cv2.line(self.image, (self.space_box.x - 5, y_pos), (self.space_box.x + 5, y_pos), self.text_color, 2)
            cv2.putText(self.image, f"{y_value:.2f}", (self.space_box.x - 60, y_pos + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1)

        return min_val, max_val 

        
    def draw_gradient_bar(self, min_val: float, max_val: float, ticks: int = 5) -> None:
        gradient = self.gradient_box.y_space().cpu().numpy()[::-1]

        # Map the gradient to a color map (e.g., Plasma)
        color_map = cv2.applyColorMap((gradient * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Put gradient on image
        self.image[
            self.gradient_box.y:self.gradient_box.y + self.gradient_box.height,
            self.gradient_box.x:self.gradient_box.x + self.gradient_box.width,
        ] = color_map

        tick_positions = np.linspace(0, self.gradient_box.height, ticks, dtype=int) + self.gradient_box.y
        
        # Add tick marks and values along the left side of the gradient bar
        for i, tick in enumerate(tick_positions):
            value = (ticks - i - 1) / (ticks - 1) * (max_val - min_val) + min_val  # Normalize the value between 0 and 1
            text = f"{value:.2f}"
            x_line = self.gradient_box.x + self.gradient_box.width 
            cv2.putText(self.image, text, (x_line + 7, tick + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.text_color, 1, cv2.LINE_AA)
            cv2.line(self.image, (x_line - 5, tick), (x_line + 5, tick), self.text_color, 2)


    def save_positions(self) -> None:
        self.history_buffer[:, 1:] = self.history_buffer[:, :-1]
        self.history_buffer[:, 0] = self.pso.positions.clone()


    def step(self) -> None:
        self.pso.step()
        self.save_positions()
        self.draw()


    def to_draw_space(self, positions: Tensor) -> Tensor:
        # Normailze in 0-1
        positions = (positions - self.space.mins) / self.space.dimensions

        # To box space
        positions = ((positions * self.space_box.size) + self.space_box.offset).type(torch.int)
        positions[..., 1] = self.height - positions[..., 1]

        return positions
        

    def draw(self) -> None:
        self.step_image = self.image.copy()
        positions = self.history_buffer[:, self.history_buffer.isnan().logical_not().all(-1).all(0)]
        positions = self.to_draw_space(positions).tolist()

        for particle_positions in positions:
            cv2.circle(self.step_image, particle_positions[0], (self.history_length * 2) // 3, self.text_color, thickness=-1)
            for i in range(len(particle_positions) - 1):
                cv2.line(self.step_image, particle_positions[i], particle_positions[i+1], self.text_color, max(1, (self.history_length - i - 1) // 2))

        if self.draw_com:
            for i in range(self.pso.particles):
                for j in range(self.pso.particles):
                    if self.pso.neighbors[i, j]:
                        cv2.line(self.step_image, positions[i][0], positions[j][0], self.text_color, 1)
        


if __name__ == "__main__":
    r = Renderer(
        1600,
        1000,
        True,
        PSOConfig(
            50,
            Topology.GLOBAL,
            0.5,
            0.2,
            0.2,
            BoxSpace(
                torch.tensor([-10, -10]),
                torch.tensor([10, 10]),
                lambda x: ((x.abs() - 2) ** 2).sum(-1) ** 0.5 + (x * 10).sin().sum(-1) / 5,
            ),
        )
    )

    cv2.namedWindow("main", cv2.WINDOW_NORMAL)
    k = -1
    while k not in [27, ord('q')]:
        r.step()
        cv2.imshow("main", r.step_image)
        k = cv2.waitKey(100)

    cv2.destroyAllWindows()
    
