"""Gymnasium-compatible multi-agent swarm environment."""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from qswarm.envs.communications import MeshMetrics, SwarmMeshNetwork
from qswarm.envs.dynamics import DroneState, SwarmDynamics
from qswarm.envs.obstacles import Obstacle, ObstacleField
from qswarm.envs.rewards import RewardWeights, SwarmRewardCalculator
from qswarm.envs.sensing import GPSDeniedSensor, RescueTarget, SearchCoverageGrid


class SwarmEnv(gym.Env):
    """Multi-drone swarm navigation and dynamic mesh optimization environment."""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_drones: int = 4,
        arena_width: float = 20.0,
        arena_height: float = 20.0,
        comm_range: float = 8.0,
        sensing_radius: float = 4.0,
        max_steps: int = 60,
        gps_noise_std: float = 0.15,
        scenario: str = "rescue_maze",
        reward_weights: Optional[RewardWeights] = None,
    ):
        super().__init__()
        self.num_drones = num_drones
        self.arena_width = arena_width
        self.arena_height = arena_height
        self.comm_range = comm_range
        self.sensing_radius = sensing_radius
        self.max_steps = max_steps
        self.gps_noise_std = gps_noise_std
        self.scenario = scenario

        # Subsystems
        self.dynamics = SwarmDynamics(
            step_size=1.0,
            arena_bounds=(self.arena_width, self.arena_height),
        )
        self.mesh_network = SwarmMeshNetwork(
            comm_range=self.comm_range,
            line_of_sight_enabled=True,
        )
        self.coverage_grid = SearchCoverageGrid(
            arena_bounds=(self.arena_width, self.arena_height),
            resolution=1.0,
        )
        self.gps_sensor = GPSDeniedSensor(noise_std=self.gps_noise_std)
        self.reward_calculator = SwarmRewardCalculator(reward_weights)

        # Action space: 5 discrete actions per drone
        # 0: STAY, 1: NORTH, 2: SOUTH, 3: EAST, 4: WEST
        self.action_space = spaces.MultiDiscrete([5] * self.num_drones)

        # Single drone observation feature size = 12:
        # [x_norm, y_norm, vx, vy, battery_norm,
        #  radar_N, radar_S, radar_E, radar_W,
        #  target_dx_norm, target_dy_norm, mesh_degree_norm]
        self.obs_dim_per_drone = 12
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.num_drones, self.obs_dim_per_drone),
            dtype=np.float32,
        )

        # State storage
        self.drones: List[DroneState] = []
        self.targets: List[RescueTarget] = []
        self.obstacle_field: ObstacleField = ObstacleField((self.arena_width, self.arena_height))
        self.current_step = 0
        self.last_mesh_metrics: Optional[MeshMetrics] = None
        self.trajectory_history: List[Dict[str, Any]] = []
        self.total_collisions = 0
        self.rng: np.random.Generator = np.random.default_rng()

    def _setup_scenario(self, seed: Optional[int] = None):
        """Construct deterministic obstacles, drone starts, and targets according to scenario."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        obstacles: List[Obstacle] = []
        targets: List[RescueTarget] = []

        if self.scenario == "open_field":
            # Sanity check: no obstacles, straight targets
            targets = [
                RescueTarget(0, self.arena_width * 0.8, self.arena_height * 0.8),
                RescueTarget(1, self.arena_width * 0.8, self.arena_height * 0.2),
            ]
        elif self.scenario == "rescue_maze":
            # Static labyrinth walls requiring coordination and detour
            obstacles = [
                Obstacle(self.arena_width * 0.35, self.arena_height * 0.5, radius=1.6),
                Obstacle(self.arena_width * 0.35, self.arena_height * 0.25, radius=1.6),
                Obstacle(self.arena_width * 0.65, self.arena_height * 0.75, radius=1.6),
                Obstacle(self.arena_width * 0.65, self.arena_height * 0.5, radius=1.6),
                Obstacle(self.arena_width * 0.5, self.arena_height * 0.5, radius=1.2),
            ]
            targets = [
                RescueTarget(0, self.arena_width * 0.85, self.arena_height * 0.85),
                RescueTarget(1, self.arena_width * 0.85, self.arena_height * 0.2),
                RescueTarget(2, self.arena_width * 0.15, self.arena_height * 0.85),
            ]
        elif self.scenario == "dynamic_hazard":
            # Static debris plus moving hazards bouncing in arena
            obstacles = [
                Obstacle(self.arena_width * 0.4, self.arena_height * 0.5, radius=1.5),
                Obstacle(self.arena_width * 0.6, self.arena_height * 0.5, radius=1.5),
                Obstacle(
                    self.arena_width * 0.5,
                    self.arena_height * 0.2,
                    radius=1.2,
                    is_dynamic=True,
                    vx=0.4,
                    vy=0.3,
                ),
                Obstacle(
                    self.arena_width * 0.5,
                    self.arena_height * 0.8,
                    radius=1.2,
                    is_dynamic=True,
                    vx=-0.3,
                    vy=-0.4,
                ),
            ]
            targets = [
                RescueTarget(0, self.arena_width * 0.9, self.arena_height * 0.5),
                RescueTarget(1, self.arena_width * 0.5, self.arena_height * 0.9),
            ]
        elif self.scenario == "mesh_tradeoff":
            # Targets in opposite corners forcing tension between target progress & mesh link
            obstacles = [
                Obstacle(self.arena_width * 0.5, self.arena_height * 0.5, radius=2.2),
            ]
            targets = [
                RescueTarget(0, self.arena_width * 0.9, self.arena_height * 0.9),
                RescueTarget(1, self.arena_width * 0.9, self.arena_height * 0.1),
                RescueTarget(2, self.arena_width * 0.1, self.arena_height * 0.9),
            ]
        elif self.scenario == "scale_test":
            # Larger arena test for 6-8 drones
            for ox in [0.3, 0.5, 0.7]:
                for oy in [0.3, 0.7]:
                    obstacles.append(
                        Obstacle(self.arena_width * ox, self.arena_height * oy, radius=1.4)
                    )
            targets = [
                RescueTarget(0, self.arena_width * 0.85, self.arena_height * 0.85),
                RescueTarget(1, self.arena_width * 0.85, self.arena_height * 0.15),
                RescueTarget(2, self.arena_width * 0.15, self.arena_height * 0.85),
                RescueTarget(3, self.arena_width * 0.5, self.arena_height * 0.5),
            ]
        else:
            # Default fallback: open rescue
            targets = [
                RescueTarget(0, self.arena_width * 0.8, self.arena_height * 0.8),
            ]

        self.obstacle_field = ObstacleField((self.arena_width, self.arena_height), obstacles)
        self.targets = targets

        # Drone start positions (clustered in base station quadrant [1.5-4.5])
        self.drones = []
        base_x, base_y = 3.0, 3.0
        offsets = [
            (0.0, 0.0),
            (1.2, 0.0),
            (0.0, 1.2),
            (1.2, 1.2),
            (-1.2, 0.0),
            (0.0, -1.2),
            (2.4, 0.0),
            (0.0, 2.4),
        ]
        for i in range(self.num_drones):
            dx, dy = offsets[i % len(offsets)]
            pos = np.array([base_x + dx, base_y + dy], dtype=np.float32)
            self.drones.append(
                DroneState(
                    drone_id=i,
                    position=pos,
                    noisy_position=pos.copy(),
                    velocity=np.zeros(2, dtype=np.float32),
                    battery=100.0,
                    is_alive=True,
                )
            )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        scenario_name = options.get("scenario") if options else None
        if scenario_name:
            self.scenario = scenario_name

        self._setup_scenario(seed=seed)
        self.coverage_grid.reset()
        self.current_step = 0
        self.total_collisions = 0
        self.trajectory_history = []

        # Initial graph & coverage
        positions = np.array([d.position for d in self.drones])
        alive_mask = np.array([d.is_alive for d in self.drones])
        graph = self.mesh_network.build_graph(positions, alive_mask, self.obstacle_field)
        self.last_mesh_metrics = self.mesh_network.compute_metrics(graph, int(np.sum(alive_mask)))
        self.coverage_grid.update_coverage(positions, sensing_radius=self.sensing_radius)

        obs = self._get_observations(graph)
        info = self._get_info()
        self._record_telemetry(graph, reward=0.0)
        return obs, info

    def step(self, actions: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1

        # Format actions into list
        if isinstance(actions, np.ndarray):
            actions_list = actions.tolist()
        elif isinstance(actions, (int, np.integer)):
            actions_list = [int(actions)] * self.num_drones
        else:
            actions_list = list(actions)

        # 1. Update dynamic obstacles
        self.obstacle_field.step()

        # 2. Previous distance to active targets for delta-progress computation
        active_targets = [t for t in self.targets if not t.is_reached]
        # Record old distances only for targets that we'll compare after the step
        old_min_dists_map = {}
        for t in active_targets:
            dists = [np.linalg.norm(d.position - t.position) for d in self.drones if d.is_alive]
            if dists:
                old_min_dists_map[t.target_id] = min(dists)

        # 3. Kinematic step for each drone
        step_collisions = 0
        min_obstacle_dist = float("inf")
        energy_spent = 0.0

        for i, drone in enumerate(self.drones):
            if not drone.is_alive:
                continue

            act = actions_list[i] if i < len(actions_list) else 0
            battery_before = drone.battery
            self.dynamics.step_drone(drone, act, self.gps_noise_std, self.rng)
            energy_spent += (battery_before - drone.battery)

            # Check obstacle collisions
            if self.obstacle_field.check_collision(drone.position, collision_padding=0.1):
                drone.in_collision = True
                step_collisions += 1
            else:
                drone.in_collision = False

            # Measure min distance to obstacles for proximity penalty
            for obs in self.obstacle_field.obstacles:
                d_obs = np.linalg.norm(drone.position - obs.position) - obs.radius
                if d_obs < min_obstacle_dist:
                    min_obstacle_dist = d_obs

        # Inter-drone collision check
        alive_drones = [d for d in self.drones if d.is_alive]
        for idx1 in range(len(alive_drones)):
            for idx2 in range(idx1 + 1, len(alive_drones)):
                dist = np.linalg.norm(alive_drones[idx1].position - alive_drones[idx2].position)
                if dist < 0.6:  # Drone physical body radius collision
                    step_collisions += 1

        self.total_collisions += step_collisions

        # 4. Check target captures
        targets_reached_this_step = 0
        for target in active_targets:
            for drone in alive_drones:
                if np.linalg.norm(drone.position - target.position) <= 1.2:
                    target.is_reached = True
                    target.reached_by_drone = drone.drone_id
                    targets_reached_this_step += 1
                    break

        # 5. Measure distance delta progress only for targets still NOT reached this step.
        # BUG FIX: old_min_dists and new_min_dists must correspond to the same targets
        # so we compare per-target using the shared map keyed by target_id.
        target_dist_delta = 0.0
        for target in active_targets:
            if not target.is_reached and target.target_id in old_min_dists_map:
                dists = [np.linalg.norm(d.position - target.position) for d in alive_drones]
                if dists:
                    new_dist = min(dists)
                    target_dist_delta += old_min_dists_map[target.target_id] - new_dist
        target_dist_delta = float(target_dist_delta)

        # 6. Update exploration coverage grid
        positions = np.array([d.position for d in self.drones])
        alive_mask = np.array([d.is_alive for d in self.drones])
        new_cells = self.coverage_grid.update_coverage(positions, self.sensing_radius)

        # 7. Build communication graph and compute mesh metrics
        graph = self.mesh_network.build_graph(positions, alive_mask, self.obstacle_field)
        mesh_metrics = self.mesh_network.compute_metrics(graph, int(np.sum(alive_mask)))
        self.last_mesh_metrics = mesh_metrics

        # 8. Termination checks
        all_targets_reached = all(t.is_reached for t in self.targets)
        all_drones_dead = not any(d.is_alive for d in self.drones)
        is_timeout = self.current_step >= self.max_steps
        terminated = all_targets_reached or all_drones_dead
        truncated = is_timeout and not terminated

        # 9. Compute reward
        step_reward, reward_breakdown = self.reward_calculator.compute_reward(
            targets_reached=targets_reached_this_step,
            target_dist_delta=target_dist_delta,
            new_cells_covered=new_cells,
            mesh_metrics=mesh_metrics,
            collisions_count=step_collisions,
            min_obstacle_distance=min_obstacle_dist,
            energy_spent=energy_spent,
            out_of_bounds=False,
            is_timeout=is_timeout,
        )

        obs = self._get_observations(graph)
        info = self._get_info()
        info["reward_breakdown"] = reward_breakdown
        info["step_reward"] = step_reward

        self._record_telemetry(graph, step_reward)
        return obs, step_reward, terminated, truncated, info

    def _get_observations(self, graph) -> np.ndarray:
        """Construct multi-agent observation matrix."""
        obs = np.zeros((self.num_drones, self.obs_dim_per_drone), dtype=np.float32)
        active_targets = [t for t in self.targets if not t.is_reached]

        for i, drone in enumerate(self.drones):
            if not drone.is_alive:
                continue

            # Normalized perceived position and velocity
            pos_x = (drone.noisy_position[0] / self.arena_width) * 2.0 - 1.0
            pos_y = (drone.noisy_position[1] / self.arena_height) * 2.0 - 1.0
            vx, vy = drone.velocity[0], drone.velocity[1]
            bat_norm = drone.battery / 100.0

            # 4-sector radar distances [N, S, E, W]
            radar = self.obstacle_field.get_radar_readings(
                drone.noisy_position, self.sensing_radius
            )

            # Closest active target relative direction (dx, dy)
            if active_targets:
                target_dists = [
                    (np.linalg.norm(drone.noisy_position - t.position), t)
                    for t in active_targets
                ]
                _, closest_target = min(target_dists, key=lambda x: x[0])
                rel_t = closest_target.position - drone.noisy_position
                rel_t_norm = np.clip(rel_t / (self.arena_width / 2.0), -1.0, 1.0)
            else:
                rel_t_norm = np.zeros(2, dtype=np.float32)

            # Local graph connectivity metric
            node_feat = self.mesh_network.extract_node_features(graph, i)
            deg_norm = node_feat["degree"] / float(max(1, self.num_drones - 1))

            obs[i] = [
                pos_x,
                pos_y,
                vx,
                vy,
                bat_norm,
                radar[0],
                radar[1],
                radar[2],
                radar[3],
                rel_t_norm[0],
                rel_t_norm[1],
                deg_norm,
            ]

        return obs

    def _get_info(self) -> Dict[str, Any]:
        targets_reached = sum(1 for t in self.targets if t.is_reached)
        total_targets = len(self.targets)
        return {
            "step": self.current_step,
            "coverage_ratio": self.coverage_grid.coverage_ratio,
            "targets_reached": targets_reached,
            "total_targets": total_targets,
            "success": targets_reached == total_targets,
            "total_collisions": self.total_collisions,
            "mesh_connected": (
                self.last_mesh_metrics.is_fully_connected if self.last_mesh_metrics else False
            ),
            "num_components": (
                self.last_mesh_metrics.num_components if self.last_mesh_metrics else 0
            ),
            "algebraic_connectivity": (
                self.last_mesh_metrics.algebraic_connectivity if self.last_mesh_metrics else 0.0
            ),
            "average_degree": (
                self.last_mesh_metrics.average_degree if self.last_mesh_metrics else 0.0
            ),
        }

    def _record_telemetry(self, graph, reward: float):
        """Append step telemetry snapshot for visualization and export."""
        edges = [
            {"source": u, "target": v, "weight": float(d.get("weight", 1.0))}
            for u, v, d in graph.edges(data=True)
        ]
        snapshot = {
            "step": self.current_step,
            "drones": [d.copy() for d in self.drones],
            "obstacles": [o.to_dict() for o in self.obstacle_field.obstacles],
            "targets": [t.to_dict() for t in self.targets],
            "edges": edges,
            "metrics": self.last_mesh_metrics,
            "coverage_ratio": self.coverage_grid.coverage_ratio,
            "reward": float(reward),
        }
        self.trajectory_history.append(snapshot)
