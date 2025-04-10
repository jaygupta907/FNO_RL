from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from tap import Tap
from typing import Dict, Tuple, Optional
import os
import json
import gymnasium as gym
import numpy as np
import jax.numpy as jnp
from jaxfluids import InputManager, InitializationManager, SimulationManager
from jaxfluids.data_types.case_setup.solid_properties import VelocityCallable
from jaxfluids_rl.helper_functions import get_advance_fn
from jaxfluids_rl.envs.cylinder2D.callback import DragCoefficientCallback
from typing import Dict, Tuple, Optional, Any
import warnings
from flax.core import FrozenDict
import jax
from jaxfluids.data_types import JaxFluidsBuffers
from jaxfluids.data_types.ml_buffers import ParametersSetup, CallablesSetup, LevelSetSetup
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

Array = jax.Array

class ArgumentParser(Tap):
    learning_rate: float = 0.0002 
    batch_size_per_learner: int = 1000
    number_of_epochs: int = 10 
    num_runners: int = 1
    checkpoint_path: str = "/home/jay/FNO_RL/rllib_model"  

args = ArgumentParser().parse_args()


        

class Cylinder2DEnv(gym.Env):
    """Cylinder 2D environment based on the publication https://www.sciencedirect.com/science/article/pii/S0142727X22000832

    The lenght of an episode is a user-defined integer multiple of the natural vortex shedding period,
    here denoted as T_SHEDDING. By default the length of an episode is 8 * T_SHEDDING.

    The length of a step (i.e., of an action) is a user-specified fraction of a vortex shedding period.
    The length of a step defaults to 0.1 * T_SHEDDING.

    The reward is computed as the negative drag coefficient of the cylinder.

    :param gym: _description_
    :type gym: _type_
    :return: _description_
    :rtype: _type_
    """

    LOW_ACTION = -1.0
    HIGH_ACTION = 1.0
    LOW_OBSERVATION = -1.0
    HIGH_OBSERVATION = 1.0

    # INFLOW CONDITIONS
    RHO_INFTY = 1.0
    U_INFTY = 0.23664319132398464
    P_INFTY = 1.0

    # CYLINDER DIAMETER
    D = 1.0            

    # VORTEX SHEDDING PERIOD
    T_SHEDDING = D / (0.2 * U_INFTY)

    RESOLUTION_DICT = {
        "coarse": {"x": (75, 150, 175), "y": (100, 100, 100)},
        "medium": {"x": (150, 300, 350), "y": (200, 200, 200)},
        "fine": {"x": (300, 600, 700), "y": (400, 400, 400)}
    }

    def __init__(self,env_config: dict) -> None:
        self.episode_length = env_config.get("episode_length", 100)
        self.action_length = env_config.get("action_length", 0.05)
        self.resolution_level = env_config.get("resolution_level", "coarse")
        self.render_mode = env_config.get("render_mode", 'save')


        assert_str = "The episode length must be a positive integer."
        assert isinstance(self.episode_length, int) and self.episode_length > 0, assert_str
        # NOTE the total duration of an episode is an integer-multiple
        # of the shedding period.
        self.end_time = self.episode_length * self.T_SHEDDING

        assert_str = "The action length must be a positive float between 0.0 and 1.0."
        assert isinstance(self.action_length, float) and (self.action_length > 0.0) and (self.action_length < 1.0), assert_str
        self.action_time = self.action_length * self.T_SHEDDING

        render_modes = (None, "save", "show")
        assert_str = f"render_mode must be one of {render_modes}"
        assert self.render_mode in render_modes, assert_str
   

        resolution_levels = ("coarse", "medium", "fine")
        assert_str = f"resolution_level must be one of {resolution_levels}"
        assert self.resolution_level in resolution_levels, assert_str

        dirname = os.path.dirname(os.path.realpath(__file__))
        inputfiles_path = os.path.join(dirname, "inputfiles")
        case_setup_path = os.path.join(inputfiles_path, "case_setup.json")
        numerical_setup_path = os.path.join(inputfiles_path, "numerical_setup.json")

        case_setup = json.load(open(case_setup_path, "r"))
        case_setup["general"]["end_time"] = self.end_time

        # NOTE Set user-specified resolution of the grid
        cells_dict = self.RESOLUTION_DICT[self.resolution_level]
        for xi in ("x", "y"):
            case_setup["domain"][xi]["cells"] = sum(cells_dict[xi])
            for i, cells_i in enumerate(cells_dict[xi]):
                case_setup["domain"][xi]["stretching"]["parameters"][i]["cells"] = cells_i

        # TODO set the restart file, will depend on the chosen
        # grid resolution
        # TODO provide multiple restart files and select one
        # at random
        case_setup["restart"] = {
            "flag": True,
            "file_path": os.path.join(dirname, "restart_files/data_1050.0096224507.h5"),
            "use_time": True,
            "time": 0.0
        }

        case_setup["boundary_conditions"]["primitives"]["west"]["primitives_callable"]["rho"] = self.RHO_INFTY
        case_setup["boundary_conditions"]["primitives"]["west"]["primitives_callable"]["u"] = self.U_INFTY
        case_setup["boundary_conditions"]["primitives"]["west"]["primitives_callable"]["p"] = self.P_INFTY

        self.input_manager = InputManager(
            case_setup, numerical_setup_path)
        self.init_manager = InitializationManager(self.input_manager)
        self.sim_manager = SimulationManager(
            self.input_manager,
            callbacks=DragCoefficientCallback(self.RHO_INFTY, self.U_INFTY)
        )

        self.advance_fn = get_advance_fn(self.sim_manager)

        is_double_precision = self.sim_manager.numerical_setup.precision.is_double_precision_compute
        dtype = np.float64 if is_double_precision else np.float32

        self.action_space = gym.spaces.Box(
            low=self.LOW_ACTION, high=self.HIGH_ACTION, shape=(1,), dtype=dtype
        )
        
        self.observation_space = gym.spaces.Box(
            low=self.LOW_OBSERVATION, high=self.HIGH_OBSERVATION, shape=(2,75), dtype=dtype
        )

        # NOTE Definition of velocity probes around the cylinder        
        r_probes = 0.8
        angles = np.arange(0.0, 2 * np.pi, 2 * np.pi / 30)
        x_probes_circ = r_probes * np.cos(angles)
        y_probes_circ = r_probes * np.sin(angles)

        d_probes = 5.0
        x_probes_grid = np.linspace(1.0, d_probes, 9)
        y_probes_grid = np.linspace(-0.8, 0.8, 5)
        X_probes_grid, Y_probes_grid = np.meshgrid(x_probes_grid, y_probes_grid)
        x_probes_grid = X_probes_grid.flatten()
        y_probes_grid = Y_probes_grid.flatten()

        self.x_probes = np.concatenate([x_probes_circ, x_probes_grid])
        self.y_probes = np.concatenate([y_probes_circ, y_probes_grid])

        # NOTE The rotational velocity of the cylinder is defined
        # via a JAX-Fluids CallablesSetup.
        self.ml_callables = CallablesSetup(
            levelset=LevelSetSetup(
                fluid_solid=FrozenDict(
                    {
                        "velocity": VelocityCallable(
                            u=lambda x,y,t,omega: jnp.where((x >= -2.0) & (x < 4.0) & (y >= -2.0) & (y < 2.0), -omega * y, 0.0),
                            v=lambda x,y,t,omega: jnp.where((x >= -2.0) & (x < 4.0) & (y >= -2.0) & (y < 2.0), +omega * x, 0.0),
                            w=lambda x,y,t,omega: jnp.zeros_like(x),
                        )
                    }
                )
            )
        )

    def step(self, action: np.ndarray | Array) -> Tuple[Array, float, bool, bool, Dict]:
        
        jxf_buffers, cb_buffers = self.state
        physical_simulation_time = jxf_buffers.time_control_variables.physical_simulation_time
        end_time = physical_simulation_time + self.action_time

        ml_parameters = self._wrap_action_for_jxf(action)

        # NOTE Time advance jxf_buffers
        jxf_buffers, cb_buffers = self.advance_fn(
            jxf_buffers=jxf_buffers,
            ml_parameters=ml_parameters,
            ml_callables=self.ml_callables,
            end_time=end_time,
            end_step=int(1e+8)
        )

        jxf_buffers: JaxFluidsBuffers

        physical_simulation_time = jxf_buffers.time_control_variables.physical_simulation_time      

        self.state = (jxf_buffers, cb_buffers)
        
        observation = self._get_obs()

        reward = self._get_reward()

        logger.info(f"Reward: {reward.item()}")
        
        terminated = physical_simulation_time > self.end_time

        truncated = False

        info = self._get_info()

        for k, v in cb_buffers.items():
            if isinstance(self.history[k], float):
                self.history[k] = v
            else:
                self.history[k] = jnp.concatenate([self.history[k], v])

        return observation, reward, terminated, truncated, info

    def reset(
            self,
            seed: Optional[int] = None,
            options: Optional[dict[str, Any]] = None
        ) -> Tuple[Array, Dict]:
        # TODO implement random burn-in time

        jxf_buffers = self.init_manager.initialization()
        self.state = (jxf_buffers, {"t": 0.0, "c_d": 0.0, "c_l": 0.0})
        self.history = {"t": 0.0, "c_d": 0.0, "c_l": 0.0}

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def render(self) -> None:

        if self.render_mode in ("show", "save"):
        
            domain_information = self.sim_manager.domain_information
            equation_information = self.sim_manager.equation_information
            output_writer = self.sim_manager.output_writer

            if domain_information.is_parallel:
                warnings.warn(
                    "You are calling the render method for a parallel simulation. "
                    "This is currently not supported."
                )
                return
            
            jxf_buffers, cb_buffers  = self.state
            velocity_probes = self._get_obs()
            
            nhx, nhy, nhz = domain_information.domain_slices_conservatives
            nhx_, nhy_, nhz_ = domain_information.domain_slices_geometry
            s_velocity = equation_information.s_velocity

            primitives = jxf_buffers.simulation_buffers.material_fields.primitives
            levelset = jxf_buffers.simulation_buffers.levelset_fields.levelset
            volume_fraction = jxf_buffers.simulation_buffers.levelset_fields.volume_fraction

            levelset = levelset[nhx, nhy, 0]
            volume_fraction = volume_fraction[nhx_, nhy_, 0]

            vortZ = output_writer.hdf5_writer._compute_vorticity(primitives[s_velocity])
            vortZ = vortZ[0,:,:,0]

            mask = volume_fraction == 0.0
            vortZ = np.ma.masked_where(mask, vortZ)

            x, y, _ = domain_information.get_device_cell_centers()
            x, y = np.squeeze(x), np.squeeze(y)
            x_id_l, x_id_r = np.argmin(np.abs(x + 2.0)), np.argmin(np.abs(x - 6.0))
            y_id_l, y_id_r = np.argmin(np.abs(y + 2.0)), np.argmin(np.abs(y - 2.0))

            X, Y = np.meshgrid(x[x_id_l:x_id_r], y[y_id_l:y_id_r], indexing="ij")
            vortZ = vortZ[x_id_l:x_id_r, y_id_l:y_id_r]
            abs_max_vortZ = np.max(np.abs(vortZ))
            levelset = levelset[x_id_l:x_id_r, y_id_l:y_id_r]

            fig = plt.figure(layout="constrained", figsize=(8,6))

            gs = GridSpec(4, 2, figure=fig)
            ax = fig.add_subplot(gs[:2,:])
            ax1 = fig.add_subplot(gs[2,0])
            ax2 = fig.add_subplot(gs[2,1])
            ax3 = fig.add_subplot(gs[3,0])
            ax4 = fig.add_subplot(gs[3,1])

            ax.pcolormesh(X, Y, vortZ, cmap="seismic")
            ax.contour(X, Y, levelset, levels=[0.0], colors=["black"], vmin=-abs_max_vortZ, vmax=abs_max_vortZ)
            ax.scatter(self.x_probes, self.y_probes, s=1, c="black")
            ax.set_title("Vorticity field")
            ax.set_xlabel(r"$x$")
            ax.set_ylabel(r"$y$")
            ax.set_aspect("equal")

            ax1.plot(velocity_probes[0,:30], marker=".")
            ax2.plot(velocity_probes[0,30:], marker=".")
            for axi in (ax1, ax2):
                axi.set_ylabel(r"u")

            ax3.plot(velocity_probes[1,:30], marker=".")
            ax4.plot(velocity_probes[1,30:], marker=".")
            for axi in (ax3, ax4):
                axi.set_ylabel(r"v")

            ax1.set_title("Cylinder probes")
            ax2.set_title("Grid probes")

            if self.render_mode == "save":
                plt.savefig("render0.png", bbox_inches="tight", dpi=400)
            else:
                plt.show()
            plt.close()


            fig, ax = plt.subplots(ncols=2, figsize=(8,3))
            ax[0].plot(cb_buffers["t"], cb_buffers["c_d"])
            ax[0].plot(self.history["t"], self.history["c_d"], linestyle="--")
            ax[1].plot(cb_buffers["t"], cb_buffers["c_l"])
            ax[1].plot(self.history["t"], self.history["c_l"], linestyle="--")
            ax[0].set_ylabel(r"$c_D$")
            ax[1].set_ylabel(r"$c_L$")
            for axi in ax:
                axi.set_xlabel(r"$t$")
                axi.set_box_aspect(1.0)
            if self.render_mode == "save":
                plt.savefig("render1.png", bbox_inches="tight", dpi=400)
            else:
                plt.show()
            plt.close()

        else:
            pass

    def close(self) -> None:
        pass

    def _get_obs(self) -> Array:
        
        def _get_pressure_obs(velocity: Array) -> Array:

            domain_information = self.sim_manager.domain_information
            x, y, z = [jnp.squeeze(xi) for xi in domain_information.get_device_cell_centers()]
            x_id = jnp.argmin(jnp.abs(x - self.x_probes[:,None]), axis=1)
            y_id = jnp.argmin(jnp.abs(y - self.y_probes[:,None]), axis=1)
            velocity_probes = velocity[:2,x_id,y_id,0]

            # NOTE gather probes from multiple devices
            is_parallel = domain_information.is_parallel
            if is_parallel:
                domain_size = domain_information.get_device_domain_size()
                is_on_device = \
                    (x[x_id] >= domain_size[0][0]) & (x[x_id] < domain_size[0][1]) \
                    & (y[y_id] >= domain_size[1][0]) & (y[y_id] < domain_size[1][1])
                velocity_probes = jax.lax.psum(velocity_probes * is_on_device, axis_name="i")

            return velocity_probes

        jxf_buffers, _ = self.state
        primitives = jxf_buffers.simulation_buffers.material_fields.primitives
        equation_information = self.sim_manager.equation_information
        s_velocity = equation_information.s_velocity

        is_parallel = False
        if is_parallel:
            obs = jax.pmap(
                _get_pressure_obs,
                axis_name="i",
                in_axes=0,
                out_axes=None
            )(primitives[s_velocity])
        else:
            obs = jax.jit(_get_pressure_obs)(primitives[s_velocity])

        return np.array(obs)

    def _get_reward(self) -> Array:
        
        def compute_reward(cb_buffers: Dict[str, Array]) -> Array:
            t = cb_buffers["t"]
            dt = t[1:] - t[:-1]
            c_d = cb_buffers["c_d"]
            mean_c_d = jnp.sum(0.5 * (c_d[1:] + c_d[:-1]) * dt) / (t[-1] - t[0])
            reward = -mean_c_d
            return reward

        compute_reward = jax.jit(compute_reward)

        _, cb_buffers = self.state
        reward = compute_reward(cb_buffers)

        return reward


    def _get_info(self):
        return {}
    
    def _wrap_action_for_jxf(self, action: np.ndarray | Array) -> ParametersSetup:
        """Wraps the given action into JAX-Fluids container types.

        :param action: _description_
        :type action: np.ndarray | Array
        :return: _description_
        :rtype: Tuple[ParametersSetup, CallablesSetup]
        """
        parameter_setup = ParametersSetup(
            levelset=LevelSetSetup(
                fluid_solid={
                    "velocity": VelocityCallable(
                        u=jnp.array(action),
                        v=jnp.array(action),
                        w=None,
                    )
                }
            )
        )

        return parameter_setup

config = (
    PPOConfig()
    .environment(
        Cylinder2DEnv,
        env_config={
            "episode_length": 100,
            "action_length": 0.05,
            "render_mode": "save"
        },
        render_env=True,
    )
    .env_runners(
        num_env_runners=args.num_runners,
        sample_timeout_s=300.0,
        num_gpus_per_env_runner=0.5
    )
    .training(
        lr=args.learning_rate,
        train_batch_size_per_learner=args.batch_size_per_learner,
        num_epochs=args.number_of_epochs,
        
    )
    .api_stack(
        enable_env_runner_and_connector_v2=False,
        enable_rl_module_and_learner=False
    )
)

# Construct the actual algorithm
ppo = config.build_algo()

for i in range(4):
    results = ppo.train()
    print(results['env_runners']['episode_return_mean'])

# Store the trained algorithm
checkpoint_path = ppo.save(args.checkpoint_path)