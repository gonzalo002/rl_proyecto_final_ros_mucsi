# Librerías estándar de Python
import sys, os
import rospy
from copy import deepcopy
from typing import List, Dict, Tuple, Any, Union
from math import pi

# Librerías de terceros
import gymnasium as gym
import numpy as np
from stable_baselines3.common.utils import set_random_seed
from tf.transformations import quaternion_from_euler
import yaml

# Librerias de ROS
from geometry_msgs.msg import Pose, Point, Quaternion

# Importaciones locales (ajustar la ruta de acuerdo a tu proyecto)
sys.path.append("/home/laboratorio/ros_workspace/src/proyecto_final/src/proyecto_final")
from control_robot import ControlRobot


class ROSEnv(gym.Env):
    def __init__(self, num_cubos_max: int, seed: int = None, visualization:bool = False, save_data:bool = False, verbose:bool = False):
        """
        Inicializa el entorno ROS para manipulación de cubos.
        
        Argumentos:
        num_cubos_max (int): Número máximo de cubos que se generarán y gestionarán en el entorno.
        seed (int, opcional): Semilla para inicializar el generador de números aleatorios (por defecto es None).
        
        Inicializa el espacio de trabajo del robot y las variables necesarias para la planificación de los cubos.
        """
        super(ROSEnv, self).__init__()
        self.control_robot = ControlRobot("robot", train_env=True)

        # rospy.init_node("entorno_robot", anonymous=True)
        self.abs_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:os.path.dirname(os.path.abspath(__file__)).split('/').index('proyecto_final')+1])

        self.num_cubos_max = num_cubos_max
        self.seed = seed
        self.visualization = visualization

        self.action_space = gym.spaces.MultiDiscrete([self.num_cubos_max] * self.num_cubos_max)

        self.observation_space = gym.spaces.Box(
            low=np.array([-1] * 8 * self.num_cubos_max),
            high=np.array([1] * 8 * self.num_cubos_max),
            dtype=np.float64
        )

        self.robot_workspace_values = {"max_x": 0.25, "min_x": -0.25, "max_y": 0.4, "min_y": 0.17, "max_alpha": pi/4, "min_alpha": -pi/4}
        self.j_home = [0, -1.569, 0, -1.569, 0, 0]
        self.j_pre_cube = [-1.502, -1.547, -1.1524, -2.011, 1.573, -0.0144]
        self.pose_cubos = []
        self.pseudo_rands_cubos = []
        self.step_count = 1
        self.flag_save_data = save_data
        self.verbose = verbose
        self.failed_cubes = 0


        self.observation = np.array([-1.0] * 8 * self.num_cubos_max)
        self.reward = 0.0
        self.info = {'Steps': self.step_count,
                    'Failed cubes' : self.failed_cubes,
                    'Fail_percentage' : self.failed_cubes / (self.step_count * 200)}
        self.terminated = False
        self.truncated = False

        self.control_robot.reset_planning_scene()
        self.control_robot.move_jointstates(self.j_home)
        self.reset(seed=self.seed)

    def _get_obs(self):
        observation = np.array([-1.0] * 8 * self.num_cubos_max)
        i = 0
        pose:Pose
        for pose in self.pose_cubos:
            observation[0 + i] = pose.position.x
            observation[1 + i] = pose.position.y
            observation[2 + i] = pose.position.z
            observation[3 + i] = pose.orientation.x
            observation[4 + i] = pose.orientation.y
            observation[5 + i] = pose.orientation.z
            observation[6 + i] = pose.orientation.w
            observation[7 + i] = 0.0
            i += 8
        return observation

    def __sample_new_cube_value(self, max_x: float, min_x: float, 
                            max_y: float, min_y: float, 
                            max_alpha: float, min_alpha: float) -> np.ndarray:
        """
        Sample a new cube value for x, y and alpha.
        Input:
            max_x: float, maximum value for x.
            min_x: float, minimum value for x.
            max_y: float, maximum value for y.
            min_y: float, minimum value for y.
            max_alpha: float, maximum value for alpha.
            min_alpha: float, minimum value for alpha.
        Output:
            np.ndarray, new cube value for x, y and alpha.
        """
        
        if max_x < min_x or max_y < min_y or max_alpha < min_alpha:
            raise ValueError("max values must be greater than min values")
        
        success = False
        
        while not success:
            # x y oz
            pseudo_rands = np.random.rand(3)
            pseudo_rands[0] = np.interp(pseudo_rands[0], [0, 1], [min_x, max_x])
            pseudo_rands[1] = np.interp(pseudo_rands[1], [0, 1], [min_y, max_y])
            pseudo_rands[2] = np.interp(pseudo_rands[2], [0, 1], [min_alpha, max_alpha])

            if (pseudo_rands[0] ** 2) + (pseudo_rands[1] ** 2) <= 0.48:
                if self.pseudo_rands_cubos != []:
                     # Verificar que el nuevo cubo esté a una distancia mínima de 0.03 metros de los cubos existentes
                    success = True
                    for pose in self.pseudo_rands_cubos:
                        dist = np.sqrt((pose[0] - pseudo_rands[0]) ** 2 + (pose[1] - pseudo_rands[1]) ** 2)
                        if dist < 0.05:
                            success = False
                            break  # Si la distancia es menor a 0.03, volver a intentar
                else:
                    success = True  
        
        if pseudo_rands[0] > 0:
            pseudo_rands[2] -= pi/2

        self.pseudo_rands_cubos.append(deepcopy(pseudo_rands))
        return pseudo_rands

    def __añadir_cubos_a_planificacion(self, cubos: List[np.ndarray]) -> None:
        for i, cubo in enumerate(cubos):
            pose_cubo = Pose(position=Point(x=cubo[0], y=cubo[1], z=0.01), 
                             orientation=Quaternion(*quaternion_from_euler(pi, 0, cubo[2], 'sxyz')))
            self.pose_cubos.append(pose_cubo)
            if self.visualization == True:
                self.control_robot.add_box_obstacle(f"Cubo_{i}", pose_cubo, (0.025, 0.025, 0.025))

    def step(self, action: List[int]) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.reward = 0.0
        if np.unique(action).size != len(action):
            self.reward -= 20.0 * abs(len(action) - np.unique(action).size)

        total_time = 0  # Total time taken for all actions

        for act in action:
            selected_pose:Pose = deepcopy(self.pose_cubos[act])
            selected_pose.position.z = 0.32
            if self.visualization:
                self.control_robot.scene.remove_world_object(f'Cubo_{int(act)}')
            trayectory_tuple = self.control_robot.plan_pose(selected_pose)
            if trayectory_tuple[0] != True: 
                self.reward -= 10.0  # Penaliza por fallar en la planificación
                self.failed_cubes += 1
                if not self.flag_save_data:
                    save_data = {'step' : self.step_count,
                                    'Pose' : {
                                    'pose_position_x' : float(selected_pose.position.x),
                                    'pose_position_y' : float(selected_pose.position.y),
                                    'pose_position_z' : float(selected_pose.position.z),
                                    'pose_orientation_x' : float(selected_pose.orientation.x),
                                    'pose_orientation_y' : float(selected_pose.orientation.y),
                                    'pose_orientation_z' : float(selected_pose.orientation.z),
                                    'pose_orientation_w' : float(selected_pose.orientation.w)},
                                    'action' : str(action),
                                    'selected_action' : int(act)}
                    with open(f'{self.abs_path}/data/rl/yaml_logs/cubos_fallidos', '+a') as f:
                        yaml.dump(save_data, f, default_flow_style=False, sort_keys=False)
            else:
                tiempo = trayectory_tuple[1].joint_trajectory.points[-1].time_from_start
                tiempo_seg = tiempo.to_sec()

                if tiempo_seg > 3:  # Penaliza si el tiempo es excesivo
                    self.reward -= 1.0
                else:  # Recompensa por tiempo dentro del rango esperado
                    self.reward += 0.5

                total_time += tiempo_seg  # Acumula el tiempo total
            if self.visualization:
                self.control_robot.add_box_obstacle(f"Cubo_{int(act)}", self.pose_cubos[act], (0.025, 0.025, 0.025))

        self.reward -= total_time
        avg_time = total_time / len(action) if len(action) > 0 else 4
        self.reward += (4 - avg_time)  # La recompensa aumenta cuando el tiempo promedio es cercano a 4 segundos

        self.terminated = True

        self.step_count += 1

        if self.verbose:
            print(f'\nStep {self.step_count} - Completado con Recompensa {self.reward} - Accion {str(action)}')
        
        return self.observation, self.reward, self.terminated, self.truncated, self.info

    def reset(self, *, seed: Union[int, None] = None, options: Union[Dict[str, Any], None] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        if not seed is None:
            self.seed = seed
        if not self.seed is None:
            set_random_seed(self.seed)

        variables_cubos = []
        self.pose_cubos = []
        self.pseudo_rands_cubos = []
        self.terminated = False

        if self.visualization == True:
            self.control_robot.reset_planning_scene()

        self.control_robot.move_jointstates(self.j_pre_cube)

        for _ in range(self.num_cubos_max):
            variables_cubos.append(self.__sample_new_cube_value(**self.robot_workspace_values))

        self.__añadir_cubos_a_planificacion(variables_cubos)
        self.observation = self._get_obs()

        return self.observation, self.info

if __name__ == '__main__':
    # Probamos a ejecutar con 2 cubos
    env = ROSEnv(num_cubos_max=2, visualization=True)

    accion = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(accion)
    env.reset()