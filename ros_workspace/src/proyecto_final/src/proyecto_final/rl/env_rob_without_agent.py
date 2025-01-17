#!/usr/bin/python3

# Librerías estándar de Python
import sys, os
from copy import deepcopy
from typing import List, Dict, Tuple, Any, Union

# Librerías de terceros
import gymnasium as gym
import numpy as np
from stable_baselines3.common.utils import set_random_seed
from proyecto_final.msg import IdCubos
from moveit_msgs.msg import RobotTrajectory, RobotState
from trajectory_msgs.msg import JointTrajectory

# Librerias de ROS
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import JointState

# Librerías propias
from proyecto_final.control_robot import ControlRobot


class ROSEnv(gym.Env):
    def __init__(self, cubos:List[IdCubos], orden_figura:list):
        """
        Constructor de la clase ROSEnv.
            @param num_cubos_max: int, número máximo de cubos a colocar.
            @param seed: int, semilla para la generación de números aleatorios.
            @param visualization: bool, indica si se desea visualizar el entorno.
            @param save_data: bool, indica si se desea guardar los datos.
            @param verbose: bool, indica si se desea mostrar mensajes en consola.
        """
        super(ROSEnv, self).__init__()

        # Variables Entorno ROS
        self.control_robot = ControlRobot("robot", train_env=True)

        # Variables Generales
        self.abs_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:os.path.dirname(os.path.abspath(__file__)).split('/').index('proyecto_final')+1])

        # Variables Entorno
        self.cube_limit = 25 # Se pone un limite de 25 cubos, de esta forma un mismo agente puede funcionar con hasta 25 cubos
        self.cubos = cubos
        self.figure_order = orden_figura
        self.n_cubos = len(orden_figura)

        # Definición de los espacios de acción y observación
        self.action_space = gym.spaces.Discrete(self.cube_limit)

        self.observation_space = gym.spaces.Box(
            low=np.array([[-1]* 8] * (self.cube_limit+1)),
            high=np.array([[-1]* 8] * (self.cube_limit+1)),
            dtype=np.float64
        )

        # Variables de Trayectorias
        self.j_link_1:JointState = self.control_robot.read_from_yaml(f'{self.abs_path}/data/trayectorias/master_positions', 'J_LINK_1')
        
        # Variables de Step
        self.action_buffer = set() # Set para las acciones realizadas
        self.orden_cubos = []
        self.total_time = 0  # Total time taken for all actions
        self.colors_found = False
        self.failed_attempts = 0
        self.available_cubes = [0]*5
        self.cubos_recogidos = 0


        # Variables de Seguimiento
        self.observation = np.array([[-1.0]*8] * (self.cube_limit+1))
        self.reward = 0.0
        self.info = {}
        self.done = False
        self.truncated = False

    def _get_obs(self):
        pose:IdCubos
        for i, cubo in enumerate(self.cubos):
            observation = np.array([-1.0] * 8)
            pose:Pose = cubo.pose
            observation[0] = pose.position.x
            observation[1] = pose.position.y
            observation[2] = pose.position.z
            observation[3] = pose.orientation.x
            observation[4] = pose.orientation.y
            observation[5] = pose.orientation.z
            observation[6] = pose.orientation.w
            observation[7] = cubo.color

            self.observation[i] = observation

        if self.cubos_recogidos < self.n_cubos:
            self.observation[-1, -1] = self.figure_order[self.cubos_recogidos]


    def _get_info(self):
        self.info = {}
        if self.terminated:
            self.info = {"orden_cubos": self.orden_cubos,
                         'trayectorias' : self.cube_trajectories}
            return self.info
        self.info = {'orden_cubos': self.orden_cubos, 
                     'trayectorias' : self.cube_trajectories}          

    def _test_figure(self) -> bool:
        '''
        Comprueba que la figura que el usuario quiere crear se puede realizar, es decir, 
        si se disponen de suficientes cubos.
            @return bool: Booleano que indica si se puede hacer o no.
        '''
        # Inicializamos contador de cubos (R, V, A, A, G)
        self.figure_cubes = [0, 0, 0, 0, 0]

        for cube in self.cubos:
            self.available_cubes[cube.color] += 1
        
        # Comprobamos si la longitud de la lista es 5
        if len(self.available_cubes) < 5:
            self.available_cubes.append(0)

        # Contamos los valoeres únicos en la matriz (se devuelve en orden)
        valores_unicos, conteo = np.unique(self.figure_order, return_counts=True)
        for i, color_id in enumerate(valores_unicos):
            if int(color_id) == 4:
                if conteo[i] > self.available_cubes[color_id]:
                    return False
                    
            elif int(color_id) != -1:
                # Se comprueba cuántos cubos de cada color se necesitan
                self.figure_cubes[color_id] = conteo[i]
                if self.figure_cubes[color_id] > self.available_cubes[color_id]:
                    return False

                # Se suma cuantos cubos sobran de cada color para saber los cubos grises disponibles    
                self.available_cubes[4] += self.available_cubes[color_id] - self.figure_cubes[color_id]

        return True    
    
    def step(self, action:int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Realiza un paso en el entorno.
            @param action: List[int], acción a realizar.
            @return: Tuple[np.ndarray, float, bool, bool, dict], observación, recompensa, si el episodio ha terminado, si la acción ha sido truncada y la información adicional.
        """
        if self.failed_attempts >= 10:
            self.terminated = True
            self.reward = -100.0
            return self.observation, self.reward, self.done, self.truncated, self.info
        
        figure_color = int(self.observation[-1, -1])
        
        if figure_color == -1:
            self.cubos_recogidos += 1
        
        if figure_color == 4: # Color desconocido
            cubos_grises = (np.array(self.available_cubes[:-1]) - np.array(self.figure_cubes[:-1]))
            figure_color = np.nonzero(cubos_grises != 0)[0][0]
            self.cubos_recogidos += 1
        
        for cube in reversed(self.cubos): # Cubos en mesa de trabajo
            cube:IdCubos
            if not cube.color == figure_color:
                continue
            
            selected_cube = cube
        
            cube_pose:Pose = selected_cube.pose # Obtiene la pose del cubo 
            cube_pose.position.z = 0.237 # Ajusta la altura del cubo para recogerlo

            prev_pose:Pose = deepcopy(cube_pose) # Copia la pose del cubo seleccionado
            prev_pose.position.z = 0.28 # Ajusta la altura del cubo para recogerlo

            self.control_robot.scene.remove_world_object(f'Cubo_{selected_cube.id}')

            start_state = RobotState()
            start_state.joint_state.position = self.j_link_1

            self.control_robot.move_group.set_start_state(start_state)

            trayectory_tuple:Tuple[bool, RobotTrajectory, float, bool] = self.control_robot.move_group.plan(prev_pose)

            if trayectory_tuple[0] != True: 
                self.control_robot.add_box_obstacle(f'Cubo_{selected_cube.id}', cube.pose, size=(0.025, 0.025, 0.025))
                self.failed_attempts += 1
                self.reward -= (self.cubos_recogidos-self.n_cubos)*-5.0
                return self.observation, self.reward, self.done, self.truncated, self.info
            
            trayectory_header:list = trayectory_tuple[1].joint_trajectory.header
            trayectory_points:list = trayectory_tuple[1].joint_trajectory.points
            trayectory_joint_names:list = trayectory_tuple[1].joint_trajectory.joint_names
    
            start_state = RobotState()
            start_state.joint_state.position = trayectory_points[-1].positions

            self.control_robot.move_group.set_start_state(start_state)
            trayectory_tuple:Tuple[bool, RobotTrajectory, float, bool] = self.control_robot.move_group.plan(cube_pose)
            
            if trayectory_tuple[0] != True: 
                self.control_robot.add_box_obstacle(f'Cubo_{selected_cube.id}', cube.pose, size=(0.025, 0.025, 0.025))
                self.failed_attempts += 1
                self.reward -= (self.cubos_recogidos-self.n_cubos)*-5.0
                return self.observation, self.reward, self.done, self.truncated, self.info
            
            pose_sim = deepcopy(selected_cube.pose)
            pose_sim.position.z = 0.0125
            self.control_robot.add_box_obstacle(f'Cubo_{selected_cube.id}', cube.pose, size=(0.025, 0.025, 0.025))
            
            for point in trayectory_tuple[1].joint_trajectory.points:
                trayectory_points.append(point)
            
            trajectory = JointTrajectory()
            trajectory.header = trayectory_header
            trajectory.joint_names = trayectory_joint_names
            trajectory.points = trayectory_points
            
            
            self.figure_cubes[figure_color] -= 1
            self.available_cubes[figure_color] -= 1
            self.cubos_recogidos += 1

            self.orden_cubos.append(deepcopy(selected_cube))
            self.cube_trajectories.append(trajectory)
            
            self.cubos[cube.id].color = -1
            self.cubos[cube.id].pose = Pose(Point(-1.0, -1.0, -1.0),
                                            Quaternion(-1.0, -1.0, -1.0, -1.0))
            
            if self.cubos_recogidos >= self.n_cubos:
                self.done=True
            
            if self.done:
                self.reward = 0.0  # La recompensa aumenta cuando el tiempo promedio es cercano a 4 segundos
                return self.observation, self.reward, self.done, self.truncated, self.info
                
            self._get_obs()
            self._get_info()
            
            return self.observation, self.reward, self.done, self.truncated, self.info

    def reset(self, seed: Union[int, None] = None, options: Union[Dict[str, Any], None] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super(ROSEnv, self).reset(seed=seed, options=options)
        if seed is None:
            pass
        if not seed is None:
            set_random_seed(seed)

        self.taken_actions = set()
        self.orden_cubos = []
        self.cube_trajectories = []
        self.cubos_recogidos = 0 # Cubos cogidos
        self.available_cubes = [0]*5
        self.failed_attempts = 0
        self.reward = 0.0
        self.terminated = False
        self.truncated = False

        self._test_figure()
        
        self._get_obs()
        self._get_info()
        
        return self.observation, self.info

if __name__ == '__main__':
    # Probamos a ejecutar con 2 cubos
    n_cubos_max = 10
    pose_cubo = Pose(position = Point(0, .25, 0.1), orientation = Quaternion(1.0, 0.0, 0.0, 0.0))
    cubo = IdCubos(pose=pose_cubo, color=0)
    env = ROSEnv(cubos=[cubo], orden_figura=[0])
    env.reset()
    terminated = False
    truncated = False
    while not terminated and not truncated:
        accion = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(accion)
