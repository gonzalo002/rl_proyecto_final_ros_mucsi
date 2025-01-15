#!/usr/bin/python3

# Importaciones Propias
from proyecto_final.control_robot import ControlRobot
from proyecto_final.MasterClient import MasterClient
from proyecto_final.funciones_auxiliares import crear_mensaje

# Importaciones de Datos
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from proyecto_final.msg import IdCubos, CubosResult, HandData, RLResult
from proyecto_final.vision.generacion_figura import FigureGenerator

# Importaciones de ROS
import rospy

# Importaciones de Python
import os
from copy import deepcopy
from typing import List

# Importaciones de Terceros
import numpy as np

class SecuenceCommander:
    ''' 
    Clase que controla las secuencias de comandos del robot.
        @method __init__ - Inicializa la clase y configura los parámetros de Aruco.
        @method _moveJoint - Mueve las articulaciones del robot a una posición especificada usando un JointState.
        @method _movePose - Mueve el robot a una posición específica (pose).
        @method _empty_workspace - Libera el espacio de trabajo del robot, moviendo y descartando los cubos encontrados.
        @method _test_figure - Comprueba que la figura que el usuario quiere crear se puede realizar, es decir, si se disponen de suficientes cubos.
        @method create_figure - Genera la figura a partir de los cubos disponibles en el espacio de trabajo.
        @method _pick_cube - Mueve el robot para agarrar un cubo y lo adjunta a la pinza.
        @method _pick_rl_secuence - Mueve el robot para agarrar un cubo y lo adjunta a la pinza.
        @method _drop_cube - Suelta el cubo en la posición deseada.
        @method detect_figure - Rastrea la posición de los cubos utilizando la cámara superior.
        @method track_cubes - Rastrea la posición de los cubos utilizando la cámara superior.
        @method _cube_to_aruco - Convierte las posiciones de los cubos a coordenadas ArUco.
        @method rob_camera_calibration - Realiza la calibración de la cámara del robot y guarda la pose de ArUco.
        @method _move_to_calibration - Mueve el robot a la posición de calibración.
        @method _put_calibration_pin - Coloca el puntero de calibración en la posición deseada.
        @method _leave_calibration_space - Deja el espacio de calibración y guarda la pose de ArUco.
        @method _drop_calibration_pin - Suelta el puntero de calibración.
        @method _free_camera_space - Libera el espacio de la cámara moviendo el robot a una posición fuera del área de visión.
        @method _hand_control_active - Activa el control del robot por medio de la mano.
        @method _hand_control_deactivate - Desactiva el control del robot por medio de la mano.
        @method _control_robot_by_hand - Controla el robot por medio de la mano.
        @method hand_data_callback - Callback para recibir los datos de la mano.
    '''
    def __init__(self, simulation:bool = False) -> None:
        '''
        Inicializa la clase.
            @param simulation (bool) - Indica si el robot está en modo simulación.
            @return None
        '''
        # --- VARIABLES GENERALES ---
        self.abs_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:os.path.dirname(os.path.abspath(__file__)).split('/').index('proyecto_final')+1])
        self.simulation = simulation
        self.message = None
        self.message_type = None
        self.name = "RobotMain" # Nombre del nodo

        # --- VARIABLES AUTO ---
        self.discarded_cubes = [0, 0, 0, 0] # RGBY
        self.matriz3D = np.full((5, 5, 5), -1, dtype=int) # Matriz 3D de la figura
        self.cubes:List[IdCubos] = [] # Lista de cubos
        self.cube_size = 0.025 # Tamaño de los cubos
        self.cube_separation = 0.02 # Separación entre cubos
        self.altura_z = 0.237 # Altura de agarre de los cubos
        self.available_cubes = [] # Cubos disponibles
        self.figure_cubes = [0, 0, 0, 0, 0] # Cubos de la figura
        self.color_name = ["rojo", "verde", "azul", "amarillo", "gris"] # Nombres de los colores
        self.workspace_range = {} # Rango de trabajo

        # --- VARIABLES MANUAL ---
        self.hand_control = False
        self.hand_detected = False
        self.hand_range = {'x_min' : -0.19061986164757674, 'x_max' : 0.23207591013811013, 
                            'y_min' : 0.23159845770381593, 'y_max' : 0.44488890481150184,
                            'z_min' : 0.2, 'z_max' : 0.398362} # Rango de la mano
        self.hand_pose = [0.0, 0.0, 0.0] # x, y, z
        self.hand_gesture = {'is_open' : False, 'is_peace' : False, 'is_dino' : False} # Tipos de Gestos

        # --- Clases ---
        self.robot = ControlRobot('robot') # Control del Robot
        self.action_client = MasterClient() # Cliente del Servidor Maestro
        self.generator = FigureGenerator() # Generador de Figuras
        rospy.Subscriber('/hand_data', HandData, self.hand_data_callback) # Suscripción a los datos de la mano

        # --- Posiciones Maestras ---
        # Poses
        self.p_figure_origin:Pose = self.robot.read_from_yaml(f'{self.abs_path}/data/trayectorias/master_positions', 'P_MATRIX_ORIGIN')
        self.p_discard_origin:Pose = self.robot.read_from_yaml(f'{self.abs_path}/data/trayectorias/master_positions', 'P_DISCARD_ORIGIN')
        self.p_discard_origin_2:Pose = self.robot.read_from_yaml(f'{self.abs_path}/data/trayectorias/master_positions', 'P_DISCARD_2_ORIGIN')       
        self.p_aruco = self.robot.read_from_yaml(f'{self.abs_path}/data/trayectorias/pose_aruco', 'P_ARUCO')        

        # JointStates
        self.j_link_1:JointState = self.robot.read_from_yaml(f'{self.abs_path}/data/trayectorias/master_positions', 'J_LINK_1')
        self.j_off_camera:JointState = self.robot.read_from_yaml(f'{self.abs_path}/data/trayectorias/master_positions', 'J_OFF_CAMERA')
        self.j_home:JointState = self.robot.read_from_yaml(f'{self.abs_path}/data/trayectorias/master_positions', 'J_HOME')
        self.j_prev_aruco:JointState = self.robot.read_from_yaml(f'{self.abs_path}/data/trayectorias/master_positions', 'J_PRE_ARUCO')
        self.j_discard_origin:JointState = self.robot.read_from_yaml(f'{self.abs_path}/data/trayectorias/master_positions', 'J_DISCARD_ORIGIN')
        self.j_discard_origin_2:JointState = self.robot.read_from_yaml(f'{self.abs_path}/data/trayectorias/master_positions', 'J_DISCARD_2_ORIGIN')
        self.j_hand_origin:JointState = self.robot.read_from_yaml(f'{self.abs_path}/data/trayectorias/master_positions', 'J_HAND_ORIGIN')
        
        # --- Inicialización ---
        self.robot.reset_planning_scene() # Resetea el espacio de trabajo
        self._moveJoint(self.j_home) # Mueve el robot a la posición de inicio
        self.robot.move_gripper(35.0, 40.0) # Abre la pinza del robot
        
        
    def _moveJoint(self, jointstate:list) -> bool:   
        ''' 
        Mueve las articulaciones del robot a una posición especificada usando un JointState.
            @param jointstate (list) - Posición de las articulaciones.
            @return success (bool) - Indica si el movimiento fue exitoso.
        '''      
        success = self.robot.move_jointstates(jointstate)
        
        if success == True:
            self.message_type = "INFO"
            self.message =  crear_mensaje("JointState alcanzada", self.message_type, self.name)
        else:
            self.message_type = "ERROR"
            self.message =  crear_mensaje("JointState inalcanzable", self.message_type, self.name)
        return success
    
    def _movePose(self, pose:list, wait:bool = True) -> bool:    
        ''' 
        Mueve el robot a una posición específica (pose).
            @param pose (list) - Posición a alcanzar.
            @return success (bool) - Indica si el movimiento fue exitoso.
        '''          
        success = self.robot.move_pose(pose, wait=wait)
        
        if success == True:
            self.message_type = "INFO"
            self.message =  crear_mensaje("La pose ha sido alcanzada.", self.message_type, self.name)
        else:
            self.message_type = "ERROR"
            self.message =  crear_mensaje("La pose NO ha sido alcanzada.", self.message_type, self.name)
        return success
    
    
    def _empty_workspace(self) -> None:
        ''' 
        Libera el espacio de trabajo del robot, moviendo y descartando los cubos encontrados.
            @return None
        '''
        if self.cubes == []:
            self.message =  crear_mensaje(f"Los cubos sobre la mesa no han sido definidos", "WARN", self.name)
            return

        self.message_type = "INFO"
        self.message =  crear_mensaje("Limpiando la mesa de trabajo...", self.message_type, self.name)
        
        x_min = self.workspace_range['x_min']
        y_min = self.workspace_range['y_min']        
        x_max = self.workspace_range['x_max']        
        y_max = self.workspace_range['y_max']   
        
        self._moveJoint(self.j_link_1)     
                
        for cube in self.cubes: # Recorremos los cubos
            pose:Pose = deepcopy(cube.pose)
            color = cube.color
            if pose.position.x > x_min and pose.position.x < x_max: # Si el cubo está en el rango de trabajo
                if pose.position.y > y_min and pose.position.y < y_max: 
                    if self._pick_cube(pose, cube.id):
                        if color < 2:
                            discard_joint = self.j_discard_origin
                        elif color < 4:
                            discard_joint = self.j_discard_origin_2
                        if self._moveJoint(discard_joint):
                            if self._drop_cube(make_figure=False, cube_id=cube.id, matrix_position=[0, color, self.discarded_cubes[color]]):
                                self.discarded_cubes[color] += 1


    def _test_figure(self) -> bool:
        '''
        Comprueba que la figura que el usuario quiere crear se puede realizar, es decir, 
        si se disponen de suficientes cubos.
            @return bool: Booleano que indica si se puede hacer o no.
        '''
        if np.all(self.matriz3D == -1):
            self.message =  crear_mensaje(f"La figura toadavía no ha sido definida", "WARN", self.name)
            return False
        
        if self.cubes == []:
            self.message =  crear_mensaje(f"Los cubos sobre la mesa no han sido definidos", "WARN", self.name)
            return False

        # Inicializamos contador de cubos (R, V, A, A, G)
        self.figure_cubes = [0, 0, 0, 0, 0]

        # Comprobamos si la longitud de la lista es 5
        if len(self.available_cubes) < 5:
            self.available_cubes.append(0)

        # Contamos los valoeres únicos en la matriz (se devuelve en orden)
        valores_unicos, conteo = np.unique(self.matriz3D, return_counts=True)
        for i, color_id in enumerate(valores_unicos):
            if color_id == 4:
                if conteo[i] > self.available_cubes[color_id]:
                    self.message_type = "ERROR"
                    self.message =  crear_mensaje(f"No hay suficientes cubos {self.color_name[color_id]} para generar la figura", self.message_type, self.name)
                    return False
                    
            elif color_id != -1:
                # Se comprueba cuántos cubos de cada color se necesitan
                self.figure_cubes[color_id] = conteo[i]
                if self.figure_cubes[color_id] > self.available_cubes[color_id]:
                    self.message_type = "ERROR"
                    self.message =  crear_mensaje(f"No hay suficientes cubos {self.color_name[color_id]} para generar la figura", self.message_type, self.name)
                    return False

                # Se suma cuantos cubos sobran de cada color para saber los cubos grises disponibles    
                self.available_cubes[4] += self.available_cubes[color_id] - self.figure_cubes[color_id]

        self.message_type = "SUCCESS"
        self.message =  crear_mensaje(f"Se disponen de suficientes cubos para generar la figura", self.message_type, self.name)

        x, z, y = self.matriz3D.shape
        margen = 1.8
        self.workspace_range = {'x_max': self.p_figure_origin.position.x+((self.cube_size + self.cube_separation)*(x+margen)), 
                                'x_min': self.p_figure_origin.position.x-((self.cube_size + self.cube_separation)*margen), 
                                'y_max': self.p_figure_origin.position.y+((self.cube_size + self.cube_separation)*(y+margen)), 
                                'y_min': self.p_figure_origin.position.y-((self.cube_size + self.cube_separation)*margen)}
        return True    
        

    def create_figure(self) -> None:
        ''' 
        Genera la figura a partir de los cubos disponibles en el espacio de trabajo.
            @return None
        '''
        self._empty_workspace() # Limpia el espacio de trabajo antes de crear la figura.
        
        self.message_type = "INFO"
        self.message =  crear_mensaje("Empezando a crear la figura...", self.message_type, self.name)

        self._moveJoint(self.j_link_1)

        x, z, y = self.matriz3D.shape
        figure_order = []

        # Recorremos la matriz para ir creando la figura.
        for j in range(z):
            for k in range(y):
                for i in range(x):
                    figure_order.append(self.matriz3D[i,j,k])
        
        result:RLResult = self.action_client.obtain_cube_order((self.cubes, figure_order))
        
        cube_order = result.cubes_correct_order
        trajectory_cubes = result.cubes_trajectories 

        num_order = []
        self.message_type = "SUCCESS"

        for cube in cube_order: # Recorremos los cubos
            num_order.append(cube.id)
        self.message = crear_mensaje(f'Orden de los Cubos Seleccionado: {num_order}', self.message_type, self.name)

        cont_cubos_dejados = 0
        for j in range(z):
            for k in range(y):
                for i in range(x):
                    cube:IdCubos = cube_order[cont_cubos_dejados] # Cubo a mover
                    cont_cubos_dejados += 1

                    if not self._pick_cube(cube.pose, cube.id): # Si no se puede coger el cubo, se para la secuencia.
                        self.message_type = "ERROR"
                        self.message =  crear_mensaje('El robot no ha podido coger el cubo.', self.message_type, 'RobotMain')
                        break

                    if not self._drop_cube(make_figure=True, cube_id=cube.id, matrix_position=[x-i, y-k, j]):
                        self.message_type = "ERROR"
                        self.message =  crear_mensaje('El robot no ha podido dejar el cubo.', self.message_type, 'RobotMain')
                        break
        
        self._moveJoint(self.j_home) # Movemos el robot a la posición de inicio.


    def _pick_cube(self, pose:Pose, cube_id:int) -> bool:
        ''' 
        Mueve el robot para agarrar un cubo y lo adjunta a la pinza.
            @param pose (Pose) - Posición objetivo del cubo.
            @param cube_id (int) - Identificador único del cubo.
            @return success (bool) - Indica si el movimiento fue exitoso.
        '''
        if not self.simulation:
            gripper, _ = self.robot.get_pinza_state(sleep_before=0.2)
            if gripper < 20.0:  # Si la pinza no está abierta, se abre.
                self.robot.move_gripper(50.0, 5.0, sleep_after=0.8)
        
        pose_previa = deepcopy(pose)
        pose_previa.position.z = 0.3  # Eleva la posición del cubo antes de moverlo

        self._movePose(pose_previa)  # Mueve el robot a la posición previa del cubo.

        # Si el movimiento al cubo es exitoso, se adjunta el cubo a la pinza.
        self.robot.scene.remove_world_object(f'Cubo_{cube_id}')
        if self.robot.move_carthesian_trayectory([pose_previa, pose]):

            if not self.simulation:
                self.robot.move_gripper(0.0, 10.0, sleep_after=1.5)  # Abre la pinza para tomar el cubo
                gripper, effort = self.robot.get_pinza_state(sleep_before=0.2)

                if not effort and gripper < 5:
                    self.message_type = "ERROR"
                    self.message =  crear_mensaje(f"El cubo {cube_id} NO ha sido agarrado.", self.message_type, self.name)
                    return False
                else:
                    self.message_type = "SUCCESS"
                    self.message =  crear_mensaje(f"El cubo {cube_id} ha sido agarrado.", self.message_type, self.name)    

            # Si el movimiento de vuelta es exitoso, vuelve a la posición inicial.
            if self.robot.move_carthesian_trayectory([pose, pose_previa]):

                return True # Secuencia Completada
        return False # Secuencia Fallida


    def _drop_cube(self, make_figure:bool = True, cube_id:int = None, matrix_position:list = [0,0,0]) -> bool:
        ''' 
        Suelta el cubo en la posición deseada.
            @param make_figure (bool) - Determina si el cubo forma parte de una figura o es un descarte.
            @param cube_id (int) - Identificador único del cubo.
            @param matrix_position (list) - Posición 3D para colocar el cubo en una matriz.
            @return success (bool) - Indica si el movimiento fue exitoso.
        '''
        if not self.simulation:
            gripper, effort = self.robot.get_pinza_state(sleep_before=0.2)

            if not effort and gripper > 30:  # Si no hay esfuerzo en la pinza, significa que no se tiene cubo.
                self.message_type = "ERROR"
                self.message = crear_mensaje(f"El cubo {cube_id} NO ha sido agarrado.", self.message_type, self.name)  
                return False

        if len(matrix_position) != 3:  # Verifica que la matriz de posición tenga 3 elementos.
            raise ValueError('Invalid Input, Matrix position len must be 3')

        # Se determina la pose inicial dependiendo si el cubo forma parte de una figura.
        if make_figure:
            pose: Pose = deepcopy(self.p_figure_origin)  # Pose inicial para formar una figura.

        else:
            if matrix_position[1] < 2:
                pose: Pose = deepcopy(self.p_discard_origin)  # Pose inicial para desechar el cubo.

            elif matrix_position[1] < 4:
                matrix_position[1] -= 2
                pose: Pose = deepcopy(self.p_discard_origin_2)  # Pose inicial para desechar el cubo.

        pose.position.z = self.altura_z # Ajuste en la posición Z.

        # Ajuste en la posición X, Y y Z basados en la matriz de posición.
        pose.position.x += ((self.cube_size + self.cube_separation*2) * matrix_position[0])
        pose.position.y += ((self.cube_size + self.cube_separation*2) * matrix_position[1])
        pose.position.z += ((self.cube_size) * matrix_position[2])

        pose_previa: Pose = deepcopy(pose)
        pose_previa.position.z = 0.35  # Ajuste en la altura para la posición previa.

        self._movePose(pose_previa)  # Mueve el robot a la posición previa.

        simulation_pose = deepcopy(pose)
        simulation_pose.position.z = ((self.cube_size) * matrix_position[2])+0.0125
        # Si el movimiento al lugar de dejar el cubo es exitoso, se quita el cubo de la pinza.
        if self.robot.move_carthesian_trayectory([pose_previa, pose]):
            
            if not self.simulation:
                rospy.sleep(0.2)
                self.robot.move_gripper(35.0, 10.0)  # Abre la pinza para soltar el cubo.
                self.message_type = "SUCCESS"
                self.message = crear_mensaje(f"El cubo {cube_id} ha sido soltado.", self.message_type, self.name)  

            # Si no es para formar una figura, se actualiza la pose del cubo en el diccionario.
            if not make_figure:
                self.cubes[cube_id].pose = deepcopy(pose)

            # Si el movimiento de vuelta es exitoso, vuelve a la posición inicial.
            if self.robot.move_carthesian_trayectory([pose, pose_previa]):
                self.robot.add_box_obstacle(f"Cubo_{cube_id}", simulation_pose, (0.025, 0.025, 0.025))
                self._moveJoint(self.j_link_1)

                return True # Secuencia Completada
        
        return False # Secuencia no Completada
                
    
    def detect_figure(self) -> None:
        ''' 
        Rastrea la posición de los cubos utilizando la cámara superior.
        '''
        # Mover el robot fuera del espacio de trabajo
        self.message_type = "INFO"
        self.message = crear_mensaje(f"Detección de Figura comenzada...", self.message_type, self.name)  
        self._free_camera_space()
        
        # Llama al Cube Tracker Action
        self.matriz3D = self.action_client.obtain_figure()
        
        
         
    def track_cubes(self, goal:int=0) -> dict:
        ''' 
        Rastrea la posición de los cubos utilizando la cámara superior.
        '''
        self.robot.reset_planning_scene()
        
        # Mover el robot fuera del espacio de trabajo
        self._free_camera_space()
        
        # Llama al Cube Tracker Action
        cubos = self.action_client.obtain_cube_pose(goal)
        
        # Guardamos las posiciones reales de los cubos
        self.cubes = self._cube_to_aruco(cubos)
        self.available_cubes = list(cubos.color_counter)

            
    def _cube_to_aruco(self, cubos:CubosResult) -> list:
        ''' 
        Convierte las posiciones de los cubos a coordenadas ArUco.
            @param cubos (list) - Lista de cubos con sus posiciones.
            @return cubos (list) - Lista de cubos con las nuevas posiciones ajustadas.
        '''
        cubo:IdCubos
        lista_cubos = []
        for cubo in cubos.cubes_position:
            new_cube = cubo
            new_cube.pose.position.x += self.p_aruco.position.x
            new_cube.pose.position.y += self.p_aruco.position.y  
                   
            self.robot.add_box_obstacle(f"Cubo_{new_cube.id}", new_cube.pose, (0.025, 0.025, 0.025))
            
            new_cube.pose.position.z = self.altura_z
            lista_cubos.append(new_cube)
            
        return lista_cubos
            
            
    def rob_camera_calibration(self) -> None:
        ''' 
        Realiza la calibración de la cámara del robot y guarda la pose de ArUco.
            @return None
        '''
        self._move_to_calibration()
        
        self._put_calibration_pin()
        
        # Solicita al usuario que coloque el robot en la pose de ArUco para la calibración.
        self._leave_calibration_space()

        # Solicita al usuario confirmación para soltar el boli
        self._drop_calibration_pin()  # Abre nuevamente la pinza.


    def _move_to_calibration(self):
        """
        Mueve el robot a la posición de calibración.
        """
        self.robot.move_gripper(20, 40)  # Cierra parcialmente la pinza para la calibración.
        self._moveJoint(self.j_prev_aruco)  # Mueve el robot a la posición previa de ArUco.

        self.message_type = "INPUT"
        self.message =  crear_mensaje("¿Está el puntero de calibración posicionado?", self.message_type, self.name)


    def _put_calibration_pin(self):
        """
        Coloca el puntero de calibración en la posición deseada.
        """
        self.robot.move_gripper(0, 20)  # Cierra la pinza para agarrar el puntero de calibración.
        rospy.sleep(2)  # Espera para estabilizar el movimiento.

        self.message_type = "INPUT"
        self.message =  crear_mensaje("¿Está el robot en la esquina del ArUco?", self.message_type, self.name)
    

    def _leave_calibration_space(self):
        """
        Deja el espacio de calibración y guarda la pose de ArUco.
        """
        # Obtiene y guarda la pose de ArUco del robot.
        self.p_aruco = self.robot.get_pose()
        self.robot.save_in_yaml(f'{self.abs_path}/data/trayectorias/pose_aruco', 'P_ARUCO', self.p_aruco, True)
        
        pose = self.p_aruco # Mirar como solucionar la parte de soltar el boli sin afectar al aruco
        pose.position.z += 0.05
        if self._movePose(pose=pose):
            return self._moveJoint(self.j_off_camera)

        self.message_type = "INPUT"
        self.message =  crear_mensaje("¿Se puede soltar el puntero de calibración?:", self.message_type, self.name)
    

    def _drop_calibration_pin(self):
        self.robot.move_gripper(30, 10)  # Abre nuevamente la pinza.
        self.message_type = "SUCCESS"
        self.message =  crear_mensaje("Calibración realizada con éxito", self.message_type, self.name)


    def _free_camera_space(self):
        ''' 
        Libera el espacio de la cámara moviendo el robot a una posición fuera del área de visión.
            @return None
        '''
        if not self.simulation:
            self.robot.move_gripper(35.0, 5.0, sleep_after=0.8)
        self._moveJoint(self.j_off_camera)  # Mueve el robot a una posición predeterminada fuera del área de la cámara.
    

    def _hand_control_active(self) -> bool:
        """
        Activa el control del robot por medio de la mano
        """
        self.robot.reset_planning_scene() # Resetea el espacio de trabajo

        res = self._moveJoint(self.j_hand_origin) # Mueve el robot a la posición de la mano
        if res:
            self.message_type = "SUCCESS"
            self.message =  crear_mensaje(f"El control por mano ha sido activado", self.message_type, self.name)

            self.hand_control = True
        else:
            self.message_type = "ERROR"
            self.message =  crear_mensaje(f"El control por mano NO ha podido ser activado", self.message_type, self.name)
        return res


    def _hand_control_deactivate(self) -> None:
        """
        Desactiva el control del robot por medio de la mano
        """
        self.message_type = "INFO"
        self.message =  crear_mensaje(f"El control por mano ha sido desactivado", self.message_type, self.name)
        self.hand_control = False
        self._moveJoint(self.j_home)


    def _control_robot_by_hand(self) -> bool:
        '''
        Controla el robot por medio de la mano.
        '''
        rate = rospy.Rate(10)
        while self.hand_control: 
            rate.sleep()
            if self.hand_gesture['is_open']: # Si la mano está abierta, se mueve el robot.
                rango_cam = [640, 480, 480]
                rango_x_robot = self.hand_range['x_max'] - self.hand_range['x_min']
                rango_y_robot = self.hand_range['y_max'] - self.hand_range['y_min']
                rango_z_robot = self.hand_range['z_max'] - self.hand_range['z_min']

                current_pose = self.robot.get_pose()
                
                # Calcular el desplazamiento proporcional
                delta_x = (self.hand_pose[0] / rango_cam[0]) * rango_x_robot * 0.1
                delta_y = (self.hand_pose[1] / rango_cam[1]) * rango_y_robot * 0.1
                delta_z = (self.hand_pose[2] / rango_cam[2]) * rango_z_robot * 0.1

                # Crear nueva pose objetivo
                target_pose = deepcopy(current_pose)
                target_pose.position.x = current_pose.position.x + delta_x
                target_pose.position.y = current_pose.position.y + delta_y
                target_pose.position.z = current_pose.position.z + delta_z

                # Verificar límites
                if target_pose.position.x > self.hand_range['x_max'] or target_pose.position.x < self.hand_range['x_min']:
                    self.message_type = "WARN"
                    self.message = crear_mensaje(f"Posición X fuera de límites: {target_pose.position.x}", self.message_type, self.name)
                    continue
                if target_pose.position.y > self.hand_range['y_max'] or target_pose.position.y < self.hand_range['y_min']:
                    self.message_type = "WARN"
                    self.message = crear_mensaje(f"Posición Y fuera de límites: {target_pose.position.y}", self.message_type, self.name)
                    continue
                if target_pose.position.z > self.hand_range['z_max'] or target_pose.position.z < self.hand_range['z_min']:
                    self.message_type = "WARN"
                    self.message = crear_mensaje(f"Posición Z fuera de límites: {target_pose.position.z}", self.message_type, self.name)
                    continue
                
                if not self._movePose(target_pose, wait=False): # Mover el robot a la nueva posición
                    self.message_type = "ERROR"
                    self.message = crear_mensaje("No se pudo ejecutar el movimiento", self.message_type, self.name)
                        
            # Verificar el estado de la pinza
            elif self.hand_gesture['is_peace']: # Cerrar la pinza
                self.robot.move_gripper(0.0, 10.0, sleep_after=1.5)
                self.message_type = "INFO"
                self.message = crear_mensaje("Pinza cerrada - Intentando agarrar objeto", self.message_type, self.name)

            elif self.hand_gesture['is_dino']: # Soltar el objeto
                self.robot.move_gripper(50.0, 30.0, sleep_after=0.5)
                self.message_type = "INFO"
                self.message = crear_mensaje("Pinza abierta - Soltando objeto", self.message_type, self.name) 

            elif self.hand_gesture['is_dislike']: # Volver a la posición inicial
                self._moveJoint(self.j_hand_origin)
                self.message_type = "INFO"
                self.message = crear_mensaje("Moviendo robot a pose inicial", self.message_type, self.name) 


    def hand_data_callback(self, msg:HandData) -> None:
        '''
        Callback para recibir los datos de la mano.
            @param msg (HandData) - Mensaje con los datos de la mano.
            @return None
        '''
        self.hand_detected = msg.hand_detected
        self.hand_pose = [msg.x, msg.y, msg.z]
        self.hand_gesture['is_open'] = msg.is_open
        self.hand_gesture['is_peace'] = msg.is_peace
        self.hand_gesture['is_dino'] = msg.is_dino
        self.hand_gesture['is_dislike'] = msg.is_dislike
    

    def reset_robot(self) -> None:
        '''
        Resetea el robot a la posición inicial.
        '''
        self.robot.reset_planning_scene()

        # Mover el robot fuera del espacio de trabajo
        link_pose = self.robot.get_pose()
        link_pose.position.z = 0.3
        self._movePose(link_pose)

        # Mover el robot a la posición de inicio
        self._moveJoint(self.j_home)
        self.robot.move_gripper(35.0, 20.0)