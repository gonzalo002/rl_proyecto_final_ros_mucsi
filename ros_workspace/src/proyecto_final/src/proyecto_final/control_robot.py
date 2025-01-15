#!/usr/bin/python3

import sys, os
from copy import deepcopy
import rospy
from math import pi
from moveit_commander import MoveGroupCommander, RobotCommander, roscpp_initialize, PlanningSceneInterface
from control_msgs.msg import GripperCommandAction, GripperCommandGoal, GripperCommandResult
from std_msgs.msg import String, Float32, Bool
from moveit_commander.conversions import list_to_pose
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from sensor_msgs.msg import JointState
import yaml
from actionlib import SimpleActionClient
from proyecto_final.funciones_auxiliares import crear_mensaje

class ControlRobot:
    """
    Clase que controla el robot
        @method get_jointstates: Devuelve las articulaciones actuales del robot
        @method get_pose: Devuelve la pose actual del robot
        @method move_jointstates: Mueve el robot a unas articulaciones
        @method move_pose: Mueve el robot a una pose
        @method plan_pose: Planifica una pose
        @method move_jointstates_trayectory: Mueve el robot siguiendo una trayectoria de articulaciones
        @method move_pose_trayectory: Mueve el robot siguiendo una trayectoria de poses
        @method move_carthesian_trayectory: Mueve el robot siguiendo una trayectoria cartesiana
        @method add_box_obstacle: Añade un obstáculo al entorno
        @method reset_planning_scene: Reinicia el entorno
        @method save_in_yaml: Guarda en un archivo yaml la clave key_name con el valor data
        @method read_from_yaml: Lee un archivo yaml y devuelve el valor de la clave key_name
        @method jointstate_from_list: Convierte una lista en un objeto JointState
        @method pose_from_list: Convierte una lista en un objeto Pose
        @method _gripper_states_callback: Callback de las articulaciones del gripper
        @method _gripper_effort_callback: Callback del esfuerzo del gripper
        @method _rad_to_width: Convierte de radianes a grados
        @method move_gripper: Mueve el gripper
        @method get_pinza_state: Devuelve el estado del gripper
    """
    def __init__(self, group_name:str = 'robot', train_env:bool = False, test_env:bool = False) -> None:
        """
        Constructor de la clase ControlRobot
            @param group_name: Nombre del grupo del robot
            @param train_env: Booleano que indica si se está en un entorno de entrenamiento
            @param test_env: Booleano que indica si se está en un entorno de test
        """
        roscpp_initialize(sys.argv)
        try:
            rospy.init_node("control_robot", anonymous=True) # En caso de ya existir
        except:
            pass
        self.robot = RobotCommander() # Objeto que contiene la información del robot
        self.scene = PlanningSceneInterface() # Objeto que contiene la información del entorno
        self.group_name = group_name # Nombre del grupo del robot
        self.move_group = MoveGroupCommander(self.group_name) # Objeto que controla el grupo del robot
        self.gripper_action_client = SimpleActionClient("rg2_action_server", GripperCommandAction) # Cliente del gripper
        self.SubsGripStates = rospy.Subscriber("/rg2/joint_states", JointState, self._gripper_states_callback) # Suscriptor de las articulaciones del gripper
        self.SubsGripEffort = rospy.Subscriber("/rg2/grip_detected", Bool, self._gripper_effort_callback) # Suscriptor del esfuerzo del gripper
        self.get_gripper_state = False # Booleano que indica si se debe obtener el estado del gripper
        self.get_gripper_effort = False # Booleano que indica si se debe obtener el esfuerzo del gripper
        self.name = "ControlRobot" # Nombre de la clase

        self.move_group.set_max_velocity_scaling_factor(1) # Factor de escala de la velocidad
        self.move_group.set_max_acceleration_scaling_factor(1) # Factor de escala de la aceleración

        if train_env: # Entorno de entrenamiento
            self.move_group.set_planning_time(2) # Tiempo de planificación a 2s
        
        if not test_env: # Entorno de test
            self.reset_planning_scene() # Reinicia el entorno
            
        self.move_group.set_num_planning_attempts(5) # Número de intentos de planificación
    
    def get_jointstates(self) -> list:
        """
        Función que devuelve las articulaciones actuales del robot
            @return self.move_group.get_current_joint_values(): Articulaciones actuales del robot
        """
        return self.move_group.get_current_joint_values()
    
    def get_pose(self) -> Pose:
        """
        Función que devuelve la pose actual del robot
            @return self.move_group.get_current_pose().pose: Pose actual del robot
        """
        return self.move_group.get_current_pose().pose
    
    def move_jointstates(self, joint_goal: list, wait: bool = True) -> bool:
        """ 
        Función que mueve el robot a unas articulaciones
            @param joint_goal: Articulaciones a las que se quiere llegar
            @param wait: Booleano que indica si se debe esperar a que termine la trayectoria
            @return self.move_group.go(joint_goal, wait=wait): Booleano que
        """
        return self.move_group.go(joint_goal, wait=wait)
    
    def move_pose(self, pose_goal: Pose, wait: bool = True) -> bool:
        """
        Función que mueve el robot a una pose
            @param pose_goal: Pose a la que se quiere llegar
            @param wait: Booleano que indica si se debe esperar a que termine la trayectoria
            @return self.move_group.go(wait=wait): Booleano que indica si se ha ejecutado la trayectoria
        """

        self.move_group.set_pose_target(pose_goal)
        return self.move_group.go(wait=wait)
    
    def plan_pose(self, pose_goal:Pose) -> bool:
        """
        Función que planifica una pose
            @param pose_goal: Pose a la que se quiere llegar
            @return self.move_group.plan(): Booleano que indica si se ha planificado la pose
        """
        self.move_group.set_pose_target(pose_goal) # Establece la pose objetivo
        return self.move_group.plan()
    
    def move_jointstates_trayectory(self, trajectory:list = []) -> bool:
        """
        Función que mueve el robot siguiendo una trayectoria de articulaciones
            @param trajectory: Lista de articulaciones por las que pasará el robot
            @return state: Booleano que indica si se ha ejecutado la trayectoria
        """
        for i in range(len(trajectory)): # Por cada punto de la trayectoria
            state = self.move_jointstates(trajectory[i]) # Mueve el robot a ese punto
            crear_mensaje(f"Punto {i} alcanzado", "INFO", self.name)
            if not state: crear_mensaje(f"Trayectoria fallida", "ERROR", self.name); return False
        crear_mensaje(f"Trayectoria alzanzada", "SUCCESS", self.name)
        return state
    
    def move_pose_trayectory(self, trajectory:list = []) -> bool:
        """
        Función que mueve el robot siguiendo una trayectoria de poses
            @param trajectory: Lista de poses por las que pasará el robot
            @return state: Booleano que indica si se ha ejecutado la trayectoria
        """
        for i in range(len(trajectory)): # Por cada punto de la trayectoria
            state = self.move_pose(trajectory[i]) # Mueve el robot a ese punto
            crear_mensaje(f"Punto {i} alcanzado", "INFO", self.name)
            if not state: crear_mensaje(f"Trayectoria fallida", "ERROR", self.name); return False
        crear_mensaje(f"Trayectoria alzanzada", "SUCCESS", self.name)
        return state
    
    def move_carthesian_trayectory(self, waypoints:list = [], eef_step:Float32 = 0.01, avoid_collisions:bool = True ,wait:bool = True) -> bool:
        """
        Función que mueve el robot siguiendo una trayectoria cartesiana
            @param waypoints: Lista de puntos por los que pasará el robot
            @param eef_step: Paso del end effector
            @param avoid_collisions: Booleano que indica si se deben evitar colisiones
            @param wait: Booleano que indica si se debe esperar a que termine la trayectoria
            @return self.move_group.execute(plan, wait=wait): Booleano que indica si se ha ejecutado la trayectoria
        """
        if eef_step == 0.0: # Si el paso del end effector es 0
            eef_step = 0.01
            crear_mensaje("Parámetro eef_step modificado a valor 0.01 por requisitos de funcionamiento", "INFO", self.name)
            
        waypoints.insert(0, self.get_pose()) # Añade la pose actual al principio de la lista
        (plan, fraction) = self.move_group.compute_cartesian_path(waypoints, eef_step = eef_step, avoid_collisions= avoid_collisions)
        
        if fraction != 1.0: # Si no se ha alcanzado el 100% de la trayectoria
            crear_mensaje(f"Trayectoria Inalcanzable. Porcentaje de la trayectoria alcanzable: {fraction*100:.2f}%", "WARN", self.name)
            return False
        else: 
            crear_mensaje(f"Ejecutando Trayectoria", "INFO", self.name)
            return self.move_group.execute(plan, wait=wait)
        
    def add_box_obstacle(self, box_name:String, box_pose:Pose, size:tuple = (.1, .1, .1)) -> None:
        """
        Función que añade un obstáculo al entorno
            @param box_name: Nombre del obstáculo
            @param box_pose: Pose del obstáculo
            @param size: Tamaño del obstáculo
        """
        box_pose_stamped = PoseStamped()
        box_pose_stamped.header.frame_id = "base_link"
        box_pose_stamped.pose = box_pose
        self.scene.add_box(box_name, box_pose_stamped, size=size)
        
    def reset_planning_scene(self) -> None:
        """
        Función que reinicia el entorno
        """
        self.scene.clear()
        self._generate_scene()

    def _generate_scene(self) -> None:
        """
        Función que genera los obstáculos del entorno
        """
        pose_suelo = Pose()
        pose_suelo.position.z -= .03

        pose_vertical_support = Pose(Point(x=0,y=-0.1,z=0.5),
                                     Quaternion(x=0,y=0,z=0,w=1))
        pose_camera_support = Pose(Point(x=0.0,y=0.0,z=0.85),
                                     Quaternion(x=0,y=0,z=0,w=1))
        
        self.add_box_obstacle('floor', pose_suelo, (2,2,.05)) # Añade el suelo
        self.add_box_obstacle('vertical_support', pose_vertical_support, (0.05,0.05,1.0)) # Añade el soporte vertical
        self.add_box_obstacle('camera_support', pose_camera_support, (0.1,1.0,.05)) # Añade el soporte de la cámara
        
    
    def save_in_yaml(self, doc_name:str, key_name:str, data:list, delete_info:bool=False) -> None:
        """
        Función que guarda en un archivo yaml la clave key_name con el valor data
            @param doc_name: Nombre del archivo yaml
            @param key_name: Nombre de la clave a guardar
            @param data: Valor de la clave key_name
            @param delete_info: Booleano que indica si se debe sobreescribir el archivo
        """
        diccionario_configuraciones = {key_name:data}
        if delete_info: # Si se debe sobreescribir el archivo
            mode = '+w' # Modo de escritura
        else:
            mode = '+a' # Modo de añadir
        with open(doc_name, mode) as f:
            yaml.dump(diccionario_configuraciones, f)
    
    def read_from_yaml(self, doc_name:str, key_name:str) -> list:
        """
        Función que lee un archivo yaml y devuelve el valor de la clave key_name
            @param doc_name: Nombre del archivo yaml
            @param key_name: Nombre de la clave a leer
            @return configuraciones[key_name]: Valor de la clave key_name
        """
        with open(doc_name, '+r') as f:
            configuraciones =  yaml.load(f, yaml.Loader)

        return configuraciones[key_name]
    
    def jointstate_from_list(self, list:list) -> JointState:
        """
        Función que convierte una lista en un objeto JointState
            @param list: Lista con los valores de las articulaciones
            @return result: Objeto JointState
        """
        if len(list) != 6:
            return False
        
        result = JointState()
        result.position = list

        return result
    
    def pose_from_list(self, list:list) -> Pose:
        """
        Función que convierte una lista en un objeto Pose
            @param list: Lista con los valores de la posición y orientación
            @return result: Objeto Pose
        """
        if len(list) < 7:
            return False
        
        return list_to_pose(pose_list=list)
    
    def _gripper_states_callback(self, data:JointState) -> float:
        """
        Callback de las articulaciones del gripper
            @param data: Mensaje de las articulaciones del gripper
        """
        if self.get_gripper_state or self.get_gripper_effort:
            self.gripper_joint_state = deepcopy(self._rad_to_width(data.position[0]))
            self.get_gripper_state = False
    
    def _gripper_effort_callback(self, data:Bool) -> float:
        """
        Callback del esfuerzo del gripper
            @param data: Mensaje del esfuerzo del gripper
        """
        if self.get_gripper_state or self.get_gripper_effort:
            self.gripper_effort_state = deepcopy(data.data)
            self.get_gripper_effort = False
    
    def _rad_to_width(self, data:float = 0.0) -> None:
        """
        Función que convierte de radianes a grados
            @param data: Valor en radianes
            @return data * pi / 180: Valor en grados
        """
        return data * pi / 180
    
    def move_gripper(self, gripper_width: float, max_effort: float, sleep_before:float = 0.4, sleep_after:float = 0.2) -> bool:
        """
        Función que mueve el gripper
            @param gripper_width: Anchura del gripper
            @param max_effort: Esfuerzo máximo del gripper
            @param sleep_before: Tiempo de espera antes de enviar el objetivo
            @param sleep_after: Tiempo de espera después de enviar el objetivo
            @return result.reached_goal: Booleano que indica si se ha alcanzado el objetivo
        """
        goal = GripperCommandGoal() # Objetivo del gripper
        goal.command.position = gripper_width # Posición del gripper
        goal.command.max_effort = max_effort # Esfuerzo máximo del gripper
        
        rospy.sleep(sleep_before)
        self.gripper_action_client.send_goal(goal)
        self.gripper_action_client.wait_for_result()
        result = self.gripper_action_client.get_result()

        rospy.sleep(sleep_after)
        
        return result.reached_goal
    
    def get_pinza_state(self, sleep_before:float=0) -> list:
        """
        Función que devuelve el estado del gripper
            @param sleep_before: Tiempo de espera antes de obtener el estado
            @return self.gripper_joint_state, self.gripper_effort_state: Estado del gripper
        """
        rospy.sleep(sleep_before)

        self.get_gripper_state = True
        self.get_gripper_effort = True

        rate = rospy.Rate(20) # Frecuencia de actualización
        while self.get_gripper_state or self.get_gripper_effort:
            rate.sleep()

        return self.gripper_joint_state, self.gripper_effort_state