#!/usr/bin/python3

import rospy, actionlib, os
from proyecto_final.funciones_auxiliares import crear_mensaje
from threading import Thread
from stable_baselines3 import PPO
from proyecto_final.rl.env_rob_demo import ROSEnv
from proyecto_final.msg import RLAction, RLFeedback, RLGoal, RLResult
from trajectory_msgs.msg import JointTrajectory


class CubeOrderActionServer(object):
    """
    Clase que implementa el servidor de acción de CubeOrder.
        @method __init__ - Método constructor
        @method execute_cb - Callback del action server del CubeOrder
        @method send_feedback - Función que envía el feedback al cliente
    """
    def __init__(self):
        """
        Método constructor de la clase CubeOrderActionServer.
        """
        # Inicializar nodo
        rospy.init_node('rl_action_server_node')
        self.name = "RLActionServer"
        
        # Definicion de variables de Python
        self.file_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:os.path.dirname(os.path.abspath(__file__)).split('/').index('proyecto_final')+1])
        self.message = None
        self.message_type = None

        # Definicion de las variables del Entorno
        self.agent:PPO = None
        self.env:ROSEnv = None
        self.cube_order:list = []
        self.cube_trajectories:list = []

        # Definicion de variables ROS
        self.action_server = actionlib.SimpleActionServer(self.name, RLAction, execute_cb=self.execute_cb, auto_start=False)

        # Action Server
        self.action_server.start()

        rospy.spin()
    

    def execute_cb(self, goal:RLGoal)->None:
        ''' 
        Callback del action server del CubeTracker
            @param goal (numpy array) - Goal recibido por el cliente
        '''
        self.running = True
        # Crear instacia de feedback
        feedback = RLFeedback()

        # Crear y ejecutar el hilo de feedback
        feedback_thread = Thread(target=self.send_feedback, args=(feedback,))
        feedback_thread.start()

        # --- PASO 1: Cargar el Entorno ---
        cubos = goal.cubes_position
        figure_order = goal.cubes_order
        self.env = ROSEnv(cubos=cubos, orden_figura=figure_order)

        # --- PASO 2: Cargar el Agente ---
        self.model = PPO.load(f'{self.file_path}/data/necessary_data/rl_agent.zip', self.env)       
        
        # --- PASO 3: Obtener el Orden y Trayectorias ---
        done = False
        failed = False
        obs, info = self.env.reset()

        while not done:
            action = self.model.predict(observation=obs)
            _, _, done, failed, info = self.env.step(action=action)

            if failed: # Cuando falle una trayectoria de forma continuada
                self.message_type = "ERROR"
                self.message =  crear_mensaje("Trayectoria Fallida - \
                                              Imposible Calcular el Orden de los Cubos", self.message_type, self.name)
                info['orden_cubos'] = []
                info['trayectorias'] = JointTrajectory()
                break
        
        resultado_final = RLResult()
        resultado_final.cubes_correct_order = info['orden_cubos']
        resultado_final.cubes_trajectories = info['trayectorias']
    
        # --- PASO 4: Enviar el resultado ---
        self.action_server.set_succeeded(resultado_final)

        # --- PASO 5: Esperar union del hilo ---
        self.running = False
        feedback_thread.join()
    
    
    def send_feedback(self, feedback:RLFeedback):
        ''' 
        Función que utiliza el hilo para enviar el feedback al cliente
            @param feedback (CubosActionFeedback) - Feedback
        '''
        while self.action_server.is_active() and self.running:
            feedback.feedback = 1
            self.action_server.publish_feedback(feedback)
            rospy.sleep(0.1)

if __name__ == "__main__":
    CubeOrderActionServer()