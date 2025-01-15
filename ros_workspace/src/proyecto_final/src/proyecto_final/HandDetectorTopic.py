#!/usr/bin/env python3

import cv2, os, rospy
from copy import deepcopy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from proyecto_final.msg import HandData
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "1"
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "2"

from proyecto_final.vision.hand_detector import HandDetector

    
class HandTopic():
    """
    Clase que se encarga de la detección de manos en las imágenes de las cámaras.
        @method __init__ - Constructor de la clase.
        @method cb_image_top - Callback del subscriptor de la cámara.
        @method cb_image_lateral - Callback del subscriptor de la cámara.
        @method hand_detector - Método que se encarga de procesar las imágenes de las cámaras.
    """
    def __init__(self):
        '''
        Constructor de la clase.
        '''
        # Inicialización del nodo de ROS
        rospy.init_node('hand_detector_node')

        self.cv_img_top = self.cv_img_lateral = [] # Imagenes de las cámaras
        
        self.bridge = CvBridge() # Conversor de imagen de ROS a OpenCV
        self.subs_cam_top = rospy.Subscriber('/top_hand_cam/image_raw', Image, self.cb_image_top) # Subscriptor de la cámara
        self.subs_cam_lateral = rospy.Subscriber('/lateral_hand_cam/image_raw', Image, self.cb_image_lateral) # Subscriptor de la cámara
        self.pub_hand_data = rospy.Publisher('/hand_data', HandData, queue_size=10) # Publicador de la información de la

        self.hand_processor = HandDetector() # Procesador de la imagen de la cámara

        self.hand_detector() # Detección de las manos



    def cb_image_top(self, image:Image)->None:
        ''' 
        Callback del subscriptor de la cámara.
            @param image (Image) - Imagen de la camara
        '''

        self.cv_img_top = deepcopy(self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough'))
        self.cv_img_top = deepcopy(cv2.cvtColor(self.cv_img_top, cv2.COLOR_RGB2BGR))

    def cb_image_lateral(self, image:Image)->None:
        ''' 
        Callback del subscriptor de la cámara.
            @param image (Image) - Imagen de la camara
        '''

        self.cv_img_lateral = deepcopy(self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough'))
        self.cv_img_lateral = deepcopy(cv2.cvtColor(self.cv_img_lateral, cv2.COLOR_RGB2BGR))


    def hand_detector(self) -> None:
        """
        Método que se encarga de procesar las imágenes de las cámaras.
        """
        self.hand_processor.define_hands() # Definir detectores de manos
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown(): # Mientras el nodo no se cierre
            if len(self.cv_img_lateral)!=0 and len(self.cv_img_top)!=0: # Si se reciben imágenes de las cámaras
                img_top = deepcopy(self.cv_img_top)
                img_lateral = deepcopy(self.cv_img_lateral)

                # Procesamiento de las imágenes
                self.hand_processor.detect_hand(img_top=img_top, img_lateral=img_lateral)

                # Publicación de la información de las manos
                self.pub_hand_data.publish(self.hand_processor.hand_data)
                self.cv_img_lateral = self.cv_img_top = deepcopy([])
            
            rate.sleep()

if __name__ == '__main__':
    HandTopic()