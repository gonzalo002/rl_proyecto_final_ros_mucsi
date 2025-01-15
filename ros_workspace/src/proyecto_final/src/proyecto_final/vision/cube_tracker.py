#!/usr/bin/env python3

import cv2, yaml, os
import numpy as np
from copy import deepcopy
from math import pi
from proyecto_final.funciones_auxiliares import crear_mensaje

class CubeTracker:
    ''' 
    Clase que procesa imágenes para detectar cubos y asignarles un color en función de su forma y tonalidad en el espacio de color HSV.
    
    Esta clase implementa varios métodos para:
        - Convertir imágenes a escala de grises o al espacio de color HSV.
        - Aplicar umbrales de Otsu para binarización.
        - Detectar bordes utilizando el filtro de Canny.
        - Encontrar contornos en las imágenes procesadas.
        - Identificar cubos en las imágenes y asignarles un color basado en su tonalidad.
        - Mostrar la imagen procesada y cerrar las ventanas de visualización.
    
    Métodos:
        - __init__() - Inicializa el objeto y crea un espacio para almacenar la imagen.
        - _aplicar_filtro_grises() - Convierte la imagen a escala de grises.
        - _aplicar_filtro_hsv() - Convierte la imagen a espacio de color HSV.
        - _procesar_umbral_otsu() - Aplica un umbral de Otsu para binarizar la imagen.
        - _detectar_bordes() - Detecta los bordes en una imagen utilizando Canny.
        - _encontrar_contornos() - Encuentra los contornos de una imagen binarizada.
        - _identificar_cubo_y_color() - Identifica cubos y les asigna un color basado en el espacio HSV.
        - analizar_imagen() - Procesa la imagen, detecta cubos y muestra los resultados si es necesario.
    
    Atributos:
        - frame (numpy array) - Almacena la imagen de entrada que se va a procesar.
    '''
    def __init__(self, cam_calib_path:str, aruco_dict=cv2.aruco.DICT_4X4_50, marker_length:float=0.05) -> None:
        '''
        Inicializa la clase y configura los parámetros de Aruco.
            @param aruco_dict (int) - Diccionario de Aruco (por defecto, 4x4 con 50 marcadores).\n
            @param marker_length (float) - Longitud del lado del marcador en metros.\n
            @param camera_matrix (numpy array) - Matriz intrínseca de la cámara.\n
            @param dist_coeffs (numpy array) - Coeficientes de distorsión de la cámara.
        '''
        # --- DEFINICIÓN DE VARIABLES ---
        self.frame = None
        self.aruco_length = marker_length
        self.aruco_corner_pos = None
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.debug = False
        self.i = 0
        self.name = "CubeTracker"

        # VARIABLE PARA TKINTER
        self.message = None
        self.message_type = None # 1=Info, 2=Warn, 3=Error 
         
        try:
            file_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:os.path.dirname(os.path.abspath(__file__)).split('/').index('proyecto_final')+1])
            with open(f'{file_path}/data/necessary_data/cube_area.yaml', 'r') as file:
                self.base_area = yaml.safe_load(file)
        except:
            self.message = "El cubo no ha sido calibrado, hazlo antes de empezar."
            self.message_type = 3
            crear_mensaje(self.message, "ERROR", self.name)
        
        # Obtener calibración de la cámara
        self._get_camara_calibration(cam_calib_path)


    def _get_camara_calibration(self, path:str) -> None:
        '''
        Lee y carga los parámetros de calibración de cámara desde un archivo YAML.
            @param path (str) - Ruta al archivo de calibración de cámara.
        '''
        try:
            with open(path, '+r') as f:
                fichero =  yaml.load(f, yaml.Loader)
        except:
            print("[ERROR] No se ha encontrado el path.")
            self.message = "No se ha encontrado el directorio del fichero de calibración."
            self.message_type = 3

        # Extraer matriz intrínseca y coeficientes de distorsión de la cámara
        self.camera_matrix = np.array(fichero["camera_matrix"]["data"]).reshape(fichero["camera_matrix"]["rows"], fichero["camera_matrix"]["cols"])
        self.dist_coeffs = np.array(fichero["distortion_coefficients"]["data"])
        

    def _aplicar_filtro_grises(self, frame:np.ndarray) -> np.ndarray:
        ''' 
        Convierte una imagen de color a escala de grises.
            @param frame (numpy array) - Imagen en formato BGR.\n
            @return gray_frame (numpy array) - Imagen en escala de grises.
        '''
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    def _aplicar_filtro_hsv(self, frame:np.ndarray) -> np.ndarray:
        ''' 
        Convierte una imagen de color a espacio de color HSV.
            @param frame (numpy array) - Imagen en formato BGR.\n
            @return hsv_frame (numpy array) - Imagen en espacio de color HSV.
        '''
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    def _procesar_umbral_otsu(self, gray:np.ndarray) -> np.ndarray:
        ''' 
        Aplica un umbral de Otsu para binarizar la imagen.
            @param gray (numpy array) - Imagen en escala de grises.\n
            @return otsu_thresh (numpy array) - Imagen binarizada tras el umbral de Otsu.
        '''
        _, otsu_thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
        if self.debug:
            cv2.imshow("Otsu", otsu_thresh)
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    def _detectar_bordes(self, imagen:np.ndarray) -> np.ndarray:
        ''' 
        Detecta los bordes de una imagen utilizando el filtro de Canny.
            @param imagen (numpy array) - Imagen de entrada (en escala de grises o color).\n
            @return edges (numpy array) - Imagen con bordes detectados.
        '''
        desenfocada = cv2.GaussianBlur(imagen, (5, 5), 0)
        canny = cv2.Canny(desenfocada, 0, 255)
        
        if self.debug:
            cv2.imshow("Canny", canny)
            
        return canny
            
    
    def _encontrar_contornos(self, edges:np.ndarray) -> list:
        ''' 
        Encuentra los contornos en una imagen binarizada.
            @param edges (numpy array) - Imagen binarizada con bordes detectados.\n
            @return contours (list) - Lista de contornos encontrados.
        '''
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        filtered_contours = []
        for i, (next_bro, _, _, _) in enumerate(hierarchy[0]):
            if next_bro != 0:
                filtered_contours.append(contours[i])

        if self.debug:
            img = self.frame.copy()
            cv2.drawContours(img, filtered_contours, -1, (0,255,0))
            cv2.imshow('Todos los Contornos', img)

        return filtered_contours


    def _identificar_cubo_y_color(self, contours: list, frame: np.ndarray, hsv_frame: np.ndarray, gray: np.ndarray, mostrar:bool = False) -> list:
        """
            Identifica cubos en una imagen basada en sus contornos, evalúa si los contornos tienen forma cuadrada y les asigna un color predominante según el espacio de color HSV. 
            También calcula información relevante como posición y ángulo de los cubos detectados.
                @param contours (list) - Lista de contornos detectados en la imagen.\n

                @param frame (np.ndarray) - Imagen original en formato BGR (color).\n
                @param hsv_frame (np.ndarray) - Imagen en el espacio de color HSV.\n
                @param gray (np.ndarray) - Imagen en escala de grises, utilizada para generar máscaras y otras operaciones de procesamiento.\n
                @param mostrar (bool) - Establece si se dibujará información adicional sobre los cubos detectados en la imagen de salida (posición y ángulo). Por defecto es `False`.\n
                @return dict (list) - Devuelve una lista de diccionarios, donde cada diccionario contiene información sobre un cubo detectado:\n
                    -"Position": Distancia en coordenadas XY desde un punto de referencia (probablemente el origen).
                    -"Angle": Ángulo de orientación del cubo en radianes.
                    -"Color": Índice del color predominante detectado en el cubo.
        """

        # Definir el mapa de colores y sus índices numéricos
        colores = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 255, 255)}  # RGB
        
        resultado = frame.copy()
        dicionario_resultado = []
        
        # Tolerancia para verificar que el rectángulo sea un cuadrado
        tolerancia = 0.2
        area_size = [cv2.contourArea(cnt) for cnt in contours]
        
        i = 0
        # Recorrer los contornos y realizar las operaciones necesarias
        for u, cnt in enumerate(contours):
            # Calcular el área y el rectángulo mínimo
            x, y, w, h = cv2.boundingRect(cnt)
            
            if  area_size[u] < self.base_area * 0.3:
                continue
            # Calcular el rectángulo mínimo y verificar si es un cuadrado
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect).astype(np.int32)
            w, h = rect[1]  # Rectángulo (w, h)
            
            # Verificar si el contorno es un cuadrado
            if h == 0:
                continue
            if abs(w / h - 1) < tolerancia:
                center_rect = rect[0]
                center = tuple(map(int, rect[0]))  # Convertir a enteros
                angle = rect[2]
                
                cv2.putText(resultado, f"{i}", (center[0]-5, center[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.circle(resultado, self.corner_center, 5, (0, 0, 0), -1)
                
                
                # Determinar el color predominante dentro del contorno
                color = self._get_dominant_color(cnt, hsv_frame)
                
                # Dibujar el contorno con el color identificado
                cv2.drawContours(resultado, [box], 0, colores[color], 2)
                
                # Añadir la información al diccionario de resultados
                dicionario_resultado.append({
                    "Position": self._distancia_xy(self.corner_center, center_rect),
                    "Angle": angle*pi/180,
                    "Color": color  # Usar el índice de color directamente
                })
                
                if mostrar:
                    cv2.putText(resultado, f"Position: {dicionario_resultado[i]['Position']}", (center[0] + 20, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                    cv2.putText(resultado, f"Angle: {dicionario_resultado[i]['Angle']:.2f}", (center[0] + 20, center[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                i+=1
                

        self.imagen_analizada = resultado
        return dicionario_resultado


    def _get_dominant_color(self, contour: np.ndarray, hsv_frame: np.ndarray) -> int:
        ''' 
        Determina el color (rojo, verde, azul y amarillo) predominante dentro de un contorno basado en el espacio de color HSV.
            @param contour (numpy array) - Contorno del que se quiere extraer el color predominante.\n
            @param hsv_frame (numpy array) - Imagen en espacio HSV.\n
            @return dominant_color (int) - Índice numérico que representa el color predominante (0=Rojo, 1=Verde, 2=Azul, 3=Amarillo).\n
        '''
        # Definir los rangos de colores en el espacio HSV
        color_ranges = {
            0: [(0, 100, 100), (20, 255, 255)], #Rojo
            1: [(40, 50, 50), (80, 255, 255)],  #Verde
            2: [(100, 100, 100), (130, 255, 255)], #Azul 
            3: [(30, 30, 30), (40, 255, 255)] #Amarillo
        }
        
        # Crear una máscara para la región definida por el contorno
        mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        
        # Crear un diccionario para contar los píxeles de cada color
        color_counts = {color: 0 for color in color_ranges}
        
        # Recorrer cada rango de color y contar los píxeles que están dentro del rango
        for color, (lower, upper) in color_ranges.items():
            lower_bound = np.array(lower)
            upper_bound = np.array(upper)
            
            # Crear la máscara para el color
            color_mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
            
            # Aplicar la máscara al área del contorno
            color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask)
            
            # Contar la cantidad de píxeles del color
            color_counts[color] = cv2.countNonZero(color_mask)
        
        # Determinar el color predominante
        dominant_color = max(color_counts, key=color_counts.get)
        
        
        return dominant_color

    
    def _detectar_aruco(self, frame: np.ndarray) -> None:
        '''
        Detecta un marcador Aruco en la imagen y calcula su posición y orientación.
            @param frame (numpy array) - Imagen en formato escala de grises.\n
        '''
        corners, ids, _ = cv2.aruco.detectMarkers(frame, self.aruco_dict, parameters=self.aruco_params)
        
        aruco_frame = frame.copy()
        
        if ids is not None:
            corners_deep = deepcopy(corners[0])
            # Dibujar el contorno del Aruco
            frame = cv2.aruco.drawDetectedMarkers(aruco_frame, corners, ids, (0, 0, 0))          
            
            # --- QUITAR ARUCO DE LA IMAGEN ---            
            mask = np.ones(aruco_frame.shape, dtype=np.uint8) * 255  # Mascara blanca
            expanded_corners = self._expand_corners(corners_deep, 1.5) #Expandir para quitar el borde del Aruco
            # Convertir las coordenadas de las esquinas a un formato adecuado
            cv2.fillPoly(mask, expanded_corners, (0, 0, 0))

            # Mostrar la máscara generada
            if self.debug:
                cv2.imshow("Mask", mask)

            # Crear la imagen final aplicando la máscara a la imagen original
            frame_with_mask = cv2.bitwise_and(frame, mask)
            
            # Calcular el lado 
            self.side_lengths_px = self._calcular_lados(corners_deep)
            self.corner_center = np.array(corners_deep[0][0], dtype=int)

            
            return frame_with_mask, aruco_frame
        else:
            return False, False
    
    def _calcular_lados(self, corners:np.ndarray) -> float:
        '''
        Calcula la longitud de los lados de un marcador Aruco en píxeles.
            @param corners (numpy array) - Coordenadas de las esquinas del marcador Aruco.\n
            @return side_length (float) - Longitud promedio de los lados del marcador en píxeles.
        '''
        # Esquinas del marcador
        #p1, p2, p3, p4 = corners
        p1 = corners[0][0]
        p2 = corners[0][1]
        p3 = corners[0][2]
        p4 = corners[0][3]
        
        # Calcular la distancia entre las esquinas (lado de la figura)
        distancias = [self._distancia(p1, p2), self._distancia(p2, p3),
                      self._distancia(p3, p4), self._distancia(p4, p1)]
        
        
        return np.mean(distancias)
    
    def _distancia(self, p1:np.ndarray, p2:np.ndarray) -> float:
        '''
        Calcula la distancia euclidiana entre dos puntos (x1, y1) y (x2, y2).
            @param p1 (numpy array): Primer punto (x1, y1).\n
            @param p2 (numpy array): Segundo punto (x2, y2).\n
            @return (float): Distancia entre los dos puntos.
        '''
        return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def _distancia_xy(self, p_aruco:np.ndarray, p_cubo:tuple) -> tuple:
        '''
        Calcula la distancia euclidiana entre dos puntos (x1, y1) y (x2, y2).
            @param p_aruco (numpy array): Primer punto (x1, y1).\n
            @param p_cubo (tuple): Segundo punto (x2, y2).\n
            @return (tuple): Distancia entre los dos puntos.
        '''
        x = float((-1)*((p_aruco[0] - p_cubo[0])*self.aruco_length/self.side_lengths_px))
        y = float((p_aruco[1] - p_cubo[1])*self.aruco_length/self.side_lengths_px)
        
        return (round(x,4), round(y,4))
    
    def _expand_corners(self, corners:np.ndarray, factor:float=1.2) -> list:
        """
        Expande las esquinas del marcador ArUco aumentando su tamaño.
            @param corners (numpy array): Esquinas de los marcadores detectados (forma de (1, 4, 2)).\n
            @param factor (float): Factor de escala para aumentar o reducir el tamaño del marcador.\n
            @return (list): Nuevas coordenadas de las esquinas ajustadas.
        """
        expanded_corners = []

        for corner in corners:
            # Calcular el centro (promedio de los puntos)
            centro = np.mean(corner, axis=0)

            new_corners = []
            for point in corner:
                new_point = centro + (point - centro) * factor
                new_corners.append(new_point)

            # Convertir a tipo entero (por si acaso) y agregar el marcador ajustado a la lista final
            expanded_corners.append(np.array(new_corners, dtype=np.int32))

        return expanded_corners

    def process_image(self, frame:np.ndarray, mostrar:bool = False, debug:bool = False) -> list:
        ''' 
        Procesa una imagen para detectar cubos y sus colores, y muestra los resultados.
            @param frame (numpy array) - Imagen de entrada en formato BGR.\n
            @param mostrar (bool) - Si es True, muestra la imagen procesada. Por defecto, False.\n
            @return resultado (list) - Imagen procesada con cubos identificados y etiquetados.
        '''
        self.debug = debug
        if self.debug:
            cv2.imshow("Imagen Original", frame)
        # Realizar copia del frame
        self.frame = deepcopy(cv2.undistort(frame, self.camera_matrix, self.dist_coeffs))
        frame = deepcopy(cv2.undistort(frame, self.camera_matrix, self.dist_coeffs))
        

        # Detectar Aruco
        mask_frame, aruco_frame = self._detectar_aruco(frame)
        if mask_frame is False:
            self.message = "El ArUco no está en la mesa."
            self.message_type = 3
            print(f"[ERROR] El Aruco no está en la mesa")
            return frame, []
            
        gray = self._aplicar_filtro_grises(mask_frame)
        hsv_frame = self._aplicar_filtro_hsv(mask_frame)
        morph_clean = self._procesar_umbral_otsu(gray)
        edges = self._detectar_bordes(morph_clean)
        contours = self._encontrar_contornos(edges)
        
        resultado = self._identificar_cubo_y_color(contours, aruco_frame, hsv_frame, gray, mostrar)

        if mostrar:
            cv2.imshow('Contoured Image', self.imagen_analizada)

        if mostrar or debug:
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        
        if resultado == []:
            self.message = "No se ha encontrado ningún cubo"
            self.message_type = 2
        else:
            self.message = "Se ha procesado la imagen."
            self.message_type = 1

        return self.imagen_analizada, resultado

if __name__ == "__main__":
    use_cam = False
    file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', "/")
    cube_tracker = CubeTracker(cam_calib_path=f"{file_path}/data/necessary_data/ost.yaml")

    if use_cam:
        cam = cv2.VideoCapture(0)
        if cam.isOpened():
            _, frame = cam.read()
            cv2.imshow("Frame Capturado", frame)
    else:
        num = 0
        ruta = f'{file_path}/data/cubos_exparcidos/Cubos_Exparcidos_{num}.png'
        frame = cv2.imread(ruta)

    _, resultado = cube_tracker.process_image(frame, mostrar=True, debug=True)
    print(resultado)
    if use_cam:
        cam.release()