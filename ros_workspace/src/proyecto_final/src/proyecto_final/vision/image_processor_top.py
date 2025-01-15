#!/usr/bin/env python3

import cv2, os, yaml
import numpy as np
from copy import deepcopy
from math import sqrt

# --- CÓDIGO COLORES ---
c = {
    "ERROR": "\033[31m",      # Rojo
    "SUCCESS": "\033[32m",     # Verde
    "WARN": "\033[33m",  # Amarillo
    "RESET": "\033[0m"       # Restablecer al color predeterminado
}

class ImageProcessor_Top:
    ''' 
    Clase que procesa imágenes para detectar y clasificar cubos en función de su color, forma y tamaño, 
    usando técnicas de procesamiento de imágenes como la detección de bordes, la segmentación de contornos 
    y el análisis del color dominante.

    Esta clase realiza los siguientes pasos:
        - Preprocesa la imagen convirtiéndola a escala de grises y aplicando filtros de detección de bordes.
        - Encuentra contornos externos y filtra los contornos según su tamaño.
        - Extrae el color dominante (rojo, verde, azul o amarillo) dentro de los contornos usando el espacio de color HSV.
        - Alinea los puntos detectados en una cuadrícula equidistante de 5x5.
        - Mapea los centros de los cubos a una matriz de 5x5 para representarlos en un formato estructurado.
        - Dibuja los contornos de los cubos sobre la imagen original, con su color asignado.

    Métodos:
        - __init__() - Inicializa el objeto y configura la matriz 5x5 y las variables necesarias.
        - _preprocess_image() - Preprocesa la imagen y encuentra los contornos de los cubos en la imagen.
        - _draw_cubes() - Dibuja los cubos encontrados y guarda su información (centro, área, color).
        - _umbralizacion() - Aplica un umbral de Otsu a la imagen en escala de grises.
        - _get_contrast_img() - Separa la imagen en diferentes colores (rojo, verde, azul, amarillo) y mejora el contraste.
    '''
        
    def __init__(self, matrix_size:int=5, aruco_dict=cv2.aruco.DICT_4X4_50) -> None:
        ''' 
        Inicializa los atributos de la clase.
        - matrix_size: Tamaño de la matriz cuadrada que representa la cuadrícula de colores (por defecto, 5x5).
        - matrix: Matriz inicializada con valores -1, que se usará para almacenar los colores detectados.
        - contour_img: Imagen utilizada para dibujar contornos detectados.
        - frame: Imagen original que se procesará.  
            @param matrix_size (int) - Tamaño de la matriz cuadrada. Por defecto, 5.
        '''
        self.matrix_size = matrix_size
        self.matrix = np.full((self.matrix_size, self.matrix_size), -1)
        self.contour_img = None
        self.frame = None
        self.debug = False
        self.name = "ImageProcessorTop"
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        self.centers = []
        self.areas = []
        self.colors = []
        
        self.diccionario_colores = {0: (0,0,255), 1: (0,255,0), 2: (255,0,0), 3: (0, 255, 255)}
        self.message = None
        self.message_type = 0 # 1=Info, 2=Warn, 3=Error  
        
        try:
            file_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:os.path.dirname(os.path.abspath(__file__)).split('/').index('proyecto_final')+1])
            with open(f'{file_path}/data/necessary_data/cube_area.yaml', 'r') as file:
                self.base_area = yaml.safe_load(file)
        except:
            print(f"{c['ERROR']}[ERROR] [{self.name}] El cubo no ha sido calibrado, hazlo antes de empezar.{c['RESET']}")
            self.message = "El cubo no ha sido calibrado, hazlo antes de empezar."
            self.message_type = 3
        
    def _preprocess_image(self) -> np.ndarray:
        """ 
        Convierte la imagen a escala de grises y aplica filtros para mejorar la detección de bordes.
            - Aplica el filtro Sobel para detectar bordes en las direcciones X e Y.
            - Combina las detecciones en una imagen de magnitud.
            - Utiliza el filtro Canny para refinar los bordes detectados.
            - Aplica operaciones morfológicas para limpiar ruido.
                @return morph_clean (numpy array) - Imagen binaria con los bordes detectados y limpiados.
        """

        cropped_frame = self._get_cubes_location(colored=True)

        if cropped_frame is not None:
            separated_colors = self._get_contrast_img(cropped_frame)

            self.centers = []
            self.areas = []
            self.colors = []
            if self.debug:
                show_contorus = []
                color_dict = {0: (0,0,255), 1:(0,255,0), 2:(255,0,0), 3:(0,255,255)}
            for i, color in enumerate(separated_colors): # [red, green, blue, yellow]
                contours, _ = cv2.findContours(color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if self.debug:
                        show_contorus.append(contours)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    
                    if area > self.base_area * 0.2:
                        self._draw_cubes(cnt, color, i)
        
            if self.debug:
                img = deepcopy(self.frame)
                cont = 0
                for temp_contours in show_contorus:
                    cv2.drawContours(img, contourIdx=-1, contours=temp_contours, color=color_dict[cont], thickness=2)
                    cont += 1
                cv2.imshow('Todos los Contornos', img)

                
    def _draw_cubes(self, cnt, img, color_cubo):
        """ 
        Dibuja los cubos sobre la imagen de contorno y guarda su centro, área y color.
        Calcula el tamaño de los cubos basándose en el contorno detectado y ajusta según el área base.
        
            @param cnt (numpy array) - Contorno de un cubo detectado en la imagen.
            @param img (numpy array) - Imagen de entrada donde se dibujarán los contornos de los cubos.
            @param color_cubo (int) - Identificador del color del cubo (0 = rojo, 1 = verde, 2 = azul, 3 = amarillo).
            """
        x, y, w, h = cv2.boundingRect(cnt)

        # Número de cubos por lado
        lado = sqrt(self.base_area)

        # Tamaño de cada cubo
        if (w//lado - w/lado) > 0.6:
            num_cubos_x = int(w/(lado-4)) + 1 
        else:
            num_cubos_x = int(w/(lado-4))
            if num_cubos_x == 0:
                num_cubos_x = 1
        
        if (h//lado - h/lado) > 0.6:
            num_cubos_y = int(h/(lado-4)) + 1 
        else:
            num_cubos_y = int(h/(lado-4))
            if num_cubos_y == 0:
                num_cubos_y = 1
        
        width_cubo = w // num_cubos_x
        height_cubo = h // num_cubos_y
        
        for i in range(num_cubos_x):
            for j in range(num_cubos_y):
                # Calcular las coordenadas de cada cubo
                x1 = x + i * width_cubo
                y1 = y + j * height_cubo
                x2 = x1 + width_cubo
                y2 = y1 + height_cubo

                centro = [round((x1+x2)/2), round((y1+y2)/2)]
                
                if img[centro[1],centro[0]] > 20:
                    # Crear el contorno del cubo
                    cubo_contorno = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
                    
                    #Almacenar los cubos
                    self.areas.append(width_cubo*height_cubo)
                    self.colors.append(color_cubo)
                    self.centers.append(centro)
                    
                    #Pintar el resultado
                    cv2.circle(self.contour_img, centro, 5, (0, 0, 0), -1)
                    cv2.polylines(self.contour_img, [cubo_contorno], isClosed=True, color=self.diccionario_colores[color_cubo], thickness=2)
                    
            
        
    def _umbralización(self, gray:np.ndarray) -> np.ndarray:
        ''' 
        Aplica un umbral para binarizar la imagen.
            - Convierte la imagen en escala de grises a una imagen binaria.
            - Utiliza un umbral para decidir el valor de corte.
            - Aplica una operación morfológica para limpiar pequeños ruidos en la imagen binaria.
                @param gray (numpy array) - Imagen en escala de grises.
                @return thresh (numpy array) - Imagen binarizada tras la umbralizacion.
        '''
        _, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

        if self.debug:
            cv2.imshow("Umbralización", thresh)

        return thresh
    

    def _get_contrast_img(self, frame:np.ndarray) -> list:
        ''' 
        Separa la imagen en distintos colores, aplicando umbrales específicos para rojo, verde, azul y amarillo.
        Genera imágenes contrastadas de cada color.
            - Convierte la imagen original a espacio de color HSV.
            - Aplica umbrales en los rangos de valores HSV para cada color (rojo, verde, azul, amarillo).
            - Realiza operaciones morfológicas para limpiar las imágenes de cada color.
            - Invertir la imagen de cada color para mejorar la detección.
                @param frame (numpy array) - Imagen a procesar.
                @return contrast_images (list) - Lista de imágenes filtradas por cada color (rojo, verde, azul, amarillo).
        '''

        # Convertir la imagen a HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   
        height, width = hsv.shape[:2]
        black_mask = np.zeros((height, width), dtype=np.uint8)
        # Definir los rangos de colores en HSV

        # Verde: 
        lower_green = np.array([30, 50, 50], np.uint8)   # Mínimo verde
        upper_green = np.array([70, 255, 255], np.uint8) # Máximo verde

        # Rojo: 
        lower_red = np.array([0, 100, 100], np.uint8)    # Minimo rojo
        upper_red = np.array([20, 255, 255], np.uint8)  # Maximo rojo

        # Azul: 
        lower_blue = np.array([100, 50, 50], np.uint8)   # Mínimo azul
        upper_blue = np.array([140, 255, 255], np.uint8) # Máximo azul

        # Amarillo: 
        lower_yellow = np.array([30, 30, 30], np.uint8)  # Mínimo amarillo
        upper_yellow = np.array([40, 255, 255], np.uint8)# Máximo amarillo

        lower_values = [lower_red, lower_green, lower_blue, lower_yellow]
        upper_values = [upper_red, upper_green, upper_blue, upper_yellow]
        filtered_images:list = []
        
        for i in range(len(lower_values)):
            # Aplicar un umbral de valor mínimo para eliminar el fondo negro (valor bajo)
            mask = cv2.inRange(hsv, lower_values[i], upper_values[i]) 
            value_threshold = 100  # Umbral mínimo de valor (a partir de este valor consideramos los colores)
            value_mask = hsv[:, :, 2] > value_threshold  # Solo seleccionamos los píxeles con valor mayor que el umbral
            value_mask = np.uint8(value_mask) * 255
            mask = cv2.bitwise_and(mask, mask, mask=value_mask)

            filtered_image = (cv2.bitwise_and(frame, frame, mask=mask))
            filtered_images.append(filtered_image)

        contrast_images = []
        kernel = np.ones((5, 5), np.uint8)
        for i in range(len(filtered_images)):
            alpha = 1.3
            beta = 1

            contrast_img = cv2.convertScaleAbs(filtered_images[i], alpha=alpha, beta=beta) 
            contrast_img = cv2.morphologyEx(contrast_img, cv2.MORPH_CLOSE, kernel)
            contrast_images.append(contrast_img)
        
        gray_r = cv2.cvtColor(contrast_images[0], cv2.COLOR_BGR2GRAY)
        _, r_inv = cv2.threshold(gray_r, 50, 255, cv2.THRESH_BINARY)

        # Invertir la imagen azul
        gray_b = cv2.cvtColor(contrast_images[2], cv2.COLOR_BGR2GRAY)
        _, b_inv = cv2.threshold(gray_b, 50, 255, cv2.THRESH_BINARY)

        # Invertir la imagen amarilla
        gray_y = cv2.cvtColor(contrast_images[3], cv2.COLOR_BGR2GRAY)
        _, y_inv = cv2.threshold(gray_y, 50, 255, cv2.THRESH_BINARY)

        # Aplicar las máscaras a la imagen negra (eliminando las zonas blancas de cada canal)
        mask_green = cv2.bitwise_or(black_mask, r_inv)
        mask_green = cv2.bitwise_or(mask_green, b_inv)
        mask_green = cv2.bitwise_or(mask_green, y_inv)
        
        mask_green = cv2.bitwise_not(mask_green)

        color_g =cv2.bitwise_and(contrast_images[1], contrast_images[1], mask=mask_green)
        gray_g = cv2.cvtColor(color_g, cv2.COLOR_BGR2GRAY)
        _, g_inv = cv2.threshold(gray_g, 40, 255, cv2.THRESH_BINARY)                     
             
        resultado= [r_inv, g_inv, b_inv, y_inv]
        
        kernel = np.ones((5, 5), np.uint8)
        for i, img in enumerate(resultado):
            resultado[i] = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
            
        if self.debug:
             cv2.imshow('Red Only', contrast_images[0])
             cv2.imshow('Green Only', color_g)
             cv2.imshow('Blue Only', contrast_images[2])
             cv2.imshow('Yellow Only', contrast_images[3])
             
        return resultado


    def _get_cubes_location(self, colored:bool = False) -> np.ndarray:
        ''' 
        Aplica un umbral de Otsu para binarizar la imagen y encuentra el bounding box del contorno más grande.
            @param gray (numpy array) - Imagen en escala de grises.
            @return morph_clean (numpy array) - Imagen binarizada tras el umbral de Otsu.
        '''

        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        
        # Binarizar la imagen utilizando un umbral fijo
        binary_image = self._umbralización(gray)
        
        # Operación de cierre para limpiar la imagen
        kernel = np.ones((5, 5), np.uint8)
        morph_clean = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Encontrar los contornos en la imagen binarizada
        contours, _ = cv2.findContours(morph_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Asegurarse de que se haya encontrado al menos un contorno
        if contours:
            # Encontrar el contorno más grande
            contour = max(contours, key=cv2.contourArea)

            # Obtener el bounding box del contorno más grande
            x, y, w, h = cv2.boundingRect(contour)

            # Ampliar el bounding box en un rango (por ejemplo, 10% más grande en cada lado)
            expand_factor = 0.1  # Porcentaje de expansión
            x -= int(w * expand_factor)
            y -= int(h * expand_factor)
            w += int(w * expand_factor * 2)  # Expandir a ambos lados
            h += int(h * expand_factor * 2)  # Expandir arriba y abajo

            # Crear una máscara del tamaño de la imagen completa, con todos los valores a 0 (negro)
            mask = np.zeros_like(gray)
            
            # Dibujar un rectángulo blanco (255) en la máscara en el área del bounding box
            mask[y:y+h, x:x+w] = 255

            if colored:
                result = self.frame
                result[:,:,0] = cv2.bitwise_and(self.frame[:,:,0], morph_clean)
                result[:,:,1] = cv2.bitwise_and(self.frame[:,:,1], morph_clean)
                result[:,:,2] = cv2.bitwise_and(self.frame[:,:,2], morph_clean)
            else:
                result = cv2.bitwise_and(gray, morph_clean)

            if self.debug:
                cv2.imshow('image_cropped', result)
    
            return result
        else:
            print('Contornos no encontrados')
            self.message = "No se ha encontrado ningún contorno."
            self.message_type = 3
            return []
    
    def delete_aruco(self):
        '''
        Detecta un marcador Aruco en la imagen y lo elimina
            @param frame (numpy array) - Imagen en formato escala de grises.\n
        '''
        corners, ids, _ = cv2.aruco.detectMarkers(self.frame, self.aruco_dict, parameters=self.aruco_params)
        
        aruco_frame = deepcopy(self.frame)
        
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
            self.frame = deepcopy(cv2.bitwise_and(self.frame, mask))
            
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

    def _align_equidistant(self, points:list, side_length:float) -> list:
        ''' 
        Alinea los puntos detectados en una cuadrícula equidistante.
            @param points (list) - Lista de coordenadas de los puntos detectados.
            @param side_length (float) - Longitud de los lados para definir la cuadrícula.
            @return aligned_points_indices (list) - Índices de los puntos alineados en la cuadrícula.
        '''

        # Encontrar los valores mínimo de X y máximo de Y
        min_x = min(points, key=lambda p: p[0])[0]
        max_y = max(points, key=lambda p: p[1])[1]

        # Definir el número de elementos de la cuadrícula
        num_elements = 5

        # Crear las listas de valores equidistantes
        lista_resultado_x = [min_x + side_length * i for i in range(num_elements)]
        lista_resultado_y = [max_y - side_length * i for i in range(num_elements)]

        if self.debug:
            grid_image = deepcopy(self.contour_img)
            for x in lista_resultado_x:
                grid_image = cv2.line(grid_image, (x, lista_resultado_y[0]), (x, lista_resultado_y[4]), (255, 0, 255), 2)

            # Dibujar las líneas horizontales
            for y in lista_resultado_y:
                grid_image = cv2.line(grid_image, (lista_resultado_x[0], y), (lista_resultado_x[4], y), (255, 0, 255), 2)
            cv2.imshow('Imagen con Rejilla', grid_image)

        # Alinear los puntos a la cuadrícula más cercana
        aligned_points_indices = []  # Lista para almacenar los índices de los puntos alineados
        for x, y in points:
            # Obtener los índices del valor más cercano en lista_resultado_x e lista_resultado_y
            aligned_x_index = min(range(len(lista_resultado_x)), key=lambda i: abs(lista_resultado_x[i] - x))
            aligned_y_index = min(range(len(lista_resultado_y)), key=lambda i: abs(lista_resultado_y[i] - y))
            
            aligned_points_indices.append((aligned_x_index, aligned_y_index))
        
        return aligned_points_indices


    def _map_to_matrix(self, centers:list, colors:list, areas:list) -> np.ndarray:
        ''' 
        Mapea los centros detectados y sus colores a una matriz 5x5.
            @param centers (list) - Lista de centros detectados.
            @param colors (list) - Lista de colores correspondientes a los centros.
            @param areas (list) - Lista de áreas de los contornos detectados.
            @return matrix (numpy array) - Matriz 5x5 representando los colores en sus posiciones.
        '''
        # Calcular el área promedio para el umbral
        side_length = round(np.sqrt(sum(areas) / len(areas)))
        
        # Alinear los centros detectados
        aligned_centers = self._align_equidistant(centers, side_length)
        
        # Empaquetar centros con colores
        centers_with_colors_areas = zip(aligned_centers, colors)
        
        
        # Definir la matriz 5x5 vacía
        matrix = np.full((5,5), -1)
    
        # Mapear cada centro a la celda correspondiente
        for center, color in centers_with_colors_areas:
            # Determinar la fila y columna basándose en el tamaño promedio del cubo
            row = 4 - center[1] # Invertir el eje Y
            col = center[0]
            matrix[row][col] = color

        return matrix

    
    def calibrate_cube_area(self) -> None:
        ''' 
        Obtiene el area de un cubo para utilizarlo como medida promedio.
            @return None
        '''
        cropped_frame = self._get_cubes_location(colored=True)

        separated_colors = self._get_contrast_img(cropped_frame)
        
        areas = []
        for color in separated_colors: # [red, green, blue, yellow]
            contours, _ = cv2.findContours(color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            try:
                for cnt in contours:
                    areas = cv2.contourArea(cnt)
            except:
                pass
        self.base_area = np.max(areas)
        if self.base_area < 2000:
            try:
                self.message = "El cubo se ha calibrado correctamente."
                self.message_type = 1
                self.base_area = float(self.base_area)
                file_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:os.path.dirname(os.path.abspath(__file__)).split('/').index('proyecto_final')+1])
                
                with open(f'{file_path}/data/necessary_data/cube_area.yaml', 'w') as file:
                    yaml.dump(self.base_area, file)
            except:
                print('Fallo al calibrar')
                self.message = "Ha habido un error al calibrar el área del cubo."
                self.message_type = 3
        

    def process_image(self, frame:np.ndarray, calibration:bool = False, mostrar:bool=False, debug:bool = False)->tuple:
        ''' 
        Ejecuta el flujo completo de procesamiento de una imagen para detectar colores y posiciones de cubos.
            - Almacena la imagen original y una copia para dibujar contornos.
            - Realiza el preprocesamiento para limpiar la imagen y detectar bordes.
            - Encuentra y filtra los contornos relevantes basándose en su tamaño.
            - Identifica los colores predominantes dentro de cada contorno y calcula las posiciones en una matriz.
            - Dibuja contornos y anotaciones sobre la imagen de salida.
            @param frame (numpy array) - Imagen de entrada en formato BGR.
            @param area_size (int) - Umbral mínimo de área para considerar un contorno como válido. Por defecto, 2000.
            @param mostrar (bool) - Si es True, muestra la imagen resultante.
            @param debug (bool) - Si es True, muestra imagenes intermedias del procesamiento.
            @return (tuple) - Una tupla con:
                - matrix (numpy array) - Matriz 5x5 que representa la cuadrícula con los colores detectados.
                - contour_img (numpy array) - Imagen con los contornos y anotaciones dibujados.
        '''
        self.message = "Se ha empezado a calibrar la imagen."
        self.message_type = 1
        self.matrix = deepcopy(np.full((self.matrix_size, self.matrix_size), -1))
        self.frame = deepcopy(frame)
        self.contour_img = deepcopy(self.frame)
        self.debug = debug

        self.delete_aruco()
        
        if calibration:
            self.calibrate_cube_area()
        else:
            # Preprocesamiento de la imagen
            self._preprocess_image()
            
            if len(self.centers) > 0:
                self.matrix = self._map_to_matrix(self.centers, self.colors, self.areas)
                self.message = "Se ha procesado la imagen."
                self.message_type = 1

            if mostrar:
                print(f'Matriz Top:\n {self.matrix}')
                cv2.imshow('Contoured Image', self.contour_img)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()

        return self.matrix, self.contour_img



# Ejecutar el programa
if __name__ == "__main__":
    use_cam = True
    num_cam = 0
    num_img = 1
    
    if use_cam:
        cam = cv2.VideoCapture(num_cam)
        if cam.isOpened():
            _, frame = cam.read()
    else:
        file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace('\\', "/")
        # frame = deepcopy(cv2.imread(f'{file_path}/data/figuras_planta/Cube_Calibration.png'))
        frame = deepcopy(cv2.imread(f'{file_path}/data/figuras_planta/Figura_{num_img}_S.png'))

    processor = ImageProcessor_Top()
    
    #matriz, imagen = processor.process_image(frame, mostrar = True, debug=True, calibration=False)
    matriz, imagen = processor.process_image(frame, calibration=True)
    