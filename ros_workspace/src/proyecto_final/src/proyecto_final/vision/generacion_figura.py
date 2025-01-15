#!/usr/bin/env python3

# Importación de Librerias
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

class FigureGenerator:
    def __init__(self) -> None:
        """
        Inicialización de la clase, define las variables de mensajes para el tkinter.
            self.message (str) - Define el mensaje a publicar
            self.message_type (int) - Indica el tipo de mensaje
        """
        self.message:str = None
        self.message_type:int = 0 # 1=Info, 2=Warn, 3=Error

    def generate_figure_from_matrix(self, matriz_planta:list, matriz_alzado:list, matriz_perfil:list, figsize:tuple=(3,3), paint:bool = False, tkinter:bool=False):
        """
        Función principal de la clase, toma como entrada las tres matrices, analiza la anchura, profundidad y altura de la figura
        y compara los datos de las diferentes matrices hasta obtener la figura tridimensional resultante.
            - Analiza las matrices en busca de los limites de la figura
            - Comprueba que los datos almacenados en las matrices son correctos
            - Compara las matrices entre si para asignar la posición y valor de cada cubo
            - En caso de pedirlo muestra la figura resultante o solamente la devuelve
                @param matriz_planta (list) - Matriz resultante del analisis de la vista planta
                @param matriz_alzado (list) - Matriz resultante del analisis de la vista alzada
                @param matriz_perfil (list) - Matriz resultante del analisis de la vista perfil
                @param figsize = (3,3) (tuple) - Tamaño de la figura donde se representa la figura
                @param paint (bool) - En caso de True realiza un plot de la figura 3D resultante
                @param tkinter (bool) - En caso de True indica que el resultado debe tener un formato valido para tkinter
            paint = True and tkinter = True 
                @return fig3D (plt.Figure) - Plot de la Figura resultante
            (paint = True and tkinter = False) or paint = False 
                @return self.matrix3D (list) - Matriz 3D de la figura resultante
        """
        # Calculo de las dimensiones de la figura y recorte de las matrices a los limites obtenidos
        profundidad, anchura, matriz_planta_recortada = self._cut_matrix_finding_shape(matriz_planta)
        altura, anchura2, matriz_alzado_recortada = self._cut_matrix_finding_shape(matriz_alzado)
        altura2, profundidad2, matriz_perfil_recortada = self._cut_matrix_finding_shape(matriz_perfil)
        
        # Comprobación del estado de las matrices adquiridas
        if (altura is None) and (anchura is None) and (profundidad is None): # Caso para tkinter
            return self._paint_matrix(np.array([[[]]]), figsize, tkinter)
        
        if altura is None or anchura is None or profundidad is None: # Matrices Vacias
            self.message = "Las matrices introduccidas son inválidas, no se puede realizar figura 3D."
            self.message_type = 3
            print('Matrices Invalidas')
            return self._paint_matrix(np.array([[[]]]), figsize, tkinter)
        
        if anchura != anchura2 or profundidad != profundidad2 or altura != altura2: # Discrepancia entre matrices
            print('Matrices Invalidas')
            self.message = "Las matrices introduccidas son inválidas, no se puede realizar figura 3D."
            self.message_type = 3
            return self._paint_matrix(np.array([[[]]]), figsize, tkinter)

        # Definición de las matriz 3D
        matriz3D = deepcopy(np.full((anchura, altura, profundidad), -1))

        # Comparación de las matrices de cada pespectiva
        for fila_planta in range(0, profundidad): # Cara mas cercana al Alzado
            for columna_planta in range(anchura-1, -1, -1): # Cara mas cercana al Perfil
                if matriz_planta_recortada[fila_planta][columna_planta] != -1: # Color de planta != -1
                    color_cubo = matriz_planta_recortada[fila_planta][columna_planta]

                    # Definición de Relaciones entre matrices
                    columna_perfil = profundidad - fila_planta -1
                    columna_alzado = x = anchura - columna_planta - 1 # Definición de X en matriz
                    y = fila_planta # Definición de posicion Y en matriz

                    cube_found = False # Booleano que indica la aparición del cubo mas elevado de la columna
                    for fila_alzado in range(altura):
                        fila_perfil = fila_alzado
                        z = altura - fila_alzado - 1 # Definición de Z en la matriz

                        # Visible desde Planta + Coincidencia entre todas las matrices
                        if not cube_found and matriz_planta_recortada[fila_planta][columna_planta] == matriz_perfil_recortada[fila_perfil][columna_perfil] and matriz_planta_recortada[fila_planta][columna_planta] == matriz_alzado_recortada[fila_alzado][columna_alzado]:
                            matriz3D[x][z][y] = color_cubo
                            # En coincidencias se asigna un valor 5 en la posicion del cubo en alzado y perfil para facilitar distinción
                            matriz_perfil_recortada[fila_perfil][columna_perfil] = matriz_alzado_recortada[fila_alzado][columna_alzado] = 5
                            cube_found = True

                        # Visible desde Planta+ Coincidencia entre Planta y Alzado, Perfil != -1 (Cubo Tapado)
                        elif not cube_found and matriz_planta_recortada[fila_planta][columna_planta] == matriz_alzado_recortada[fila_alzado][columna_alzado] and matriz_perfil_recortada[fila_perfil][columna_perfil] != -1:
                            matriz3D[x][z][y] = color_cubo
                            matriz_alzado_recortada[fila_alzado][columna_alzado] = 5
                            cube_found = True

                        # Visible desde Planta + Coincidencia entre Planta y Perfil, Alzado != -1 (Cubo Tapado)
                        elif not cube_found and matriz_planta_recortada[fila_planta][columna_planta] == matriz_perfil_recortada[fila_perfil][columna_perfil] and matriz_alzado_recortada[fila_alzado][columna_alzado] != -1:
                            if columna_planta == profundidad-1:
                                pass
                            else:
                                matriz3D[x][z][y] = color_cubo
                                matriz_perfil_recortada[fila_perfil][columna_perfil] = 5
                                cube_found = True

                        # Oculto desde Planta+ Coincidencia entre Perfil y Alzado, valor de cubo != 5 (Cubo no representado previamente)
                        elif cube_found and matriz_perfil_recortada[fila_perfil][columna_perfil] == matriz_alzado_recortada[fila_alzado][columna_alzado] and matriz_alzado_recortada[fila_alzado][columna_alzado] != 5:
                            matriz3D[x][z][y] = matriz_perfil_recortada[fila_perfil][columna_perfil]
                            matriz_perfil_recortada[fila_perfil][columna_perfil] = matriz_alzado_recortada[fila_alzado][columna_alzado] = 5

                        # Oculto desde Planta + Visto desde Perfil pero tapado desde Alzado
                        elif cube_found and matriz_perfil_recortada[fila_perfil][columna_perfil] != 5 and matriz_alzado_recortada[fila_alzado][columna_alzado] == 5:
                            # Se comprueba que no sea un cubo con mismo color que aparezca antes
                            for i in range(columna_planta, anchura):
                                if matriz_perfil_recortada[fila_perfil][columna_perfil] == matriz_planta_recortada[fila_planta][i]:
                                    matriz3D[x][z][y] = 4
                            if matriz3D[x][z][y] != 4:
                                matriz3D[x][z][y] = matriz_perfil_recortada[fila_perfil][columna_perfil]
                                matriz_perfil_recortada[fila_perfil][columna_perfil] = 5

                        # Oculto desde Planta + Visto desde Alzado pero tapado desde Perfil
                        elif cube_found and matriz_alzado_recortada[fila_alzado][columna_alzado] != 5 and matriz_perfil_recortada[fila_perfil][columna_perfil] == 5:
                            # Se comprueba que no sea un cubo con mismo color que aparezca antes
                            for i in range(fila_planta, 0):
                                if matriz_alzado_recortada[fila_alzado][columna_alzado] == matriz_planta_recortada[i][columna_planta]:
                                    matriz3D[x][z][y] = 4
                            if matriz3D[x][z][y] != 4:
                                matriz3D[x][z][y] = matriz_alzado_recortada[fila_alzado][columna_alzado]
                                matriz_alzado_recortada[fila_alzado][columna_alzado] = 5

                        # Oculto desde Planta + Visible desde unicamente una perspectiva
                        elif cube_found and matriz_alzado_recortada[fila_alzado][columna_alzado] != 5 and matriz_perfil_recortada[fila_perfil][columna_perfil] != 5:
                            if columna_planta != profundidad-1:
                                matriz3D[x][z][y] = matriz_perfil_recortada[fila_perfil][columna_perfil]
                                matriz_perfil_recortada[fila_perfil][columna_perfil] = 5
                            else:
                                matriz3D[x][z][y] = matriz_alzado_recortada[fila_alzado][columna_alzado]
                                matriz_alzado_recortada[fila_alzado][columna_alzado] = 5

                        # Cubo oculto desde las 3 Perspectivas
                        elif cube_found and matriz_alzado_recortada[fila_alzado][columna_alzado] == 5 and matriz_perfil_recortada[fila_perfil][columna_perfil] == 5:
                            matriz3D[x][z][y] = 4

                    # Cubo unicamente visible desde Planta (Se asume altura = 0)
                    if not cube_found:
                        z= 0
                        matriz3D[x][z][y] = color_cubo
                            
        self.message = "Se ha calculado la figura 3D"
        self.message_type = 1
        if paint:
            return self._paint_matrix(matriz3D, figsize, tkinter)
        else:
            return matriz3D
                
    def _cut_matrix(self, matriz:list, num_filas:int, num_columnas:int) -> list:
        """
        Recorta la matriz al tamaño definido
            @param matriz (list) - Matriz a recortar
            @param num_filas (int) - Numero de fila de la matriz recortada
            @param num_columnas (int) - Numero de columnas de la matriz recortada
            @return matriz_recortada (list) - Matriz recortada
        """
        # Recortar la matriz hasta el tamaño deseado
        matriz_recortada = [fila[:num_columnas] for fila in matriz[5-num_filas:5]]
        return matriz_recortada

    def _cut_matrix_finding_shape(self, matrix:list) -> tuple:
        """
        Busca el primer elemento distinto de -1 en las filas y columnas de la matriz y 
        devuelve las dimensiones de la matriz donde se encuentran los datos, así como la 
        matriz recortada
            @param matriz (list) - Matriz a analizar y recortar
            @return (tuple)
                i_max - primera fila donde hay valores distintos de -1
                j_max - primera columna donde hay valores distintos de -1
                matriz_recortada - matriz donde se encuentran los valores distintos de -1 de la matriz original
        """
        i_max = None
        j_max = None
        # Iteramos sobre las filas de la última a la primera
        for i in range(len(matrix)): # Desde la primera fila hasta la ultima
            for j in range(len(matrix[i])): # Desde la última columna hasta la primera
                if matrix[i][4-j] != -1: # Si encontramos un valor distinto a -1
                    if i_max == None:
                        i_max = 5-i # Devolver la fila
                if matrix[j][4-i] != -1:
                    if j_max == None:
                        j_max = 5-i # Devolver la columna inversa
                if j_max != None and i_max != None:
                    return i_max, j_max, self._cut_matrix(matrix, i_max, j_max)

        return None, None, matrix # Si no hay valores diferentes a -1, retornar None

    def _paint_matrix(self, matriz3D, figsize:tuple=(3,3), tkinter:bool=False):
        """
        Plotea la matriz para visualizarla en un grafico 3D
            @param matriz3D (list) - Matriz 3D con los datos de la figura
            @param figsize (tuple) - Tamaño de la figura mostrada en pantalla
            @param tkinter (bool) - True si se requiere un resultado para tkinter
        tkinter = True
            @return fig3d (plt.Figure) - Grafico 3D de la figura
        tkinter = False
            @return matriz3D (list) - Matriz 3D de la figura
        """

        # Definición de las variables
        size = 1 # Tamaño del cubo
        color_map = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'gray'} # Colores del cubo
        anchura, altura, profundidad = matriz3D.shape # Dimensiones de la figura

        # Dependiendo de si lo mostramos en la interfaz o no, se realizan dos acciones de plt
        if tkinter:
            fig_3d = plt.Figure(figsize=figsize, dpi=100)
            ax = fig_3d.add_subplot(111, projection='3d')
        else:
            fig_3d, ax = plt.subplots(figsize=figsize, dpi=100, subplot_kw={'projection': '3d'})


        # Se recorre la matriz para dibujar la figura
        for x in range(anchura):
            for y in range(profundidad):
                for z in range(altura): 
                    if matriz3D[x, z, y] != -1:
                        ax.bar3d(x*size,y*size,z*size, size, size, size, color=color_map[matriz3D[x, z, y]])
        
        # Configurar los límites del gráfico
        ax.set_xlim([0, 5 * size])
        ax.set_ylim([0,  5 * size])
        ax.set_zlim([0, 5 * size])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Ajustar la vista para que la figura se vea al fondo
        ax.view_init(elev=30, azim=-60)

        # Mostrar o devolver la figura
        if tkinter:
            return fig_3d
        else:
            plt.show()
            return matriz3D