#!/usr/bin/python3

import subprocess
import numpy as np
from time import sleep
from copy import deepcopy
from sympy import Q
import ttkbootstrap as ttk
import cv2, rospy
import matplotlib.pyplot as plt
from tkinter import messagebox
from geometry2D import Geometry2D
from PIL.ImageTk import PhotoImage
from cv_bridge import CvBridge
from ttkbootstrap.constants import *
import sensor_msgs.msg
from threading import Thread
from PIL import Image, Image, ImageDraw, ImageFont, ImageTk
from proyecto_final.rob_main import SecuenceCommander
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from proyecto_final.vision.generacion_figura import FigureGenerator


class RoboticaTkinter:
    """
    Clase que representa la pestaña de Robótica en la interfaz gráfica.
    """
    def __init__(self) -> None:
        """
        Constructor de la clase RoboticaTkinter.
        """
        # --- ROOT ---
        self.root = ttk.Window(title="Robótica", themename="vision")
        self.root.resizable(True, True)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width//2-70}x{screen_height}+0+0")


        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.Geometry3D = FigureGenerator()
        self.Geometry2D = Geometry2D()
        self.RobMain = SecuenceCommander()

        # --- ROS COMPONENTS ---
        self.bridge = CvBridge()
        self.subs_cam_top = rospy.Subscriber('/top_cam/image_raw', sensor_msgs.msg.Image, self.cb_image_top)
        self.subs_cam_alzado = rospy.Subscriber('/alzado_cam/image_raw', sensor_msgs.msg.Image, self.cb_image_alzado)
        self.subs_cam_lateral = rospy.Subscriber('/perfil_cam/image_raw', sensor_msgs.msg.Image, self.cb_image_lateral)
        self.subs_cam_hand_lateral = rospy.Subscriber('/lateral_hand_cam/image_raw', sensor_msgs.msg.Image, self.cb_hand_lateral)
        self.subs_cam_hand_top = rospy.Subscriber('/top_hand_cam/image_raw', sensor_msgs.msg.Image, self.cb_hand_top)

        # --- VARIABLES ---
        self.LF_rviz = None
        self.LF_quinta_fila = None
        self.img_aspect_ratio = 0.4
        self.Logic_MakeFigure = False
        self.Logic_DetectFigure = False
        self.Logic_Calibrate = True
        self.Contador_Calibrate = 0

        # --- INICIALIZACIÖN ---
        self.estilo()
        self.start_robot_tab()
        self._rviz_launch()
        self.root.mainloop()
        
    
    def start_robot_tab(self):
        """
        Inicializa la pestaña de Robótica.
        """
        self.frame_rob = ttk.Frame(self.root, borderwidth=0)
        self.frame_rob.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        #Configuración grid
        self.frame_rob.grid_columnconfigure(0, weight=1)
        self.frame_rob.grid_rowconfigure(0, weight=1)
        self.frame_rob.grid_rowconfigure(1, weight=2)
        self.frame_rob.grid_rowconfigure(2, weight=2)
        self.frame_rob.grid_rowconfigure(3, weight=1)
        self.frame_rob.grid_rowconfigure(4, weight=1)

        self._fila_terminal()
        self._primera_fila()


    def stop_robot_tab(self):
        """
        Detiene la pestaña de Robótica.
        """
        self.close_rviz()
    

    def manual_tab(self):
        """
        Crea la pestaña de Robótica en modo manual.
        """
        self._clear_tab()
        self._fila_camaras_manual()
        self._fila_geometria_manual()
        self._fila_acciones_man()


    def auto_tab(self):
        """
        Crea la pestaña de Robótica en modo automático.
        """
        self._clear_tab()
        self._fila_camaras_auto()
        self._fila_geometria_auto()
        self._fila_acciones_auto()

    
    def _clear_tab(self):
        """
        Limpia la pestaña de Robótica.
        """
        for child in self.frame_rob.winfo_children():
            if child not in {self.F_primera_fila, self.LF_quinta_fila}:
                child.destroy()

        self.widgets_eliminados = [child for child in self.frame_rob.winfo_children() if child not in {self.F_primera_fila, self.LF_quinta_fila}]
        

    def _primera_fila(self):
        """
        Crea la primera fila de la pestaña de Robótica.
        """
        # --- FRAME: Primera fila ---
        self.F_primera_fila = ttk.Frame(self.frame_rob)
        self.F_primera_fila.grid(row=0, column=0, sticky="nsew",padx=10, pady=5)
        self.F_primera_fila.grid_rowconfigure(0, weight=1)
        self.F_primera_fila.grid_columnconfigure(0, weight=1)
        self.F_primera_fila.grid_columnconfigure(1, weight=1)

        self.LF_primera_col = ttk.LabelFrame(self.F_primera_fila, text="  Modo Funcionamiento  ")
        self.LF_primera_col.grid(row=0, column=0, sticky="nsew", padx=[0,10], pady=0)
        self.LF_primera_col.grid_rowconfigure(0, weight=1)
        self.LF_primera_col.grid_columnconfigure(0, weight=1)


        self.F_primera_col = ttk.Frame(self.LF_primera_col)
        self.F_primera_col.grid(row=0, column=0, sticky="nsew",padx=10, pady=10)
        self.F_primera_col.grid_rowconfigure(0, weight=1)
        self.F_primera_col.grid_columnconfigure(0, weight=2)
        self.F_primera_col.grid_columnconfigure(1, weight=1)
        self.F_primera_col.grid_columnconfigure(2, weight=2)

        # --- ELEMENTS ---
        # Text MANUAL
        self.L_automatico = ttk.Label(self.F_primera_col, text="AUTOMÁTICO" )
        self.L_automatico.grid(row=0, column=0, sticky="e", padx=[0, 10])

            # Checkbutton
        self.V_modo = ttk.IntVar()  # Variable para el estado del Checkbutton
        self.CB_modo = ttk.Checkbutton(
            self.F_primera_col,
            variable=self.V_modo,
            command=self.cambio_funcionamiento,
            bootstyle="primary-round-toggle")
        self.CB_modo.grid(row=0, column=1)

        # Text MANUAL
        self.L_manual = ttk.Label(self.F_primera_col, text="MANUAL" )
        self.L_manual.grid(row=0, column=2, sticky="w", padx=[10, 0])

        # --- SEGUNDA COLUMNA ---
        self.LF_segunda_col = ttk.LabelFrame(self.F_primera_fila, text="  Control del Robot  ")
        self.LF_segunda_col.grid(row=0, column=1, sticky="nsew", padx=[10,0], pady=0)
        self.LF_segunda_col.grid_rowconfigure(0, weight=1)
        self.LF_segunda_col.grid_columnconfigure(0, weight=1)


        self.F_segunda_col= ttk.Frame(self.LF_segunda_col)
        self.F_segunda_col.grid(row=0, column=0, sticky="nsew",padx=0, pady=0)
        self.F_segunda_col.grid_rowconfigure(0, weight=1)
        self.F_segunda_col.grid_columnconfigure(0, weight=1)

        self.B_Stop = ttk.Button(self.F_segunda_col, bootstyle="danger", text="STOP")
        self.B_Stop.grid(row=0, column=0, sticky="nsew",padx=10, pady=10)

        self.cambio_funcionamiento()


    def cambio_funcionamiento(self):
        """
        Cambia el modo de funcionamiento de la interfaz gráfica.
        """
        # --- MANUAL ---
        if self.V_modo.get() == 1: 
            self.L_manual.config(font=("Montserrat", 10, "bold"), foreground="#8c85f7")
            self.L_automatico.config(font=("Montserrat", 10), foreground="#474B4E")
            self.manual_tab() 
        
        # --- AUTOMATICO ---
        else:
            self.L_manual.config(font=("Montserrat", 10), foreground="#474B4E")
            self.L_automatico.config(font=("Montserrat", 10, "bold"), foreground="#8c85f7")
            self.auto_tab()

    def _fila_camaras_auto(self):
        """
        Crea la fila de cámaras en modo automático.
        """
        # --- FRAME ---
        # Label Frame
        self.LF_segunda_fila = ttk.LabelFrame(self.frame_rob, text="  Visualización Cámaras  ")
        self.LF_segunda_fila.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.LF_segunda_fila.grid_rowconfigure(0, weight=1)
        self.LF_segunda_fila.grid_columnconfigure(0, weight=1)

        # Inner Frame
        self.F_segunda_fila = ttk.Frame(self.LF_segunda_fila)
        self.F_segunda_fila.grid(row=0, column=0, sticky="nsew",padx=10, pady=10)
        self.F_segunda_fila.grid_rowconfigure(0, weight=1)
        self.F_segunda_fila.grid_columnconfigure(0, weight=1)
        self.F_segunda_fila.grid_columnconfigure(1, weight=1)
        self.F_segunda_fila.grid_columnconfigure(2, weight=1)

        # --- ELEMENTS ---
        # Image Frame 1
        self.F_image_1 = ttk.Frame(self.F_segunda_fila)
        self.F_image_1.grid_rowconfigure(0, weight=1)
        self.F_image_1.grid_rowconfigure(1, weight=1)
        self.F_image_1.grid_columnconfigure(0, weight=1)
        self.F_image_1.grid(row=0, column=0, sticky="nsew")

        ttk.Label(self.F_image_1,text="CÁMARA SUPERIOR", 
                  font=("Montserrat SemiBold", 10), 
                  foreground="#000000").grid(row=0, column=0, pady=[0,5],sticky="N")
        
        self.L_img_top = ttk.Label(self.F_image_1)
        self.L_img_top.grid(row=1, column=0)

        self.F_image_2 = ttk.Frame(self.F_segunda_fila)
        self.F_image_2.grid_rowconfigure(0, weight=1)
        self.F_image_2.grid_rowconfigure(1, weight=1)
        self.F_image_2.grid_columnconfigure(0, weight=1)
        self.F_image_2.grid(row=0, column=1, sticky="nsew")

        ttk.Label(self.F_image_2,text="CÁMARA ALZADO", 
                  font=("Montserrat SemiBold", 10),
                  foreground="#000000").grid(row=0, column=0, pady=[0,5],sticky="N")
        
        self.L_img_alzado = ttk.Label(self.F_image_2)
        self.L_img_alzado.grid(row=1, column=0)

        # Image Frame 3
        self.F_image_3 = ttk.Frame(self.F_segunda_fila)
        self.F_image_3.grid_rowconfigure(0, weight=1)
        self.F_image_3.grid_rowconfigure(1, weight=1)
        self.F_image_3.grid_columnconfigure(0, weight=1)
        self.F_image_3.grid(row=0, column=2, sticky="nsew")

        ttk.Label(self.F_image_3,text="CÁMARA LATERAL", 
                  font=("Montserrat SemiBold", 10), 
                  foreground="#000000").grid(row=0, column=0, pady=[0,5],sticky="N")
        
        self.L_img_lateral = ttk.Label(self.F_image_3)
        self.L_img_lateral.grid(row=1, column=0)
        

        self._update_images()
    
    def _fila_camaras_manual(self):
        """
        Crea la fila de cámaras en modo manual.
        """
        # --- FRAME ---
        # Label Frame
        self.LF_segunda_fila = ttk.LabelFrame(self.frame_rob, text="  Visualización Cámaras  ")
        self.LF_segunda_fila.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.LF_segunda_fila.grid_rowconfigure(0, weight=1)
        self.LF_segunda_fila.grid_columnconfigure(0, weight=1)

        # Inner Frame
        self.F_segunda_fila = ttk.Frame(self.LF_segunda_fila)
        self.F_segunda_fila.grid(row=0, column=0, sticky="nsew",padx=10, pady=10)
        self.F_segunda_fila.grid_rowconfigure(0, weight=1)
        self.F_segunda_fila.grid_columnconfigure(0, weight=1)
        self.F_segunda_fila.grid_columnconfigure(1, weight=1)

        # --- ELEMENTS ---
        # Image Frame 1
        self.F_image_1 = ttk.Frame(self.F_segunda_fila)
        self.F_image_1.grid_rowconfigure(0, weight=1)
        self.F_image_1.grid_rowconfigure(1, weight=1)
        self.F_image_1.grid_columnconfigure(0, weight=1)
        self.F_image_1.grid(row=0, column=0, sticky="nsew")

        ttk.Label(self.F_image_1,text="CÁMARA SUPERIOR", 
                  font=("Montserrat SemiBold", 10), 
                  foreground="#000000").grid(row=0, column=0, pady=[0,5],sticky="N")
        
        self.L_img_top = ttk.Label(self.F_image_1)
        self.L_img_top.grid(row=1, column=0)

        self.F_image_2 = ttk.Frame(self.F_segunda_fila)
        self.F_image_2.grid_rowconfigure(0, weight=1)
        self.F_image_2.grid_rowconfigure(1, weight=1)
        self.F_image_2.grid_columnconfigure(0, weight=1)
        self.F_image_2.grid(row=0, column=1, sticky="nsew")

        ttk.Label(self.F_image_2,text="CÁMARA LATERAL", 
                  font=("Montserrat SemiBold", 10),
                  foreground="#000000").grid(row=0, column=0, pady=[0,5],sticky="N")
        
        self.L_img_lateral = ttk.Label(self.F_image_2)
        self.L_img_lateral.grid(row=1, column=0)
        

        self._update_images()
    
    def _fila_geometria_auto(self):
        """
        Crea la fila de geometría en modo automático.
        """
        # --- FRAME: Primera fila ---
        self.LF_tercera_fila = ttk.LabelFrame(self.frame_rob, text="  Representación Gráfica  ")
        self.LF_tercera_fila.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        self.LF_tercera_fila.grid_rowconfigure(0, weight=1)
        self.LF_tercera_fila.grid_columnconfigure(0, weight=1)


        self.F_tercera_fila = ttk.Frame(self.LF_tercera_fila)
        self.F_tercera_fila.grid(row=0, column=0, sticky="nsew",padx=10, pady=10)
        self.F_tercera_fila.grid_rowconfigure(0, weight=1)
        self.F_tercera_fila.grid_columnconfigure(0, weight=1)
        self.F_tercera_fila.grid_columnconfigure(1, weight=1)

        # --- ELEMENTS ---
        
        self.geometry1_frame = ttk.Frame(self.F_tercera_fila)
        self.geometry1_frame.grid_rowconfigure(0, weight=1)
        self.geometry1_frame.grid_rowconfigure(1, weight=1)
        self.geometry1_frame.grid_columnconfigure(0, weight=1)
        self.geometry1_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ttk.Label(self.geometry1_frame,text="REPRESENTACIÓN FIGURA 3D", 
                  font=("Montserrat SemiBold", 10), 
                  foreground="#000000").grid(row=0, column=0, pady=[0,5],sticky="N")
        
        fig_3d = self.Geometry3D._paint_matrix(np.array([[[]]]), tkinter=True, figsize=(3,2.5))

        # Mostrar la figura 3D en el canvas
        self.canvas_3d = FigureCanvasTkAgg(fig_3d, self.geometry1_frame)
        self.canvas_3d.get_tk_widget().grid(row=1, column=0)


        self.geometry2_frame = ttk.Frame(self.F_tercera_fila)
        self.geometry2_frame.grid_rowconfigure(0, weight=1)
        self.geometry2_frame.grid_rowconfigure(1, weight=1)
        self.geometry2_frame.grid_columnconfigure(0, weight=1)
        self.geometry2_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        ttk.Label(self.geometry2_frame,text="REPRESENTACIÓN INICIAL DEL WS", 
                  font=("Montserrat SemiBold", 10), 
                  foreground="#000000").grid(row=0, column=0, pady=[0,5],sticky="N")
        
        fig_2d = self.Geometry2D.draw_2d_space([], tkinter=True,figsize=(4,3))

        # Mostrar la figura 3D en el canvas
        self.canvas_2d = FigureCanvasTkAgg(fig_2d, self.geometry2_frame)
        self.canvas_2d.get_tk_widget().grid(row=1, column=0)
    
    def _fila_geometria_manual(self):
        """
        Crea la fila de geometría en modo manual.
        """
        # --- FRAME: Tercera fila ---
        self.F_tercera_fila = ttk.Frame(self.frame_rob)
        self.F_tercera_fila.grid(row=2, column=0, sticky="nsew",padx=10, pady=10)
        self.F_tercera_fila.grid_rowconfigure(0, weight=1)
        self.F_tercera_fila.grid_columnconfigure(0, weight=1)
        self.F_tercera_fila.grid_columnconfigure(1, minsize=self.frame_rob.winfo_width()//2)

        # --- LABEL FRAME: Representación Gráfica ---

        self.LF_tercera_fila_geometria = ttk.LabelFrame(self.F_tercera_fila, text="  Representación Gráfica  ")
        self.LF_tercera_fila_geometria.grid(row=0, column=0, sticky="nsew", padx=[0,10], pady=5)
        self.LF_tercera_fila_geometria.grid_rowconfigure(0, weight=1)
        self.LF_tercera_fila_geometria.grid_columnconfigure(0, weight=1)


        self.F_tercera_fila_geometria = ttk.Frame(self.LF_tercera_fila_geometria)
        self.F_tercera_fila_geometria.grid(row=0, column=0, sticky="nsew",padx=10, pady=10)
        self.F_tercera_fila_geometria.grid_rowconfigure(0, weight=1)
        self.F_tercera_fila_geometria.grid_columnconfigure(0, weight=1)
      
        self.fig = plt.Figure(figsize=(3, 3))
        self.ax = self.fig.add_subplot(111, projection='3d')
        

        self.setup_figure()

        # Mostrar la figura 3D en el canvas
        self.canvas_3d = FigureCanvasTkAgg(self.fig, self.F_tercera_fila_geometria)
        self.canvas_3d.get_tk_widget().grid(row=0, column=0)


        # --- LABEL FRAME: Gestos de la Mano ---
        self.LF_tercera_fila_gestos = ttk.LabelFrame(self.F_tercera_fila, text="  Gestos de la Mano  ")
        self.LF_tercera_fila_gestos.grid(row=0, column=1, sticky="nsew", padx=[10,0], pady=5)
        self.LF_tercera_fila_gestos.grid_rowconfigure(0, weight=1)
        self.LF_tercera_fila_gestos.grid_columnconfigure(0, weight=1)

        self.F_tercera_fila_gestos = ttk.Frame(self.LF_tercera_fila_gestos)
        self.F_tercera_fila_gestos.grid(row=0, column=0, sticky="nsew",padx=10, pady=10)
        self.F_tercera_fila_gestos.grid_rowconfigure(0, weight=1)
        self.F_tercera_fila_gestos.grid_columnconfigure(0, weight=1)

        self.L_img_mano = ttk.Label(self.F_tercera_fila_gestos)
        self.L_img_mano.grid(row=0, column=0)
        


    def _fila_acciones_auto(self):
        """
        Crea la fila de acciones en modo automático.
        """
        # --- FRAME: Primera fila ---
        self.LF_cuarta_fila = ttk.LabelFrame(self.frame_rob, text="  Acciones  ")
        self.LF_cuarta_fila.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
        self.LF_cuarta_fila.grid_rowconfigure(0, weight=1)
        self.LF_cuarta_fila.grid_columnconfigure(0, weight=1)


        self.F_cuarta_fila = ttk.Frame(self.LF_cuarta_fila)
        self.F_cuarta_fila.grid(row=0, column=0, sticky="nsew",padx=10, pady=10)
        self.F_cuarta_fila.grid_rowconfigure(0, weight=1)
        self.F_cuarta_fila.grid_columnconfigure(0, weight=1)
        self.F_cuarta_fila.grid_columnconfigure(1, weight=1)
        self.F_cuarta_fila.grid_columnconfigure(2, weight=1)

        # --- ELEMENTS ---
        self.B_MakeFigure = ttk.Button(self.F_cuarta_fila, command=self.command_detect_figure, bootstyle="primary", text="DETECT FIGURE")
        self.B_MakeFigure.grid(row=0, column=0, sticky="nsew",padx=10, pady=0)

        self.B_Calibrate = ttk.Button(self.F_cuarta_fila, command=self.command_calibrate_aruco, bootstyle="primary", text="CALIBRAR ARUCO")
        self.B_Calibrate.grid(row=0, column=1, sticky="nsew",padx=10, pady=0)

        self.B_RemakeFigure = ttk.Button(self.F_cuarta_fila, command=self.command_remake_figure, bootstyle="primary", state="disabled", text="RECONSTRUCT FIGURE")
        self.B_RemakeFigure.grid(row=0, column=2, sticky="nsew",padx=10, pady=0)
    
    def _fila_acciones_man(self):
        """
        Crea la fila de acciones en modo manual.
        """
        # --- FRAME: Primera fila ---
        self.LF_cuarta_fila = ttk.LabelFrame(self.frame_rob, text="  Acciones  ")
        self.LF_cuarta_fila.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
        self.LF_cuarta_fila.grid_rowconfigure(0, weight=1)
        self.LF_cuarta_fila.grid_columnconfigure(0, weight=1)


        self.F_cuarta_fila = ttk.Frame(self.LF_cuarta_fila)
        self.F_cuarta_fila.grid(row=0, column=0, sticky="nsew",padx=10, pady=10)
        self.F_cuarta_fila.grid_rowconfigure(0, weight=1)
        self.F_cuarta_fila.grid_columnconfigure(0, weight=1)


        # --- ELEMENTS ---
        self.V_HandControl = ttk.IntVar(0)
        self.B_HandControl = ttk.Checkbutton(self.F_cuarta_fila, 
                                             variable=self.V_HandControl, 
                                             command=self.command_hand_control, 
                                             bootstyle="toolbutton-primary", 
                                             text="EMPEZAR CONTROL POR MANO")
        self.B_HandControl.grid(row=0, column=0, sticky="nsew",padx=10, pady=0)


    def _fila_terminal(self):
        """
        Crea la fila de terminal en la interfaz gráfica.
        """
        self.LF_quinta_fila = ttk.LabelFrame(self.frame_rob, text="  Terminal  ")
        self.LF_quinta_fila.grid(row=4, column=0, sticky="nsew", padx=10, pady=5)
        self.LF_quinta_fila.grid_rowconfigure(0, weight=1)
        self.LF_quinta_fila.grid_columnconfigure(0, weight=1)
        self.F_terminal = ttk.Frame(self.LF_quinta_fila)
        self.F_terminal.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")
        self.F_terminal.grid_rowconfigure(0, weight=1)
        self.F_terminal.grid_columnconfigure(0, weight=1)

        self.terminal = ttk.ScrolledText(self.F_terminal, 
                                    wrap=ttk.WORD, 
                                    font=("Courier", 10),
                                    height=2,
                                    state=ttk.DISABLED)
        self.terminal.configure(
            bg="#1e1e1e",  # Fondo negro
            fg="white",  # Texto blanco
            insertbackground="white",  # Cursor blanco
        )
        self.terminal.vbar.pack_forget() 
        self.terminal.grid(row=0, column=0, sticky="nsew")
        self.terminal.tag_configure("ERROR", foreground="red")  # Estilo para errores
        self.terminal.tag_configure("INPUT", foreground="magenta")  # Estilo para advertencias
        self.terminal.tag_configure("INFO", foreground="white")
        self.terminal.tag_configure("WARN", foreground="yellow")
        self.terminal.tag_configure("SUCCESS", foreground="green")

        self._update_terminal()
        
    def _rviz_launch(self):
        """
        Inicializa RVIZ
        """
        try:
            # Lanzar RViz en un proceso separado y dar tiempo a que aparezca
            self.rviz_process = subprocess.Popen(
                    ['rviz', '-d', '/home/laboratorio/ros_workspace/src/proyecto_final/data/tkinter_img/tkinter.rviz'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            sleep(1)
            
            # Usa xwininfo para obtener la ventana de RViz
            try:
                window_inf_str = subprocess.check_output(["xwininfo","-tree","-root"]).decode('utf-8')
            except subprocess.CalledProcessError as e:
                print(f"Error al buscar ventana: {e.output.decode('utf-8')}")

            rviz_win_name = [x for x in window_inf_str.split("\n") if "- RViz" in x][0].split('"')[1]
            self.win_info = subprocess.check_output(['xdotool', 'search', '--name', rviz_win_name]).decode('utf-8').strip()

            self.root.after(1000, self.reparent_rviz, self.win_info)

        except Exception as e:
            messagebox.showerror("Error", f"No se pudo incrustar RViz: {e}")



    def reparent_rviz(self, rviz_window_id):
        """
        Ajusta la ventana de RViz para que se muestre en la interfaz gráfica.
        """
                # Tamaño de la pantalla
        window_width = self.root.winfo_screenwidth()//2
        window_height = self.root.winfo_screenheight()


        # Desplazamiento hacia la derecha (posición x empieza en la mitad de la pantalla)
        x_position = window_width
        y_position = 0  # Inicia desde la parte superior

        try:
            subprocess.run(['xdotool', 'windowsize', rviz_window_id, str(window_width), str(window_height)])
            subprocess.run(['xdotool', 'windowmove', rviz_window_id, str(x_position), str(y_position)])
            

        except Exception as e:
            messagebox.showerror("Error", f"Error al ajustar la ventana: {e}")

    def close_rviz(self):
        """
        Cierra la ventana de RViz.
        """
        try:
           
            # Envía la señal de cierre a la ventana de RViz
            subprocess.call(['xdotool', 'windowclose', self.win_info])
            
            print("Ventana RViz cerrada")
        except subprocess.CalledProcessError as e:
            print(f"Error al cerrar la ventana: {e}")
    

    def _update_images(self):
        """
        Actualiza las imágenes de las cámaras en la interfaz gráfica.
        """
        aspect_ratio = 0.55
        if not hasattr(self, "V_modo"):
            return
        if self.V_modo.get() != 1:
            aspect_ratio = 0.4
            img2 = self._create_image_with_text(f"/cam_alzado\nNO PUBLICADO", aspect_ratio)
            self.L_img_alzado.config(image=img2)
            self.L_img_alzado.image = img2

            nombre_top = "/cam_top"
            nombre_lateral = "/cam_lateral"
        else:
            nombre_top = "/cam_hand_top"
            nombre_lateral = "/cam_hand_lateral"

        # Actualizar las etiquetas en la interfaz
        img1 = self._create_image_with_text(f"{nombre_top}\nNO PUBLICADO", aspect_ratio)
        self.L_img_top.config(image=img1)
        self.L_img_top.image = img1

        img3 = self._create_image_with_text(f"{nombre_lateral}\nNO PUBLICADO", aspect_ratio)
        self.L_img_lateral.config(image=img3)
        self.L_img_lateral.image = img3


    def _create_image_with_text(self, text:str, aspect_ratio=0.6):
        """
        Crea una imagen con un texto centrado.
            @param text (str) - Texto a mostrar en la imagen
            @param aspect_ratio (float) - Relación de aspecto de la imagen
            @return (PhotoImage) - Imagen con el texto
        """
        img_size = (int(640 * aspect_ratio), int(480 * aspect_ratio))

        image = Image.new('RGB', img_size, color='black')
        
        # Crear un objeto para dibujar en la imagen
        draw = ImageDraw.Draw(image)
        
        # Establecer el texto y su color
        text_color = (255, 255, 255)  # Blanco
        
        # Intentar cargar una fuente del sistema o usar una por defecto
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/montserrat/montserrat/Montserrat-Regular.ttf", 20)  # Usa una fuente si está disponible
        except IOError:
            font = ImageFont.load_default()  
        
        # Dividir el texto en líneas
        lines = text.split("\n")
        
        # Calcular la altura total del texto (sumando las alturas de las líneas y los márgenes)
        total_text_height = sum([draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines])
        line_height = draw.textbbox((0, 0), lines[0], font=font)[3] - draw.textbbox((0, 0), lines[0], font=font)[1]  # Altura de una línea

        # Calcular la posición inicial para centrar las líneas
        y_offset = (img_size[1] - total_text_height) // 2

        # Dibujar cada línea en la imagen
        for line in lines:
            text_bbox = draw.textbbox((0, 0), line, font=font)  # Obtiene las dimensiones del texto
            text_width = text_bbox[2] - text_bbox[0]
            position = ((img_size[0] - text_width) // 2, y_offset)  # Centrar horizontalmente y ajustar verticalmente
            draw.text(position, line, font=font, fill=text_color)
            
            # Aumentar el desplazamiento vertical para la siguiente línea
            y_offset += line_height

        # Convertir la imagen a un formato compatible con tkinter
        tk_image = PhotoImage(image, master=self.root)
        
        return tk_image
    
    def update_image(self, img: np.ndarray, label:ttk.Label) -> None:
        """
        Actualiza la imagen en el widget Tkinter.
            @param img (np.ndarray) - Imagen a mostrar
            @param label (ttk.Label) - Widget donde se mostrará la imagen
        """
        # Convertir la imagen PIL a PhotoImage para Tkinter
        if label.winfo_exists():
            # Convertir de nuevo a PIL
            img = Image.fromarray(img)
            
            # Redimensionar la imagen
            if self.V_modo.get() != 1: 
                img_size = (int(img.size[0] * self.img_aspect_ratio), int(img.size[1] * self.img_aspect_ratio))
            else:
                img_size = (int(img.size[0] * 0.55), int(img.size[1] * 0.55))
            img = img.resize(img_size, Image.Resampling.LANCZOS)

            # Convertir TK
            tk_image = ImageTk.PhotoImage(img, master=self.root)

            label.config(image=tk_image)
            label.image = tk_image

    def cb_image_top(self, image:Image) -> None:
        ''' 
        Callback del subscriptor de la cámara.
            @param image (Image) - Imagen de la camara
        '''
        if not hasattr(self, "V_modo"):
            return
        if self.V_modo.get() != 1: 
            # Convertir la imagen de ROS (sensor_msgs/Image) a OpenCV
            img = deepcopy(self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough'))

            # Convertir la imagen PIL a PhotoImage para Tkinter
            if self.L_img_top.winfo_exists():
                self.root.after(0, self.update_image, img, self.L_img_top)
        
    
    def cb_image_alzado(self, image:Image) -> None:
        ''' 
        Callback del subscriptor de la cámara.
            @param image (Image) - Imagen de la camara
        '''
        if not hasattr(self, "V_modo"):
            return

        if self.V_modo.get() != 1: 
            # Convertir la imagen de ROS (sensor_msgs/Image) a OpenCV
            img = deepcopy(self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough'))

            # Convertir la imagen PIL a PhotoImage para Tkinter
            if self.L_img_alzado.winfo_exists():
                self.root.after(0, self.update_image, img, self.L_img_alzado)


    def cb_image_lateral(self, image:Image) -> None:
        ''' 
        Callback del subscriptor de la cámara.
            @param image (Image) - Imagen de la camara
        '''
        if not hasattr(self, "V_modo"):
            return
        if self.V_modo.get() != 1: 
            # Convertir la imagen de ROS (sensor_msgs/Image) a OpenCV
            img = deepcopy(self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough'))

            # Convertir la imagen PIL a PhotoImage para Tkinter
            if self.L_img_lateral.winfo_exists():
                self.root.after(0, self.update_image, img, self.L_img_lateral)


    def cb_hand_lateral(self, image:Image) -> None:
        ''' 
        Callback del subscriptor de la cámara.
            @param image (Image) - Imagen de la camara
        '''
        if self.V_modo.get() == 1: 
            # Convertir la imagen de ROS (sensor_msgs/Image) a OpenCV
            img = deepcopy(self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough'))

            # Convertir la imagen PIL a PhotoImage para Tkinter
            if self.L_img_lateral.winfo_exists():
                self.root.after(0, self.update_image, img, self.L_img_lateral)
    

    def cb_hand_top(self, image:Image) -> None:
        ''' 
        Callback del subscriptor de la cámara.
            @param image (Image) - Imagen de la camara
        '''
        if not hasattr(self, "V_modo"):
            return
        if self.V_modo.get() == 1: 
            # Convertir la imagen de ROS (sensor_msgs/Image) a OpenCV
            img = deepcopy(self.bridge.imgmsg_to_cv2(image, desired_encoding='passthrough'))

            # Convertir la imagen PIL a PhotoImage para Tkinter
            if self.L_img_top.winfo_exists():
                self.root.after(0, self.update_image, img, self.L_img_top)


    def command_detect_figure(self) -> None:
        """
        Comando para detectar la figura en la interfaz gráfica.
        """
        Thread(target=self.detect_figure_thread, daemon=True).start()


    def detect_figure_thread(self) -> None:
        """
        Hilo para detectar la figura en la interfaz gráfica.
        """
        self.B_MakeFigure.state([ttk.DISABLED])
        self.B_Calibrate.state([ttk.DISABLED])
        self.B_RemakeFigure.state([ttk.DISABLED])
        self.CB_modo.state([ttk.DISABLED])

        self.RobMain.detect_figure()

        if hasattr(self, "canvas_3d"):
            self.canvas_3d.get_tk_widget().destroy()

        fig_3d = self.Geometry3D._paint_matrix(self.RobMain.matriz3D, tkinter=True, figsize=(3,2.5))
        self.canvas_3d = FigureCanvasTkAgg(fig_3d, self.geometry1_frame)
        self.canvas_3d.get_tk_widget().grid(row=1, column=0)

        if not np.all(self.RobMain.matriz3D == -1):
            self.Logic_DetectFigure = True
            if self.Logic_Calibrate and self.Logic_DetectFigure:
                self.B_RemakeFigure.state(["!disabled"])

        self.B_MakeFigure.state(["!disabled"])
        self.B_Calibrate.state(["!disabled"])
        self.CB_modo.state(["!disabled"])


    def command_calibrate_aruco(self) -> None:
        """
        Comando para calibrar el ArUco en la interfaz gráfica.
        """
        Thread(target=self.calibrate_aruco_thread, daemon=True).start()


    def calibrate_aruco_thread(self) -> None:
        """
        Hilo para calibrar el ArUco en la interfaz gráfica.
        """
        # --- DESABILITAR BOTONES ---
        self.B_MakeFigure.state([ttk.DISABLED])
        self.B_RemakeFigure.state([ttk.DISABLED])
        self.B_Calibrate.state([ttk.DISABLED])
        self.CB_modo.state([ttk.DISABLED])

        # --- REALIZAR ACCIÓN ---
        if self.Contador_Calibrate == 0:
            self.RobMain._move_to_calibration()
        
        elif self.Contador_Calibrate == 1:
            self.RobMain._put_calibration_pin()
        
        elif self.Contador_Calibrate == 2:
            self.RobMain._leave_calibration_space()

        else:
            self.RobMain._drop_calibration_pin()  # Abre nuevamente la pinza.

        # --- DISEÑO Y HABILITAR BOTONES ---
        if self.Contador_Calibrate == 3:
            self.B_Calibrate.config(bootstyle="primary", text="CALIBRAR ARUCO")
            self.Contador_Calibrate = 0
            self.Logic_Calibrate = True

            self.B_MakeFigure.state(["!disabled"])
            if self.Logic_Calibrate and self.Logic_DetectFigure:
                self.B_RemakeFigure.state(["!disabled"])
        else:
            self.B_Calibrate.config(bootstyle="warning", text="Siguiente")
            self.Contador_Calibrate += 1

        self.CB_modo.state(["!disabled"])
        self.B_Calibrate.state(["!disabled"])


    def command_hand_control(self) -> None:
        """
        Comando para activar/desactivar el control por mano en la interfaz gráfica.
        """
        if self.V_HandControl.get() == 1:
            if self.RobMain._hand_control_active():
                self.B_HandControl.config(bootstyle="toolbutton-danger", text="PARAR CONTROL POR MANO")
                self._update_hand_status()
                self.feedback_thread = Thread(target=self.RobMain._control_robot_by_hand)
                self.feedback_thread.start()
        else:
            self.RobMain.hand_control = False
            self.feedback_thread.join()
            self.RobMain._hand_control_deactivate()
            self.B_HandControl.config(bootstyle="toolbutton-primary", text="EMPEZAR CONTROL POR MANO")
            self.cv_img = []

    def command_remake_figure(self) -> None:
        """
        Comando para reconstruir la figura en la interfaz gráfica.
        """
        Thread(target=self.remake_figure_thread, daemon=True).start()

    def remake_figure_thread(self) -> None:
        """
        Hilo para reconstruir la figura en la interfaz gráfica.
        """
        self.CB_modo.state([ttk.DISABLED])
        self.RobMain.track_cubes()

        if hasattr(self, "canvas_2d"):
            self.canvas_2d.get_tk_widget().destroy()
        fig_2d = self.Geometry2D.draw_2d_space(self.RobMain.cubes, tkinter=True,figsize=(4,3))
        self.canvas_2d = FigureCanvasTkAgg(fig_2d, self.geometry2_frame)
        self.canvas_2d.get_tk_widget().grid(row=1, column=0)

        sleep(0.5)

        if self.RobMain._test_figure():
            self.RobMain.create_figure()
        self.CB_modo.state(["!disabled"])

    def _update_hand_status(self) -> None:
        """
        Actualiza el estado de la mano en la interfaz gráfica.
        """
        path = "/home/laboratorio/ros_workspace/src/proyecto_final/data/tkinter_img/"
        if self.RobMain.hand_detected:
            if self.RobMain.hand_gesture['is_dino']:
                gesture = "is_dino"
            elif self.RobMain.hand_gesture['is_peace']:
                gesture = "is_peace"
            elif self.RobMain.hand_gesture['is_dislike']:
                gesture = "is_dislike"
            elif self.RobMain.hand_gesture['is_open']:
                gesture = "is_open"
            else:
                gesture = "is_not_open"
        else:
            gesture = "is_not_detected"

        img_path = f"{path}{gesture}.png"
        img_gesture = cv2.cvtColor(cv2.imread(img_path,cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
        img_gesture = Image.fromarray(img_gesture)
            
        # Redimensionar la imagen
        img_size = (int(img_gesture.size[0] * self.img_aspect_ratio), int(img_gesture.size[0] * self.img_aspect_ratio))
        img_gesture = img_gesture.resize(img_size, Image.Resampling.LANCZOS)

        # Convertir TK
        tk_image = ImageTk.PhotoImage(img_gesture, master=self.root)
        self.L_img_mano.config(image=tk_image)
        self.L_img_mano.image = tk_image

        self.scatter.set_sizes([100])
        self.scatter._offsets3d = (
            [self.RobMain.hand_pose[0]],
            [self.RobMain.hand_pose[1]],
            [self.RobMain.hand_pose[2]],
        )

        # Redibujar el canvas para reflejar los cambios
        self.canvas_3d.draw()

        if self.V_HandControl.get() == 1:
            self.root.after(100, self._update_hand_status)
        else:
            self.L_img_mano.config(image="")
            self.L_img_mano.image = None


    def setup_figure(self) -> None:
        """
        Configura la figura inicial y el scatter point.
        """
        self.ax.set_xlim(-300, 300)
        self.ax.set_ylim(-300, 300)
        self.ax.set_zlim(-300, 300)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.scatter = self.ax.scatter(
            self.RobMain.hand_pose[0],
            self.RobMain.hand_pose[1],
            self.RobMain.hand_pose[2],
            color="red",
            s=0,
        )
        self.ax.legend()

    def _update_terminal(self) -> None:
        """
        Actualiza el terminal en la interfaz gráfica.
        """
        if self.RobMain.message is not None and self.RobMain.message_type is not None:
            self.terminal.config(state=ttk.NORMAL)
            self.terminal.insert(ttk.END, f"{self.RobMain.message}\n", self.RobMain.message_type)
            self.terminal.yview(ttk.END)
            self.terminal.config(state=ttk.DISABLED)
            self.RobMain.message = None
            self.RobMain.message_type = None

        self.root.after(49, self._update_terminal)
        

    def estilo(self) -> None:
        """
        Establece el estilo de la interfaz gráfica.
        """
        style = ttk.Style()
        style.configure("Custom.TNotebook.Tab",
            font= ("Montserrat Medium", 12), 
            background= "white",
            )

        style.configure("Custom.TNotebook",
            background= "white",
            )

        style.configure("Custom.TFrame",
            font= ("Montserrat Medium", 12), 
            background= "white",
            borderwidth=0,  # Grosor del borde
            )

        style.map(
            "Custom.TNotebook.Tab",
            background=[("selected", "#8C85F7"), ("active", "#8C85F7"), ("!selected", "white")],     # Background color when hovered
            foreground=[("selected", "black"), ("active", "black"), ("!selected", "black")],   # Text color when hovered
        )

        style.configure("Vision.TNotebook.Tab",
            font= ("Montserrat Medium", 10), 
            background= "white",
            borderwidth=0,  # Grosor del borde
            )

        style.configure("Vision.TNotebook",
            background= "white",
            borderwidth=0
            )

        style.configure("Custom.TFrame",
            font= ("Montserrat Medium", 12), 
            background= "white",
            borderwidth=0,  # Grosor del borde
            )

        style.map(
            "Vision.TNotebook.Tab",
            background=[("selected", "#C5C1F7"), ("active", "#C5C1F7"), ("!selected", "white")],     # Background color when hovered
            foreground=[("selected", "black"), ("active", "black"), ("!selected", "black")],   # Text color when hovered
        )

        style.configure(
        "TCheckbutton",  # Nombre del estilo
        font=("Montserrat", 10),  # Fuente personalizada
        )

        style.configure(
        "TButton",  # Nombre del estilo
        font=("Montserrat", 10),  # Fuente personalizada
        )

        style.configure(
        "TLabelframe.Label",  # Nombre del estilo
            font=("Montserrat", 10),
        )
    
    def _on_closing(self) -> None:
        """Esta función se ejecuta cuando la ventana se cierra."""
        self.stop_robot_tab()
        
        # Cerrar la ventana de Tkinter
        self.root.destroy()
        exit()
        
if __name__ == "__main__":
    RoboticaTkinter()