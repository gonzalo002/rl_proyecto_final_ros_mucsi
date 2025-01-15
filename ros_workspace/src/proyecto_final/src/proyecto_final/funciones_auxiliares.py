def crear_mensaje(mensaje:str, tipo:str='ERROR', nombre_clase:str="", tkinter:bool=True) -> str:
    """
    Función que crea un mensaje con colores para mostrar en consola.
        @param mensaje: Mensaje a mostrar.
        @param tipo: Tipo de mensaje. Puede ser ERROR, SUCCESS, WARN, INFO, INPUT.
        @param nombre_clase: Nombre de la clase que llama a la función.
        @param tkinter: Indica si se está ejecutando en un entorno de tkinter.
        @return: Mensaje con colores.
    """
    # --- CÓDIGO COLORES ---
    c = {
        "ERROR":   "\033[31m",  # Rojo
        "SUCCESS": "\033[32m",  # Verde
        "WARN":    "\033[33m",  # Amarillo
        "INFO":    "\033[0m" ,  # Restablecer
        "RESET":   "\033[0m" ,  # Restablecer
        "INPUT":   "\033[35m",  # Magenta
    }

    msg = f"{c[tipo]}[{tipo.rjust(7)}] [{nombre_clase.rjust(20)}]: {mensaje} {c['RESET']}"
    if tipo == "INPUT" and not tkinter:
        return input(msg)
    else:
        print(msg)
    
    return msg