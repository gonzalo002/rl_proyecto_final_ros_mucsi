from env_rob_train import ROSEnv
import os
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
import datetime
from time import time
import yaml
from stable_baselines3.common.evaluation import evaluate_policy

def train(n_cubos_max: int = 2, verbose: bool = True):
    file_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:os.path.dirname(os.path.abspath(__file__)).split('/').index('proyecto_final') + 1])

    try:
        os.mkdir(f'{file_path}/data/rl')
        os.mkdir(f'{file_path}/data/rl/agentes_entrenados')
        os.mkdir(f'{file_path}/data/rl/tb_logs')
        os.mkdir(f'{file_path}/data/rl/yaml_logs')
        os.mkdir(f'{file_path}/data/rl/agentes_entrenados')
    except:
        pass

    env = ROSEnv(num_cubos_max=n_cubos_max, verbose=verbose)
    env.reset()

    # Parámetros específicos de DQN
    parameters = {
        'policy': 'MlpPolicy',
        'env': env,
        'learning_rate': 1e-3,  # Aprendizaje más bajo comparado con PPO
        'buffer_size': 500000,  # Tamaño de la experiencia de repetición
        'learning_starts': 100,  # Número de pasos antes de empezar el entrenamiento
        'batch_size': 64,  # Tamaño de batch para la actualización
        'tau': 1.0,  # Factor de suavizado para la actualización de la red objetivo
        'gamma': 0.99,  # Factor de descuento
        'train_freq': 4,  # Frecuencia de entrenamiento
        'gradient_steps': 1,  # Número de pasos de optimización por iteración
        'replay_buffer_class': None,  # Clase de buffer de repetición
        'optimize_memory_usage': False,  # Optimizar el uso de memoria
        'target_update_interval': 500,  # Cada cuantos pasos se actualiza la red objetivo
        'exploration_fraction': 0.1,  # Fracción de la exploración
        'exploration_initial_eps': 1.0,  # Epsilon inicial de la política de exploración
        'exploration_final_eps': 0.02,  # Epsilon final de la política de exploración
        'max_grad_norm': 10,  # Norma máxima del gradiente
        'tensorboard_log': f'{file_path}/data/rl/tb_logs',
        'policy_kwargs': None,
        'verbose': 1,
        'seed': None,
        'device': "cpu"
    }

    model = DQN(**parameters)
    
    today = datetime.datetime.now()
    date = f'{today.year}_{today.month}_{today.day}_{today.hour}_{today.minute}'
    tb_log_name = f'DQN_{date}_cubes_{n_cubos_max}'
    total_timesteps = 500000

    # Callback para guardar el modelo cada 100,000 pasos
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=f'{file_path}/data/rl/agentes_entrenados',
        name_prefix=f'dqn_rosenv_{date}_cubes_{n_cubos_max}'
    )

    ini_time = time()

    # Entrenar el modelo
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=1000,
        tb_log_name=tb_log_name,
        callback=checkpoint_callback
    )

    end_time = time()

    model_path = f'{file_path}/data/rl/agentes_entrenados/dqn_rosenv_{date}_cubes_{n_cubos_max}'

    parameters.pop('env')

    # Crear una estructura jerárquica para las secciones
    save_in_yaml = {'Agente': {
        'Agent_Info': {
            'Agent_name': model_path,
            'Log_name': tb_log_name,
            'date': date},
        'Train_Info': {
            'model': model.__module__,
            'total_timesteps': total_timesteps,
            'train_time': end_time - ini_time,
            'cubos': n_cubos_max
        },
        'Agent_Parameters': parameters,
        'Comentario': 'Introduce Comentario'
    }}

    model.save(model_path)
    with open(f'{file_path}/data/rl/yaml_logs/logs_agentes', '+a') as f:
        yaml.dump(save_in_yaml, f, default_flow_style=False, sort_keys=False)


def retrain(model_name: str, n_cubos_max: int = 2, verbose: bool = True):
    """
    Función para reentrenar un modelo previamente entrenado
        @param model_name: str - Ruta del modelo
        @param n_cubos_max: int - Número máximo de cubos
        @param verbose: bool - Mostrar información del entorno
    """
    file_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:os.path.dirname(os.path.abspath(__file__)).split('/').index('proyecto_final') + 1])

    model_path = f'{file_path}/data/rl/agentes_entrenados/{model_name}'
    model = DQN.load(model_path,
                    env=ROSEnv(n_cubos_max, verbose=verbose),
                    tensorboard_log=f'{file_path}/data/rl/tb_logs',
                    learning_starts=40,
                    exploration_initial_eps=0.2
                    )

    today = datetime.datetime.now()
    date = f'{today.year}_{today.month}_{today.day}_{today.hour}_{today.minute}'

    tb_log_name = f'DQN_{date}_cubes_{n_cubos_max}'

    total_timesteps = 100000

    # Callback para guardar el modelo cada 100,000 pasos
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=f'{file_path}/data/rl/agentes_entrenados',
        name_prefix=f'dqn_rosenv_{date}_cubes_{n_cubos_max}'
    )

    ini_time = time()

    # Entrenar el modelo
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=1000,
        tb_log_name=tb_log_name,
        callback=checkpoint_callback
    )

    end_time = time()

    model_path = f'{file_path}/data/rl/agentes_entrenados/dqn_rosenv_{date}_cubes_{n_cubos_max}'
    model.save(model_path)

    # Crear una estructura jerárquica para las secciones
    save_in_yaml = {'Agente': {
        'Agent_Info': {
            'Agent_name': model_path,
            'Log_name': tb_log_name,
            'date': date},
        'Train_Info': {
            'model': model.__module__,
            'total_timesteps': total_timesteps,
            'train_time': end_time - ini_time,
            'cubos': n_cubos_max
        },
        'Previous_Agent': model_name,
        'Comentario': 'Introduce Comentario'
    }}

    with open(f'{file_path}/data/rl/yaml_logs/logs_agentes', '+a') as f:
        yaml.dump(save_in_yaml, f, default_flow_style=False, sort_keys=False)


def test(n_cubos_max: int = 2, seed: int = None):
    env = ROSEnv(n_cubos_max)

    model = DQN.load('src/proyecto_final/scripts/rl/agentes_entrenados/dqn_rosenv_2024_12_8_23_38_cubes_2.zip')

    while True:
        obs, _ = env.reset(seed=seed)
        action, _ = model.predict(obs)
        _, _, _, _ = env.step(action)

        choice = input('Continuar? (q para salir)')
        if choice == 'q':
            break

if __name__ == '__main__':
    n_cubos_max = 6
    training = True
    re_train = False
    if training:
        if re_train:
            retrain('dqn_rosenv_2025_1_16_11_8_cubes_8.zip', n_cubos_max=n_cubos_max)
        else:
            train(n_cubos_max=n_cubos_max)
    else:
        test(n_cubos_max=n_cubos_max, seed=2)
