from env_rob_train import ROSEnv
import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import datetime
from time import time
import yaml
from stable_baselines3.common.evaluation import evaluate_policy

def train(n_cubos_max: int = 2, verbose:bool = True):
    file_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:os.path.dirname(os.path.abspath(__file__)).split('/').index('proyecto_final')+1])

    try:
        os.mkdir(f'{file_path}/data/rl')
        os.mkdir(f'{file_path}/data/rl/agentes_entrenados')
        os.mkdir(f'{file_path}/data/rl/tb_logs')
        os.mkdir(f'{file_path}/data/rl/yaml_logs')
        os.mkdir(f'{file_path}/data/rl/agentes_entrenados')
    except:
        pass

    env = ROSEnv(num_cubos_max=n_cubos_max,verbose=verbose)
    env.reset()

    default_parameters = {'policy': 'MlpPolicy',
                            'env' : env,
                            'learning_rate': 3e-4,
                            'n_steps' : 2048,
                            'batch_size' : 64,
                            'n_epochs' : 10,
                            'gamma':  0.99,
                            'gae_lambda':  0.95,
                            'clip_range': 0.2,
                            'clip_range_vf' : None,
                            'normalize_advantage': True,
                            'ent_coef': 0.0,
                            'vf_coef': 0.5,
                            'max_grad_norm': 0.5,
                            'use_sde': False,
                            'sde_sample_freq': -1,
                            'target_kl': None,
                            'tensorboard_log': f'{file_path}/data/rl/tb_logs',
                            'policy_kwargs': None,
                            'verbose': 1,
                            'seed': None,
                            'device': "auto"
                            }

    parameters = {'policy': 'MlpPolicy',
                    'env' : env,
                    'learning_rate': 3e-4,
                    'n_steps' : 256,
                    'batch_size' : 32,
                    'n_epochs' : 10,
                    'gamma':  0.99,
                    'gae_lambda':  0.95,
                    'clip_range': 0.2,
                    'clip_range_vf' : None,
                    'normalize_advantage': True,
                    'ent_coef': 0.005,
                    'vf_coef': 0.5,
                    'max_grad_norm': 0.5,
                    'use_sde': False,
                    'sde_sample_freq': -1,
                    'target_kl': None,
                    'tensorboard_log': f'{file_path}/data/rl/tb_logs',
                    'policy_kwargs': {
                                    'net_arch': [128, 128],  # hidden_units = 128, num_layers = 2
                                    },
                    'verbose': 1,
                    'seed': None,
                    'device': "auto"
                    }


    model = PPO(**parameters)
    model.__module__
    
    today = datetime.datetime.now()
    date = f'{today.year}_{today.month}_{today.day}_{today.hour}_{today.minute}'
    tb_log_name = f'PPO_{date}_cubes_{n_cubos_max}'
    total_timesteps = 100000

    # Callback para guardar el modelo cada 20,000 pasos
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=f'{file_path}/data/rl/agentes_entrenados',
        name_prefix=f'ppo_rosenv_{date}_cubes_{n_cubos_max}'
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
    
    # mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

    model_path = f'{file_path}/data/rl/agentes_entrenados/ppo_rosenv_{date}_cubes_{n_cubos_max}'

    parameters.pop('env')

    # Crear una estructura jerárquica para las secciones
    save_in_yaml = {'Agente' : {
        'Agent_Info': {
            'Agent_name': model_path,
            'Log_name': tb_log_name,
            'date': date},
        'Train_Info' : {
            'model' : model.__module__,
            'total_timesteps': total_timesteps,
            'train_time': end_time - ini_time,
            # 'mean_reward': float(mean_reward),
            # 'std_reward': float(std_reward),
            'cubos': n_cubos_max
        },
        # 'Success_Rates' : info,
        'Agent_Parameters': parameters,
        'Comentario' : 'Introduce Comentario'
    }}

    model.save(model_path)
    with open(f'{file_path}/data/rl/yaml_logs/logs_agentes', '+a') as f:
            yaml.dump(save_in_yaml, f, default_flow_style=False, sort_keys=False)

def retrain(model_name:str, n_cubos_max:int = 2, verbose:bool = True):
    """
    Función para reentrenar un modelo previamente entrenado
        @param model_path: str - Ruta del modelo
        @param n_cubos_max: int - Número máximo de cubos
        @param verbose: bool - Mostrar información del entorno
    """
    file_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:os.path.dirname(os.path.abspath(__file__)).split('/').index('proyecto_final')+1])

    model_path = f'{file_path}/data/rl/agentes_entrenados/{model_name}'
    model = PPO.load(model_path, env=ROSEnv(n_cubos_max, verbose=verbose))

    today = datetime.datetime.now()
    date = f'{today.year}_{today.month}_{today.day}_{today.hour}_{today.minute}'

    tb_log_name = f'PPO_{date}_cubes_{n_cubos_max}'

    total_timesteps = 1000000

    # Callback para guardar el modelo cada 20,000 pasos
    checkpoint_callback = CheckpointCallback(
        save_freq=100000, 
        save_path=f'{file_path}/data/rl/agentes_entrenados',
        name_prefix=f'ppo_rosenv_{date}_cubes_{n_cubos_max}'
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

    model_path = f'{file_path}/data/rl/agentes_entrenados/ppo_rosenv_{date}_cubes_{n_cubos_max}'
    model.save(model_path)

    # Crear una estructura jerárquica para las secciones
    save_in_yaml = {'Agente' : {
        'Agent_Info': {
            'Agent_name': model_path,
            'Log_name': tb_log_name,
            'date': date},
        'Train_Info' : {
            'model' : model.__module__,
            'total_timesteps': total_timesteps,
            'train_time': end_time - ini_time,
            'cubos': n_cubos_max
        },
        'Previous_Agent': model_name,
        'Comentario' : 'Introduce Comentario'
    }}

    with open(f'{file_path}/data/rl/yaml_logs/logs_agentes', '+a') as f:
        yaml.dump(save_in_yaml, f, default_flow_style=False, sort_keys=False)


def test(n_cubos_max:int = 2, seed:int = None):
    env = ROSEnv(n_cubos_max)

    model = PPO.load('src/proyecto_final/scripts/rl/agentes_entrenados/ppo_rosenv_2024_12_8_23_38_cubes_2.zip')

    while True:
        obs, _ = env.reset(seed=seed)
        action, _ = model.predict(obs)
        _, _, _, _ = env.step(action)

        choice = input('Continuar? (q para salir)')
        if choice == 'q':
            break

if __name__ == '__main__':
    n_cubos_max = 4
    training = True
    re_train = False
    if training:   
        if re_train:
            retrain('ppo_rosenv_2025_1_15_9_55_cubes_10_140000_steps.zip', n_cubos_max=n_cubos_max)
        else:
            train(n_cubos_max=n_cubos_max)
    else:
        test(n_cubos_max=n_cubos_max, seed=2)