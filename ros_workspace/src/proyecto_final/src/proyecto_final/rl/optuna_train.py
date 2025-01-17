from proyecto_final.rl.env_rob_train_discrete import ROSEnv

import os
import optuna
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
import datetime

class OptunaTrain:
        def __init__(self, n_cubos_max: int = 8):
            self.abs_path = '/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:os.path.dirname(os.path.abspath(__file__)).split('/').index('proyecto_final') + 1])
            self.env = ROSEnv(num_cubos_max=n_cubos_max, verbose=True)
            self.env.reset()
            self.model = None

            self.n_cubos_max = n_cubos_max
            self.timesteps = 1_000
            

        def optuna_train(self, model:str = 'PPO', n_trials:int = 100) -> None:
            try:
                os.mkdir(f'{self.abs_path}/data/rl/optuna')
                os.mkdir(f'{self.abs_path}/data/rl/optuna/best_model')
                os.mkdir(f'{self.abs_path}/data/rl/optuna/logs')
                os.mkdir(f'{self.abs_path}/data/rl/optuna/html')
                os.mkdir(f'{self.abs_path}/data/rl/optuna/db')
            except:
                pass


            today = datetime.datetime.now()
            date = f'{today.year}_{today.month}_{today.day}_{today.hour}_{today.minute}'

            study_name = f'optuna_study_{model}_rosenv_{date}_cubos_{self.n_cubos_max}'
            storage_path = f'sqlite:///{self.abs_path}/data/rl/optuna/db/{model}_optuna.db'

            study = optuna.create_study(study_name=study_name, direction="maximize", storage=storage_path)

            if model == 'PPO':
                # Ejecutar el estudio de Optuna
                study.optimize(self.ppo_optuna, n_trials=n_trials)

            elif model == 'DQN':
                # Ejecutar el estudio de Optuna
                study.optimize(self.dqn_optuna, n_trials=n_trials)

            # Generate the improtant figures of the results
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_html(f"{self.abs_path}/data/rl/optuna/html/{model}_optimization_history.html")
            fig = optuna.visualization.plot_contour(study)
            fig.write_html(f"{self.abs_path}/data/rl/optuna/html/{model}_contour.html")
            fig = optuna.visualization.plot_slice(study)
            fig.write_html(f"{self.abs_path}/data/rl/optuna/html/{model}_slice.html")
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_html(f"{self.abs_path}/data/rl/optuna/html/{model}_param_importances.html")

        def dqn_optuna(self, trial:optuna.Trial):
            """
            Define los hiperparámetros que quieres optimizar
            """
            learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
            learning_starts = trial.suggest_int("learning_starts", 100, 1000)
            gamma = trial.suggest_uniform("gamma", 0.95, 0.999)
            tau = trial.suggest_uniform("tau", 0.001, 0.1)
            train_freq = trial.suggest_int("train_freq", 1, 10)
            gradient_steps = trial.suggest_int("gradient_steps", 1, 10)
            exploration_fraction = trial.suggest_uniform("exploration_fraction", 0.1, 0.5)
            exploration_initial_eps = trial.suggest_uniform("exploration_initial_eps", 0.1, 1.0)
            max_grad_norm = trial.suggest_uniform("max_grad_norm", 0.1, 1.0)

            """
            Configuración del modelo DQN
            """
            self.model = DQN("MlpPolicy", self.env,
                        learning_rate=learning_rate,
                        buffer_size=500_000,
                        learning_starts=learning_starts,
                        batch_size=64,
                        gamma=gamma,
                        tau=tau,
                        train_freq=train_freq,
                        gradient_steps=gradient_steps,
                        target_update_interval=500,
                        exploration_fraction=exploration_fraction,
                        exploration_initial_eps=exploration_initial_eps,
                        exploration_final_eps=0.02,
                        max_grad_norm=max_grad_norm,
                        verbose=0)

            """
            Eval callback para monitorear el rendimiento
            """
            eval_callback = EvalCallback(self.env, best_model_save_path=f'{self.abs_path}/data/rl/optuna/best_model/', log_path=f'{self.abs_path}/data/rl/optuna/logs/',
                                         eval_freq=1000, deterministic=True, render=False)

            today = datetime.datetime.now()
            date = f'{today.year}_{today.month}_{today.day}_{today.hour}_{today.minute}'

            # Entrenamiento
            tb_log_name = f'DQN_{date}_cubes_{self.n_cubos_max}'
            self.model.learn(total_timesteps=self.timesteps, callback=eval_callback, tb_log_name=tb_log_name, log_interval=400)

            # Evaluación del rendimiento
            mean_reward, _ = evaluate_policy(self.model, self.env, n_eval_episodes=10, deterministic=True)

            return mean_reward
        

        def ppo_optuna(self, trial:optuna.Trial):
            """
            Define los hiperparámetros que quieres optimizar
            """
            learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-3)
            n_steps = trial.suggest_int("n_steps", 512, 1024)
            n_epochs = trial.suggest_int("n_epochs", 1, 10)
            gamma = trial.suggest_uniform("gamma", 0.9, 0.999)
            gae_lambda = trial.suggest_uniform("gae_lambda", 0.8, 0.999)
            ent_coef = trial.suggest_loguniform("ent_coef", 1e-8, 1e-1)
            max_grad_norm = trial.suggest_uniform("max_grad_norm", 0.1, 1.0)

            """
            Configuración del modelo PPO
            """
            self.model = PPO("MlpPolicy", self.env,
                        learning_rate=learning_rate,
                        n_steps=n_steps,
                        batch_size=64,
                        n_epochs=n_epochs,
                        gamma=gamma,
                        gae_lambda=gae_lambda,
                        clip_range=0.2,
                        ent_coef=ent_coef,
                        vf_coef=0.5,
                        max_grad_norm=max_grad_norm,
                        verbose=0)

            """
            Eval callback para monitorear el rendimiento
            """
            eval_callback = EvalCallback(self.env, best_model_save_path=f'{self.abs_path}/data/rl/optuna/best_model/', log_path=f'{self.abs_path}/data/rl/optuna/logs/',
                                        eval_freq=1000, deterministic=True, render=False)

            today = datetime.datetime.now()
            date = f'{today.year}_{today.month}_{today.day}_{today.hour}_{today.minute}'

            # Entrenamiento
            tb_log_name = f'PPO_{date}_cubes_{self.n_cubos_max}'
            self.model.learn(total_timesteps=2, callback=eval_callback, tb_log_name=tb_log_name, log_interval=400)
            
            # Evaluación del rendimiento
            mean_reward, _ = evaluate_policy(self.model, self.env, n_eval_episodes=10, deterministic=True)
            
            return mean_reward

if __name__ == '__main__':
    optuna_train = OptunaTrain()
    optuna_train.optuna_train(model='DQN')