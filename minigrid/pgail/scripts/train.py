import argparse
import time
import datetime
import sys, os
 
from pgail import PGAILAlgo
 
import tensorboardX
import sys

import utils
from utils import device
from model import ACModel, DiscModel
import git
import pickle

# Parse arguments

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument("--algo", default='ppo',
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--demonstration", default = None,
                    help="location of the demonstration (default: {ENV}_{ALGO}_{COMMIT_ID})")
parser.add_argument("--number-demonstration", type=int, default=-1,
                    help="number of demonstrations")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{COMMIT_ID})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")
parser.add_argument("--no-cuda", action="store_true", default=False,
                    help="disable cuda")
parser.add_argument("--entropy", action="store_true", default=False,
                    help="disable cuda")
# Parameters for main algorithm
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--pair-coef", type=float, default=1.e-3,
                    help="clipping epsilon for pair (default: 1.e-3)")                    
parser.add_argument("--ac-recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--disc-recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

if __name__ == "__main__":
    args = parser.parse_args()

    args.ac_mem = args.ac_recurrence > 1
    args.disc_mem = args.disc_recurrence > 1

    # Set run dir
    commit_id = git.Repo(search_parent_directories=True).head.object.hexsha
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{commit_id}" 

    
    model_name = (args.model or default_model_name)

    ac_model_name = model_name + '_ac'
    print(ac_model_name)
    
    disc_model_name = model_name + '_disc'
    print(disc_model_name)

    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    tb_writer = tensorboardX.SummaryWriter(model_dir)

    # Log command and all script arguments

    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device
    if args.no_cuda:
        device = 'cpu'
    txt_logger.info(f"Device: {device}\n")
    
    # Load environments
    protagonist_envs = []
    for i in range(args.procs):
        protagonist_envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    antagonist_envs = []
    for i in range(args.procs):
        antagonist_envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    txt_logger.info("Environments loaded\n")


   

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"protagonist_num_frames": 0, "antagonist_num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(protagonist_envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")


    # Load model

    protagonist_acmodel = ACModel(obs_space, protagonist_envs[0].action_space, args.ac_mem, args.text)
    if args.model and "protagonist_ac_model_state" in status:
        protagonist_acmodel.load_state_dict(status["protagonist_model_state"])
        txt_logger.info("protagonist AC Model loaded\n")
    protagonist_acmodel.to(device)
    txt_logger.info("{}\n".format(protagonist_acmodel))

    antagonist_acmodel = ACModel(obs_space, antagonist_envs[0].action_space, args.ac_mem, args.text)
    if args.model and "antagonist_ac_model_state" in status:
        antagonist_acmodel.load_state_dict(status["antagonist_model_state"])
        txt_logger.info("antagonist AC Model loaded\n")
    antagonist_acmodel.to(device)
    
    txt_logger.info("{}\n".format(antagonist_acmodel))

    discmodel = DiscModel(obs_space, antagonist_envs[0].action_space, args.disc_mem, args.text)
    if "disc_model_state" in status:
        discmodel.load_state_dict(status["disc_model_state"])
        txt_logger.info("Disc Model loaded\n")
    discmodel.to(device)
    txt_logger.info("{}\n".format(discmodel))

    # Load algo
    
    algo = PGAILAlgo(protagonist_envs, antagonist_envs, protagonist_acmodel, antagonist_acmodel, discmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                            args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.ac_recurrence, args.disc_recurrence,
                            args.optim_eps, args.clip_eps, args.epochs, args.batch_size, pair_coef = args.pair_coef, preprocess_obss=preprocess_obss, entropy_reward = args.entropy)
    
    if "protagonist_optimizer_state" in status:
        algo.protagonist_optimizer.load_state_dict(status["protagonist_ac_optimizer_state"])
    if "antagonist_optimizer_state" in status:
        algo.antagonist_optimizer.load_state_dict(status["antagonist_ac_optimizer_state"])
    if "disc_optimizer_state" in status:
        algo.disc_optimizer.load_state_dict(status["disc_optimizer_state"])    
 
    txt_logger.info("Optimizer loaded\n")

    # Load demonstration

    
    if args.demonstration is None:
        args.demonstration = os.path.join(os.path.dirname(os.getcwd()), f'expert_demo/exp_demo_{args.env}.p')
    fp = open(args.demonstration, 'rb')
    demos = algo.init_demonstrations(pickle.load(fp))
    fp.close()
    txt_logger.info(f"Demonstrations loaded: {len(demos)} frames\n")
    
    # Train model

    protagonist_num_frames = status["protagonist_num_frames"]
    print(protagonist_num_frames)

    antagonist_num_frames = status["antagonist_num_frames"]
    print(antagonist_num_frames)
 


    update = status["update"]
    start_time = time.time()
    """
    while antagonist_num_frames < args.frames:
        # Update model parameters
        update_start_time = time.time()
        protagonist_exps, logs1 = algo.collect_protagonist_experiences()
        antagonist_exps, logs2 = algo.collect_antagonist_experiences()
        logs3, logs4 = algo.update_ac_parameters(protagonist_exps, antagonist_exps)
        logs = {**logs1, **logs2, **logs3, **logs4}
         
        update_end_time = time.time()

        antagonist_num_frames += logs["antagonist_num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["antagonist_num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["antagonist_return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["antagonist_reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["antagonist_num_frames_per_episode"])

            header = ["update", "antagonist_frames", "FPS", "duration"]
            data = [update, antagonist_num_frames, fps, duration]
            header += ["antagonist_rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["antagonist_num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["antagonist_entropy", "antagonist_value", "antagonist_policy_loss", "antagonist_value_loss", "antagonist_ac_grad_norm"]
            data += [logs["antagonist_entropy"], logs["antagonist_value"], logs["antagonist_policy_loss"], logs["antagonist_value_loss"], logs["antagonist_ac_grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))
 
            if status["antagonist_num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()
 

        # Save status
    """
    while protagonist_num_frames < args.frames and antagonist_num_frames < args.frames:
        # Update model parameters
        update_start_time = time.time()
        
        protagonist_exps, logs1 = algo.collect_protagonist_experiences()
        antagonist_exps, logs2 = algo.collect_antagonist_experiences()
 
        demos = algo.collect_demonstrations(demos)
        logs3, logs4 = algo.update_ac_parameters(protagonist_exps, antagonist_exps)
        logs5 = algo.update_disc_parameters(protagonist_exps, antagonist_exps, demos)
        logs = {**logs1, **logs2, **logs3, **logs4, **logs5}
        
        update_end_time = time.time()

        protagonist_num_frames += logs["protagonist_num_frames"]
        antagonist_num_frames += logs["antagonist_num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            protagonist_fps = logs["protagonist_num_frames"] / (update_end_time - update_start_time)
            antagonist_fps = logs["antagonist_num_frames"] / (update_end_time - update_start_time)
            
            duration = int(time.time() - start_time)
            protagonist_return_per_episode = utils.synthesize(logs["protagonist_return_per_episode"])
            protagonist_rreturn_per_episode = utils.synthesize(logs["protagonist_reshaped_return_per_episode"])
            protagonist_num_frames_per_episode = utils.synthesize(logs["protagonist_num_frames_per_episode"])


            antagonist_return_per_episode = utils.synthesize(logs["antagonist_return_per_episode"])
            antagonist_rreturn_per_episode = utils.synthesize(logs["antagonist_reshaped_return_per_episode"])
            antagonist_num_frames_per_episode = utils.synthesize(logs["antagonist_num_frames_per_episode"])


            header = ["update", "protagonist_num_frames", "protagonist_FPS", "antagonist_num_frames", "antagonist_FPS", "duration"]
            data = [update, protagonist_num_frames, protagonist_fps, antagonist_num_frames, antagonist_fps,duration]
            header += ["protagonist_rreturn_" + key for key in protagonist_rreturn_per_episode.keys()]
            data += protagonist_rreturn_per_episode.values()
            header += ["antagonist_rreturn_" + key for key in antagonist_rreturn_per_episode.keys()]
            data += antagonist_rreturn_per_episode.values()
            
            header += ["protagonist_num_frames_" + key for key in protagonist_num_frames_per_episode.keys()]
            data += protagonist_num_frames_per_episode.values()

            header += ["antagonist_num_frames_" + key for key in antagonist_num_frames_per_episode.keys()]
            data += antagonist_num_frames_per_episode.values()

            header += ["protagonist_entropy", "protagonist_value", "protagonist_policy_loss", "protagonist_value_loss", "protagonist_ac_grad_norm"]
            data += [logs["protagonist_entropy"], logs["protagonist_value"], logs["protagonist_policy_loss"], logs["protagonist_value_loss"], logs["protagonist_ac_grad_norm"]]

            header += ["antagonist_entropy", "antagonist_value", "antagonist_policy_loss", "antagonist_value_loss", "antagonist_ac_grad_norm"]
            data += [logs["antagonist_entropy"], logs["antagonist_value"], logs["antagonist_policy_loss"], logs["antagonist_value_loss"], logs["antagonist_ac_grad_norm"]]

            header += ["irl_loss", "pair_loss", "disc_grad_norm", "protagonist_exps_acc", "antagonist_exps_acc", "demos_acc"]
            data += [logs['irl_loss'], logs["pair_loss"], logs['disc_grad_norm'], logs["protagonist_exps_acc"], logs["antagonist_exps_acc"], logs["demos_acc"]]

            info = dict(zip(header, data))
            txt_logger.info(
                "U {} | pF {:06} | aF {:06} | prR:μσmM {:.2f} {:.2f} {:.2f} | arR:μσmM {:.2f} {:.2f} {:.2f} | ppL {:.3f} | pvL {:.3f} | p∇a {:.3f} | apL {:.3f} | avL {:.3f} | a∇a {:.3f} | irlL: {:3f} | pairL: {:3f} | ∇d {:.3f} | p_acc {:.3f} | a_acc {:.3f} | demos_acc {:.3f}"
                .format(info['update'], \
                    info['protagonist_num_frames'], info['antagonist_num_frames'], \
                        info["protagonist_rreturn_mean"], info["protagonist_rreturn_std"], info["protagonist_rreturn_max"], \
                            info["antagonist_rreturn_mean"], info["antagonist_rreturn_std"], info["antagonist_rreturn_max"], \
                                info["protagonist_policy_loss"], info["protagonist_value_loss"], info["protagonist_ac_grad_norm"], \
                                    info["antagonist_policy_loss"], info["antagonist_value_loss"], info["antagonist_ac_grad_norm"], \
                                        info['irl_loss'], info['pair_loss'], info['disc_grad_norm'], info["protagonist_exps_acc"], info["antagonist_exps_acc"], info["demos_acc"]
                        )
            )
 
            if status["protagonist_num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            #tb_writer.add_scalar(info['protagonist_num_frames'], info["protagonist_rreturn_mean"])
            #tb_writer.add_scalar(info['antagonist_num_frames'], info["antagonist_rreturn_mean"])
        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"protagonist_num_frames": protagonist_num_frames, \
                "antagonist_num_frames": antagonist_num_frames,\
                    "update": update,
                      "protagonist_ac_model_state": protagonist_acmodel.state_dict(), "protagonist_ac_optimizer_state": algo.protagonist_ac_optimizer.state_dict(), 
                      "antagonist_ac_model_state": antagonist_acmodel.state_dict(), "antagonist_ac_optimizer_state": algo.antagonist_ac_optimizer.state_dict(), 
                      "disc_model_state:": discmodel.state_dict(), "disc_optimizer_state:": algo.disc_optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
    