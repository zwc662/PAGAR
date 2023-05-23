import argparse
import time
import datetime
import sys, os
#python_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'torch-irl/')
#sys.path.append(python_path)
 
from gail import GAILAlgo
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
    logdir = os.path.join(os.path.dirname(__file__), 'logs')
    if not os.path.isdir(logdir):
        os.makedirs(logdir)
    tb_writer = tensorboardX.SummaryWriter(logdir)

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

    envs = []
    for i in range(args.procs):
        envs.append(utils.make_env(args.env, args.seed + 10000 * i))
    txt_logger.info("Environments loaded\n")


   

    # Load training status

    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor

    obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")


    # Load model

    acmodel = ACModel(obs_space, envs[0].action_space, args.ac_mem, args.text)
    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("AC Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))

    discmodel = DiscModel(obs_space, envs[0].action_space, args.disc_mem, args.text)
    if "model_state" in status:
        discmodel.load_state_dict(status["model_state"])
    discmodel.to(device)
    txt_logger.info("Disc Model loaded\n")
    txt_logger.info("{}\n".format(discmodel))

    # Load algo

    algo = GAILAlgo(envs, acmodel, discmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.ac_recurrence, args.disc_recurrence,
                                args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss, entropy_reward = args.entropy)
     

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")
    
    # Load demonstration

    if args.demonstration is None:
        args.demonstration = os.path.join(os.path.dirname(os.getcwd()), f'expert_demo/exp_demo_{args.env}.p')
    fp = open(args.demonstration, 'rb')
    demos = algo.collect_demonstrations(pickle.load(fp))
    fp.close()
    txt_logger.info(f"Demonstrations loaded: {len(demos)} frames\n")

    # Train model

    num_frames = status["num_frames"]
    print(num_frames)
    update = status["update"]
    start_time = time.time()

    while num_frames < args.frames:
        # Update model parameters
        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_ac_parameters(exps)
        logs3 = algo.update_disc_parameters(exps, demos)
        logs = {**logs1, **logs2, **logs3}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"] / (update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "ac_grad_norm", "disc_loss", "disc_grad_norm", "exps_acc", "demos_acc"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["ac_grad_norm"], logs['disc_loss'], logs['disc_grad_norm'], logs["exps_acc"], logs["demos_acc"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇a {:.3f} | dL: {:3f} | ∇d {:.3f} | exps_acc {:.3f} | demos_acc {:.3f}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            #for field, value in zip(header, data):
            tb_writer.add_scalar(return_per_episode[0], num_frames)

        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "acmodel_state": acmodel.state_dict(), "ac_optimizer_state": algo.ac_optimizer.state_dict(), 
                      "discmodel_state": discmodel.state_dict(), "dc_optimizer_state": algo.disc_optimizer.state_dict(), 
                      }
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")
