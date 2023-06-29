# This code is mainly excerpted from openai baseline code.
# https://github.com/openai/baselines/blob/master/baselines/bench/monitor.py
import os
import csv
import gym
import json
import time
import numpy as np
from glob import glob
import os.path as osp
from gym.core import Wrapper
from collections import deque

import toy_2d_2g as maze


__all__ = ['Monitor', 'get_monitor_files', 'load_results']

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


class Monitor(Wrapper):
    EXT = "monitor_.csv"
    RECORD = "performance.csv"
    f = None
    PROJECT_ROOT = os.path.dirname(os.path.realpath(__file__))

    def __init__(
        self, env, max_ep_len, update_freq,
        allow_early_resets=False, reset_keywords=(),
        basic_keywords=('r', 'l', 't', 'reward_ctrl', 'goal_idx'), 
        info_keywords=(
            'aver_return_iter', 'aver_return_smooth', 'success_rate', 
            'aver_mmd_distance', 'com',
            ),
        extra_keywords=('num_goal_0', 'num_goal_1', 'num_goal_2')
        ):
        Wrapper.__init__(self, env=env)
        self.env = env
        self.tstart = time.time()
        folder_name = self.env.spec + "_tcppo_" + time.strftime("%d-%m-%Y_%H-%M-%S") 
        self.f_path = os.path.join(self.__class__.PROJECT_ROOT, "data/", folder_name)
        self.results_writer = None
        self.reset_keywords = reset_keywords
        self.basic_keywords = basic_keywords
        self.info_keywords = info_keywords
        self.extra_keywords = extra_keywords
        self.max_ep_len = max_ep_len
        self.update_freq = update_freq
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.episode_num = 0
        self.mmd_distances = deque(maxlen=self.update_freq)
        self.success_indicator = deque(maxlen=update_freq)
        self.epres = deque(maxlen=update_freq)
        self.episode_returns = deque(maxlen=100)
        # extra info about the current episode, that was passed in during reset()
        self.current_reset_info = {} 

    def reset(self, **kwargs):
        self.reset_state()
        for k in self.reset_keywords:
            v = kwargs.get(k)
            if v is None:
                raise ValueError('Expected you to pass kwarg %s into reset'%k)
            self.current_reset_info[k] = v
        return self.env.reset(**kwargs)

    def reset_state(self):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.needs_reset = False

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, infos = self.env.step(action)
        
        infos = self.update(ob, rew, done, infos)
        return ob, rew, done, infos

    def update(self, ob, rew, done, infos):
        aver_return_iter = 0. 
        aver_return_smooth = 0.
        success_rate = 0.
        self.rewards.append(rew)
        if done or len(self.rewards) == self.max_ep_len:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            
            self.episode_rewards.append(eprew)
            self.epres.append(eprew)
            self.episode_returns.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            self.success_indicator.append(done)
            
            aver_return_iter = round(np.mean([value for value in self.epres]), 2)    
            aver_return_smooth = round(np.mean([value for value in self.episode_returns]), 2)
            
            # Compute the success rate.
            length = len(self.success_indicator)
            success_rate = 0.0
            for success in self.success_indicator:
                success_rate += float(success) / length
            
            epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
            epinfo["aver_return_iter"] = aver_return_iter 
            epinfo["aver_return_smooth"] = aver_return_smooth
            epinfo["success_rate"] = success_rate
            
            for k in infos.keys():
                epinfo[k] = infos[k]
            epinfo.update(self.current_reset_info)
            # self.results_writer.write_row(epinfo)

            if not isinstance(infos, dict):
                infos = {}
            infos['episode'] = epinfo

        self.total_steps += 1
        
        return infos
    
    def write_data(self, epinfos, mmd_distances):
        num_goal_0 = 0
        num_goal_1 = 0 
        num_goal_2 = 0
        for idx, epinfo in enumerate(epinfos):
            self.episode_num += 1
            if mmd_distances:
                self.mmd_distances.append(mmd_distances[idx])
                aver_mmd_distance = round(np.mean([value for value in self.mmd_distances]), 2)
                epinfo["aver_mmd_distance"] = aver_mmd_distance
            elif not mmd_distances:
                epinfo["aver_mmd_distance"] = 0
            
            aver_return_iter = epinfo["aver_return_iter"]
            aver_return_smooth = epinfo["aver_return_smooth"]
            success_rate = epinfo["success_rate"]
            aver_mmd_distance = epinfo["aver_mmd_distance"]
            goal = epinfo["goal_idx"]
            print("!!!!!!!goal: ", goal)
            
            if goal == maze.goals[0]:
                num_goal_0 += 1
            elif goal == maze.goals[1]:
                num_goal_1 += 1
            elif goal == maze.goals[2]:
                num_goal_2 += 1
            
            perform_info = {}
            if self.episode_num % self.update_freq == 0:
                perform_info["aver_return_iter"] = aver_return_iter 
                perform_info["aver_return_smooth"] = aver_return_smooth
                perform_info["success_rate"] = success_rate
                perform_info["aver_mmd_distance"] = aver_mmd_distance
                perform_info["num_goal_0"] = num_goal_0
                perform_info["num_goal_1"] = num_goal_1
                perform_info["num_goal_2"] = num_goal_2
                self.performance_writer.write_row(perform_info)
                
            self.results_writer.write_row(epinfo)
    
    def init_results_writer(self, policy_num):
        # initialize parameters
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.episode_num = 0
        # initialize deques
        self.mmd_distances = deque(maxlen=self.update_freq)
        self.success_indicator = deque(maxlen=self.update_freq)
        self.epres = deque(maxlen=self.update_freq)
        self.episode_returns = deque(maxlen=100)
        
        filename = self.configure_output_dir(policy_num)
        self.results_writer = ResultsWriter(
            Monitor.EXT,
            filename,
            header={"t_start": time.time(), "env_id": self.env.spec},
            extra_keys=self.basic_keywords + self.info_keywords)
        self.performance_writer = ResultsWriter(
            Monitor.RECORD,
            filename,
            header={"t_start": time.time(), "env_id": self.env.spec},
            extra_keys=self.info_keywords + self.extra_keywords)
        self.check_results_writer(policy_num)
    
    def configure_output_dir(self, policy_num, d=None):
        """
        Set output directory to d, or to /tmp/somerandomnumber if d is None
        """
        # Add this attribution to indicate which policy the object of the class ResultsWriter
        # is used for
        self.policy_num = policy_num
        
        d = os.path.join(self.f_path, "policy_"+str(policy_num))
        output_dir = d or "/tmp/experiments/%i"%int(time.time())
        if osp.exists(output_dir):
            print("Log dir %s already exists!"%output_dir)
        else:
            os.makedirs(output_dir)
        # atexit.register(self.output_file.close)
        print(colorize("Logging data to %s"%output_dir, 'green', bold=True))
        return output_dir
    
    def check_results_writer(self, policy_num):
        if self.results_writer == None or self.policy_num != policy_num:
            return False
        else:
            return True

    def close(self):
        if self.f is not None:
            self.f.close()
    
    def reset_monitor(self):
        self.epres = deque(maxlen=self.update_freq)
        self.episode_returns = deque(maxlen=100)

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_episode_times(self):
        return self.episode_times

class LoadMonitorResultsError(Exception):
    pass


class ResultsWriter(object):
    def __init__(self, file, filename=None, header='', extra_keys=()):
        self.extra_keys = extra_keys
        self.file = file
        if filename is None:
            self.f = None
            self.logger = None
        else:
            if not filename.endswith(self.file):
                if osp.isdir(filename):
                    filename = osp.join(filename, self.file)
                else:
                    filename = filename + "." + self.file
            self.f = open(filename, "wt")
            if isinstance(header, dict):
                header = '# {} \n'.format(json.dumps(header))
            self.f.write(header)
            self.logger = csv.DictWriter(self.f, fieldnames=tuple(extra_keys))
            self.logger.writeheader()
            self.f.flush()

    def write_row(self, epinfo):
        if self.logger:
            self.logger.writerow(epinfo)
            self.f.flush()


def get_monitor_files(dir):
    return glob(osp.join(dir, "*" + Monitor.EXT))

def load_results(dir):
    import pandas
    monitor_files = (
        glob(osp.join(dir, "*monitor.json")) +
        glob(osp.join(dir, "*monitor.csv"))) # get both csv and (old) json files
    if not monitor_files:
        raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % Monitor.EXT, dir)
    dfs = []
    headers = []
    for fname in monitor_files:
        with open(fname, 'rt') as fh:
            if fname.endswith('csv'):
                firstline = fh.readline()
                if not firstline:
                    continue
                assert firstline[0] == '#'
                header = json.loads(firstline[1:])
                df = pandas.read_csv(fh, index_col=None)
                headers.append(header)
            elif fname.endswith('json'): # Deprecated json format
                episodes = []
                lines = fh.readlines()
                header = json.loads(lines[0])
                headers.append(header)
                for line in lines[1:]:
                    episode = json.loads(line)
                    episodes.append(episode)
                df = pandas.DataFrame(episodes)
            else:
                assert 0, 'unreachable'
            df['t'] += header['t_start']
        dfs.append(df)
    df = pandas.concat(dfs)
    df.sort_values('t', inplace=True)
    df.reset_index(inplace=True)
    df['t'] -= min(header['t_start'] for header in headers)
    df.headers = headers # HACK to preserve backwards compatibility
    return df

def test_monitor():
    env = gym.make("CartPole-v1")
    env.seed(0)
    mon_file = "/tmp/baselines-test-%s.monitor.csv" % uuid.uuid4()
    menv = Monitor(env, mon_file)
    menv.reset()
    for _ in range(1000):
        _, _, done, _ = menv.step(0)
        if done:
            menv.reset()

    f = open(mon_file, 'rt')

    firstline = f.readline()
    assert firstline.startswith('#')
    metadata = json.loads(firstline[1:])
    assert metadata['env_id'] == "CartPole-v1"
    assert set(metadata.keys()) == {'env_id', 'gym_version', 't_start'},  "Incorrect keys in monitor metadata"

    last_logline = pandas.read_csv(f, index_col=None)
    assert set(last_logline.keys()) == {'l', 't', 'r'}, "Incorrect keys in monitor logline"
    f.close()
    os.remove(mon_file)
    
def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)