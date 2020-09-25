from cv2 import VideoWriter, VideoWriter_fourcc
from os import mkdir
from os.path import abspath
import numpy as np
import datetime
import imageio

from params import Params


class VideoRecorder:

    def __init__(self):

        if not Params.RECORD_VIDEO:
            return

        self.frame_size = Params.FRAME_SIZE.numpy()
        self.pad_len = 100
        self.out_type = Params.RECORD_VIDEO_TYPE

        if np.less(self.frame_size, 64).any():
            raise Exception("Frame size must be > 64px")

        self.video_writer = None
        self.writer_path = 'recorded/' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

        try:
            mkdir(self.writer_path)
        except OSError:
            raise FileNotFoundError(f"Creation of the directory {self.writer_path} failed")

    def pad(self):
        for i in range(self.pad_len):
            if self.video_writer is not None:
                self.video_writer.write(np.zeros(self.frame_size.astype(int)))

    def save_video(self, frames, n_episode, ep_avg_reward):
        path = self.writer_path + '/episode_' + str(n_episode.numpy()) + '_avg_reward_' + str(int(ep_avg_reward))

        if self.out_type == "GIF":
            path += ".gif"
            with imageio.get_writer(path, mode='I', duration=0.04) as writer:
                for i in range(frames.shape[0]):
                    writer.append_data(frames[i].numpy())

        elif self.out_type == "MP4":
            path += ".mp4"
            self.video_writer = VideoWriter(path, VideoWriter_fourcc(*'mp4v'), 120., tuple(self.frame_size), isColor=False)  # alternative codec: MJPG

            for i in range(frames.shape[0]):
                self.video_writer.write(frames[i].numpy())

            self.pad()
            self.video_writer.release()

        self.video_writer = None

        print("saved video ...")

        return abspath(path)
