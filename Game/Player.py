# Under MIT License, see LICENSE.txt
from ..Util.Pose import Pose
from ..Util.Vector import Vector
from ..Util.constant import DELTA_T

import numpy as np


class Player:
    def __init__(self, team, id):
        self.cmd = [0, 0, 0]
        self.id = id
        self.team = team
        self.pose = Pose()
        dt = DELTA_T
        self.transition_model = [[1, 0, dt, 0, 0, 0], [0, 1, 0, dt, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, dt], [0, 0, 0, 0, 0, 0]]
        control_input_model = [[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]
        observation_model = [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0]]

        process_covariance = 10 ** (-4) * np.eye(6)
        observation_covariance = np.eye(3) / 1

        initial_state_estimation = [self.pose.position.x, self.pose.position.y, 0, 0, self.pose.orientation, 0]
        initial_state_covariance = np.eye(6) / 1000
        self.kf = Kalman(self.transition_model, control_input_model, observation_model, process_covariance,
                         observation_covariance, initial_state_estimation, initial_state_covariance)

        self.velocity = Vector(0, 0)
        self.filtered_state_means = [self.pose.position.x, self.pose.position.y, 0, 0, self.pose.orientation, 0]
        self.filtered_state_covariances = np.eye(6)
        self.observations = [self.pose.position.x, self.pose.position.y, self.pose.orientation]

    def has_id(self, pid):
        return self.id == pid

    def update(self, pose, delta=DELTA_T):
        self.observations = [pose.position.x, pose.position.y, pose.orientation]
        self.transition_model = [[1, 0, delta, 0, 0, 0], [0, 1, 0, delta, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 1, delta], [0, 0, 0, 0, 0, 0]]
        print('cmd kalman', self.cmd[0], self.cmd[1], self.cmd[2])
        output = self.kf.filter(self.observations, self.cmd, self.transition_model)
        self.pose.position.x = output[0]
        self.pose.position.y = output[1]
        self.velocity = [output[2], output[3], 0]
        self.pose.orientation = 0

        old_pose = self.pose
        delta_position = pose.position - old_pose.position
        # if self.id == 1 and not self.team.is_team_yellow():
        #     print(self.pose.position.x, self.pose.position.y)

        # try:
        #     self.velocity.x = delta_position.x / delta
        #     self.velocity.y = delta_position.y / delta
        # except ZeroDivisionError:
        #     self.velocity = Vector(0, 0)
        # self.pose = pose

    def set_command(self, cmd):
        self.cmd = [cmd.pose.position.x, cmd.pose.position.y, self.pose.orientation]


class Kalman:
    def __init__(self, transition_model, control_input_model, observation_model, process_covariance,
                 observation_covariance, initial_state_estimation, initial_state_covariance):
        self.F = np.array(transition_model)
        self.B = np.array(control_input_model)
        self.H = np.array(observation_model)
        self.Q = np.array(process_covariance)
        self.R = np.array(observation_covariance)

        self.x = np.array(initial_state_estimation)
        self.P = np.array(initial_state_covariance)

    def predict(self, command):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, np.array(command))
        self.P = np.dot(np.dot(self.F, self.P), np.transpose(self.F)) + self.Q

    def update(self, observation):
        y = np.array(observation) - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), np.transpose(self.H)) + self.R
        K = np.dot(np.dot(self.P, np.transpose(self.H)), np.linalg.inv(S))

        self.x += np.dot(K, y)
        self.P = np.dot((np.eye(self.P.shape[0]) - np.dot(K, self.H)), self.P)

    def filter(self, observation, command, transition_model = None):
        if transition_model is None:
            self.F = np.array(transition_model)

        self.predict(command)
        self.update(observation)
        return self.x.tolist()
