import os
from datetime import datetime
import numpy as np


class Logger(object):
    """A class for logging.

    """

    ################################################################################

    def __init__(
        self,
        logging_freq_hz: int,
        state_length: int = 12,
        control_length: int = 4,
        num_drones: int = 1,
        duration_sec: int = 0,
    ):
        """Logger class __init__ method.

        Parameters
        ----------
        logging_freq_hz : int
            Logging frequency in Hz.
        num_drones : int, optional
            Number of drones.
        duration_sec : int, optional
            Used to preallocate the log arrays (improves performance).

        """
        self.state_length = state_length
        self.control_length = control_length
        self.LOGGING_FREQ_HZ = logging_freq_hz
        self.NUM_DRONES = num_drones
        self.PREALLOCATED_ARRAYS = False if duration_sec == 0 else True
        self.counters = np.zeros(num_drones)
        self.nb_steps = duration_sec * self.LOGGING_FREQ_HZ
        self.timestamps = np.zeros((num_drones, self.nb_steps))
        #### Note: this is the suggest information to log ##############################
        self.states = np.zeros(
            (num_drones, self.state_length, self.nb_steps)
        )  #### 12 states:
        # pos_x, pos_y, pos_z,
        # vel_x, vel_y, vel_z,
        # roll, pitch, yaw,
        # ang_vel_x, ang_vel_y, ang_vel_z,
        #### Note: this is the suggest information to log ##############################
        self.controls = np.zeros(
            (num_drones, self.control_length, self.nb_steps)
        )  #### 4 control targets:
        # vel_x, vel_y, vel_z, ang_vel_z

    ################################################################################

    def log(self, drone: int, timestamp, state, control=np.zeros(4)):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        drone : int
            Id of the drone associated to the log entry.
        timestamp : float
            Timestamp of the log in simulation clock.
        state : ndarray
            (20,)-shaped array of floats containing the drone's state.
        control : ndarray, optional
            (4,)-shaped array of floats containing the drone's control target.

        """
        if (
            drone < 0
            or drone >= self.NUM_DRONES
            or timestamp < 0
            or len(state) != self.state_length
            or len(control) != self.control_length
        ):
            print(f" State Length : {self.state_length}, Control Length : {self.control_length}")
            print("[ERROR] in Logger.log(), invalid data")
        current_counter = int(self.counters[drone])
        #### Add rows to the matrices if a counter exceeds their size
        if current_counter >= self.timestamps.shape[1]:
            self.timestamps = np.concatenate(
                (self.timestamps, np.zeros((self.NUM_DRONES, 1))), axis=1
            )
            self.states = np.concatenate(
                (self.states, np.zeros((self.NUM_DRONES, self.state_length, 1))), axis=2
            )
            self.controls = np.concatenate(
                (self.controls, np.zeros((self.NUM_DRONES, self.control_length, 1))),
                axis=2,
            )
        #### Advance a counter is the matrices have overgrown it ###
        elif (
            not self.PREALLOCATED_ARRAYS and self.timestamps.shape[1] > current_counter
        ):
            current_counter = self.timestamps.shape[1] - 1
        #### Log the information and increase the counter ##########
        self.timestamps[drone, current_counter] = timestamp
        #### Re-order the kinematic obs (of most Aviaries) #########
        self.states[drone, :, current_counter] = state
        self.controls[drone, :, current_counter] = control
        self.counters[drone] = current_counter + 1

    ################################################################################

    def save(self, file_path=None, file_name=None):
        """Save the logs to file."""
        if file_path == None:
            file_path = (
                os.path.dirname(os.path.abspath(__file__)) + "/../logs/"
            )
        if file_name == None:
            file_name = "save-flight-" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        with open(file_path + file_name + ".npy", "wb") as out_file:
            np.savez(
                out_file,
                timestamps=self.timestamps,
                states=self.states,
                controls=self.controls,
            )

