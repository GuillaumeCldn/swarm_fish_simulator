"""
PointSim Interface
==================
Point-mass simulation interface for testing swarm algorithms.

This module provides the PointSim class for simple kinematic drone simulation
and the PointSimController for managing fleets of simulated drones.
"""

import logging
#import time
import threading
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from dataclasses import dataclass, field

#
# Utility functions
#
def correct_angle(angle: float) -> float:
    """
    Normalize an angle to the range [-pi, pi).

    Parameters
    ----------
    angle : float
        Input angle in radians.

    Returns
    -------
    float
        Normalized angle in radians.
    """
    while angle < -np.pi:
        angle += 2 * np.pi
    while angle >= np.pi:
        angle -= 2 * np.pi
    return angle

#
# UAV State
#

@dataclass
class QuadrotorState:
    """Complete state representation of a quadrotor.

    Attributes:
        pos: Position in world frame (m)
        vel: Linear velocity in world frame (m/s)
        acc: Linear acceleration in world frame (m/s^2)
        att: Attitude [roll, pitch, yaw] in radians
        omega: Angular velocity in body frame (rad/s)
        omegad: Angular acceleration in body frame (rad/s^2)
    """
    pos: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    vel: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    acc: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    att: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))  # roll, pitch, yaw
    omega: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    omegad: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))

    @property
    def R(self) -> np.ndarray:
        """Rotation matrix (body to world) computed from attitude."""
        return Rotation.from_euler('xyz', self.att).as_matrix()

#
# Simulation model
#

class PointSim:
    """
    Simulated drone for testing swarm algorithms without physical hardware.
    
    This class simulates a single drone's dynamics and state withsimple kinematic models.
    
    Attributes
    ----------
    position : np.ndarray
        Current position [x, y, z] in meters.
    velocity : np.ndarray
        Current velocity [vx, vy, vz] in m/s.
    attitude : np.ndarray
        Current orientation [phi, theta, psi] in radians
    omega : np.ndarray
        Current rates [p, q, r] in radians/s
    
    """

    DEFAULT_MAX_ACCEL_H = 4.0  # m/s^2
    DEFAULT_MAX_ACCEL_V = 2.0  # m/s^2
    DEFAULT_MAX_OMEGAD = np.radians(500)  # rad/s^2
    DEFAULT_T_CONST_H = 0.1 # s
    DEFAULT_T_CONST_V = 0.25 # s
    DEFAULT_T_CONST_Y = 0.1 #
    
    def __init__(self, init_position: np.ndarray, init_yaw: float = 0.0) -> None:
        """
        Initialize a PointSim drone.
        
        Parameters
        ----------
        init_position : np.ndarray
            Initial position [x, y, z].
        init_yaw : float, optional
            Initial yaw angle in radians (default: 0.0).
        """
        
        drone_type = 'point_sim'  # Fixed type for PointSim
        ##config_path = get_config_path(drone_type)
        ##
        ### Load config
        ##with open(config_path, 'r', encoding='utf-8') as f:
        ##    cfg: dict = yaml.safe_load(f)
        ##
        ### Handle empty YAML files
        ##if cfg is None:
        ##    logging.warning(f"Config file '{config_path}' is empty. Using default parameters.")
        ##    cfg = {}

        self.a_max_h = PointSim.DEFAULT_MAX_ACCEL_H #safe_get(cfg, 'max_accel', PointSim.DEFAULT_MAX_ACCEL, float)
        self.a_max_v = PointSim.DEFAULT_MAX_ACCEL_V #safe_get(cfg, 'max_accel', PointSim.DEFAULT_MAX_ACCEL, float)
        self.omegad_max = PointSim.DEFAULT_MAX_OMEGAD #safe_get(cfg, 'max_omegad', PointSim.DEFAULT_MAX_OMEGAD, float)
        self.t_const = np.array([PointSim.DEFAULT_T_CONST_H, PointSim.DEFAULT_T_CONST_H, PointSim.DEFAULT_T_CONST_V])
        self.t_const_yaw = PointSim.DEFAULT_T_CONST_Y
        
        # Set drone state with individual attributes
        self.position: np.ndarray = np.zeros(3) if init_position is None else init_position
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.attitude = np.array([0.0, 0.0, float(init_yaw)])  # [roll, pitch, yaw]
        self.omega = np.zeros(3)
        self.omegad = np.zeros(3)
    
        self.lock = threading.Lock()

    def update_state(self, desired_velocity: np.ndarray, desired_yaw_rate: float, time_step: float):
        """
        Update drone state based on desired velocity and yaw rate.
        
        Parameters
        ----------
        desired_velocity : np.ndarray
            Desired velocity vector [vx, vy, vz] in m/s.
        yaw_rate : float
            Desired yaw rate in radians/s.
        time_step : float
            Time step for integration in seconds.
        """
        with self.lock:
            # Simple kinematic model without dynamics
            # Calculate required acceleration. Split horizontal and vertical
            acceleration = (desired_velocity - self.velocity) / self.t_const
            accel_h_norm = np.linalg.norm(acceleration[0:2])
            accel_v_norm = np.abs(acceleration[2])
            if accel_h_norm > self.a_max_h: #limit acceleration
                acceleration[0:2] = (acceleration[0:2] / accel_h_norm) * self.a_max_h
            if accel_v_norm > self.a_max_v:
                acceleration[2] = (acceleration[2] / accel_v_norm) * self.a_max_v
            
            # Update position using current velocity (before acceleration is applied)
            position = self.position + self.velocity * time_step
            # Then update velocity for next iteration
            velocity = self.velocity + acceleration * time_step

            # Calculate required yaw acceleration and bound
            omegad_z = (desired_yaw_rate - self.omega[2]) / self.t_const_yaw
            omegad_z = max(-self.omegad_max, min(self.omegad_max, omegad_z))
            omegad = np.array([0.0, 0.0, float(omegad_z)])
            # Update attitude using current omega (z axis only)
            attitude_yaw = self.attitude[2]  + self.omega[2] * time_step
            attitude = np.array([0.0, 0.0, correct_angle(attitude_yaw)])  # Normalize angles
            # Calculate omega
            omega_z = self.omega[2] + omegad_z * time_step
            omega = np.array([0.0, 0.0, float(omega_z)])

            # Update individual state attributes
            self.position = position
            self.velocity = velocity
            self.acceleration = acceleration
            self.attitude = attitude
            self.omega = omega
            self.omegad = omegad

    def get_position(self) -> np.ndarray:
        """
        Get current drone position.
        
        Returns
        -------
        np.ndarray
            Position vector [x, y, z] in meters.
        """
        with self.lock:
            position = self.position.copy()
        return position
    
    def get_velocity(self) -> np.ndarray:
        """
        Get current drone velocity.
        
        Returns
        -------
        np.ndarray
            Velocity vector [vx, vy, vz] in m/s.
        """
        with self.lock:
            velocity = self.velocity.copy()
        return velocity
    
    def get_acceleration(self) -> np.ndarray:
        """
        Get current drone acceleration.
        
        Returns
        -------
        np.ndarray
            Acceleration vector [ax, ay, az] in m/s^2.
        """
        with self.lock:
            acceleration = self.acceleration.copy()
        return acceleration
    
    def get_attitude(self) -> np.ndarray:
        """
        Get current drone attitude.
        
        Returns
        -------
        np.ndarray
            Attitude vector [roll, pitch, yaw] in radians.
        """
        with self.lock:
            attitude = self.attitude.copy()
        return attitude
    
    def get_angular_velocity(self) -> np.ndarray:
        """
        Get current drone angular velocity.
        
        Returns
        -------
        np.ndarray
            Angular velocity vector [wx, wy, wz] in rad/s.
        """
        with self.lock:
            omega = self.omega.copy()
        return omega
    
    def get_angular_acceleration(self) -> np.ndarray:
        """
        Get current drone angular acceleration.
        
        Returns
        -------
        np.ndarray
            Angular acceleration vector [alphax, alphay, alphaz] in rad/s^2.
        """
        with self.lock:
            omegad = self.omegad.copy()
        return omegad
    
    def get_state(self) -> QuadrotorState:
        """
        Get current full drone state.
        
        Returns
        -------
        QuadrotorState
            Current state of the drone.
        """
        with self.lock:
            state = QuadrotorState(
                pos=self.position.copy(),
                vel=self.velocity.copy(),
                att=self.attitude.copy(),
                omega=self.omega.copy(),
                acc=self.acceleration.copy(),
                omegad=self.omegad.copy()
            )
        return state


