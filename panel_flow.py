from __future__ import annotations

import numpy as np

from itertools import compress
from dataclasses import dataclass
from building import Building


@dataclass
class Vehicle:
    """Vehicle with the minimal required information: position, velocity, altitude etc"""
    ID:str
    position:np.ndarray
    goal:np.ndarray
    sink_strength:float = 5.
    imag_source_strength:float = 0. #2.


class PanelFlow:
    def __init__(self) -> None:
        pass

    def prepare_buildings(self, obstacles:list[Building], panel_size:float = 0.05):
        for building in obstacles:
            if building.K_inv is None:
                building.panelize(panel_size)
                building.calculate_coef_matrix()

    def calculate_unknown_vortex_strengths(self, vehicle:Vehicle, obstacles:list[Building])->None:
        '''vehicles are the personal_vehicle_list containing all other vehicles'''

        # Remove buildings with heights below cruise altitude:
        altitude_mask = self.altitude_mask(vehicle, obstacles)
        # related_buildings keeps only the buildings for which the altitude_mask is 1, ie buildings that are higher than the altitude
        # of the vehicle in question
        related_buildings:list[Building] = list(compress(obstacles, altitude_mask))
        # Vortex strength calculation (related to panels of each building):
        for building in related_buildings:
            self.gamma_calc(building, vehicle)


    def altitude_mask(self, vehicle: Vehicle, obstacles:list[Building]):
        mask = np.zeros((len(obstacles)))
        for index, panelledbuilding in enumerate(obstacles):
            if (panelledbuilding.vertices[:, 2] > vehicle.position[2]).any():
                mask[index] = 1
        return mask

    def calculate_induced_sink_velocity(self,vehicle:Vehicle):

        
        position_diff_3D = vehicle.position - vehicle.goal
        position_diff_2D = position_diff_3D[:2]
        squared_distance = np.linalg.norm(position_diff_2D)**2

        # Avoid division by zero (in case the vehicle is exactly at the sink)
        # squared_distances[squared_distances == 0] = 1

        # Calculate induced velocity
        induced_v = (
            -vehicle.sink_strength
            * position_diff_2D
            / (2 * np.pi * squared_distance)
        )

        return induced_v


    def distance_effect_function(self, distance_array: np.ndarray, max_distance:float) -> np.ndarray:
        """Drop-off effect based on distance, accepting an array of distances."""
        max_distance = 3
        ratio = distance_array / max_distance
        linear_dropoff = 1 - ratio
        exponential_dropoff = 1 / (1 + np.exp(10 * (ratio - 0.7)))
        
        effect = np.zeros_like(distance_array)  # Initialize array to store results

        # Apply conditions
        near_indices = distance_array < 1
        far_indices = ~near_indices  # Elements not satisfying the 'near' condition

        effect[near_indices] = (1 / (2 * np.pi * distance_array[near_indices] ** 4))
        effect[far_indices] = (1 / (2 * np.pi * distance_array[far_indices] ** 2)) * exponential_dropoff[far_indices]

        return effect



    
    def calculate_induced_building_velocity(self, main_vehicle: Vehicle, obstacles:list[Building]):
        # arena = main_vehicle.arena
        # buildings = main_vehicle.relevant_obstacles
        # Determine the number of buildings and the maximum number of panels in any building
        num_buildings = len(obstacles)
        max_num_panels = max(building.nop for building in obstacles)

        # Initialize the all_pcp array with zeros
        all_vp = np.zeros((num_buildings, max_num_panels, 2))

        # Populate the all_pcp array
        for i, building in enumerate(obstacles):
            num_panels = building.nop  # Number of panels in the current building
            all_vp[i, :num_panels, :] = building.vp[:num_panels, :2]

        # Initialize the all_gammas array with zeros or NaNs
        all_gammas = np.zeros((num_buildings, max_num_panels))

        # Populate the all_gammas array
        for i, building in enumerate(obstacles):
            num_panels = building.nop  # Number of panels in the current building
            if main_vehicle.ID in building.gammas:
                all_gammas[i, :num_panels] = building.gammas[main_vehicle.ID][:num_panels].ravel()

        # Get position of the main_vehicle
        main_vehicle_position = main_vehicle.position[:2]

        # Calculate position differences and distances
        diff = main_vehicle_position - all_vp
        squared_distances = np.sum(diff ** 2, axis=-1)

        # Create the numerator for all buildings
        vec_to_vehicle = np.zeros((num_buildings, max_num_panels, 2))

        #don't need to reshape, broadcasting is possible
        vec_to_vehicle =  main_vehicle_position - all_vp 

        # Normalize all_gammas
        all_gammas_normalized = all_gammas / (2 * np.pi)

        # uv calculations
        uv = all_gammas_normalized[:, :, np.newaxis] * vec_to_vehicle / squared_distances[:, :, np.newaxis]

        # Summing across num_buildings and num_panels axes
        V_gamma_main = np.sum(uv, axis=(0, 1))
        #set x to y and y to -x (rotate vec_to_vehicle by pi/2 clockwise)
        #rotate 90 degrees clockwise to corresponding to vortex effect
        V_gamma_main[0], V_gamma_main[1] = V_gamma_main[1], -V_gamma_main[0]

        return V_gamma_main

    
    
    
    def gamma_calc(self, building:Building, vehicle:Vehicle):
        """Calculate the unknown vortex strengths of the building panels

        Args:
            vehicle (Vehicle): _description_
            othervehicles (list[Vehicle]): _description_
        """

        # Initialize arrays in case no other vehicles
        vel_sink = np.zeros((building.nop, 2))
        vel_source_imag = np.zeros((building.nop, 2))
        RHS = np.zeros((building.nop, 1))

        # Pre-calculate repeated terms
        sink_diff = building.pcp[:,:2] - vehicle.goal[:2]
        sink_sq_dist = np.sum(sink_diff ** 2, axis=-1)
        imag_diff = building.pcp[:,:2] - vehicle.position[:2]
        imag_sq_dist = np.sum(imag_diff ** 2, axis=-1)

        # Velocity calculations for sink and imag_source
        vel_sink = -vehicle.sink_strength * sink_diff / (2 * np.pi * sink_sq_dist)[:, np.newaxis]
        vel_source_imag = vehicle.imag_source_strength * imag_diff / (2 * np.pi * imag_sq_dist)[:, np.newaxis]

       
        # RHS calculation
        cos_pb = np.cos(building.pb)
        sin_pb = np.sin(building.pb)

        normal_vectors = np.array([cos_pb, sin_pb]).T
        
        # Combine all velocity components into a single array before summing
        total_velocity = (
            # effect from sink if using
            vel_sink +
            # free stream velocity
            # vehicle.v_free_stream + 
            vel_source_imag 
        )
        RHS[:, 0] = -np.sum(total_velocity * normal_vectors, axis=1)
        # Solve for gammas
        #gammas is dictionary because a building might have different gammas for different vehicles
        building.gammas[vehicle.ID] = np.matmul(building.K_inv, RHS)

    def is_inside_buildings(self, buildings:list[Building], position:np.ndarray)->None|Building:
        for b in buildings:
            if b.contains_point(position):
                return b
        return None
    
    def eject(self, vehicle:Vehicle, building:Building)->np.ndarray:
        pos2d = vehicle.position[:2]
        nearest_point = building.nearest_point_on_perimeter(pos2d)
        ejection_vector = nearest_point-pos2d
        return ejection_vector/np.linalg.norm(ejection_vector)
        

    def Flow_Velocity_Calculation(self,
        vehicle:Vehicle, obstacles:list[Building]|None = None)->np.ndarray:

        V_gamma, V_sink = np.zeros(2), np.zeros(2)
        # Calculating unknown vortex strengths using panel method:
        if obstacles:
            self.prepare_buildings(obstacles) #calculate matrices 
            #first if a vehicle is inside any buildings, push it out by the "nearest exit"
            containing_building = self.is_inside_buildings(obstacles,vehicle.position[:2])
            if containing_building is not None:
                return self.eject(vehicle, containing_building)
            #calculates unknown building vortex strengths
            self.calculate_unknown_vortex_strengths(vehicle, obstacles)
        # --------------------------------------------------------------------
            V_gamma = self.calculate_induced_building_velocity(vehicle, obstacles)


        # Velocity induced by 2D point sink, eqn. 10.2 & 10.3 in Katz & Plotkin:
        #calculate effect of sink
        V_sink = self.calculate_induced_sink_velocity(vehicle)
        # V_sink = vehicle.v_free_stream
    
        
        #########################################################################################################################
        # # Summing the effects for all vehicles at once
        V_sum = V_gamma + V_sink #+ vehicle.v_free_stream
        # Added a small constant to avoid division by zero
        V_sum /= (np.linalg.norm(V_sum)+1e-10)
        # Normalization and flow calculation for all vehicles
        flow_vels = V_sum   # no need to normalise if this is done in vehicle, TODO could normalise here instead, need to decide
        #########################################################################################################################
        return flow_vels


if __name__ == "__main__":
    verts1 = np.array([[0,0,2],[1,0,2],[1,1,2],[0,1,2]])
    verts2 = np.array([[0,0,2],[1,0,2],[1,1,2],[0,1,2]]) + np.array([0,2,0])

    B1 = Building(verts1)
    B2 = Building(verts2)
    obstacles = [B1, B2]
    v1 = Vehicle(ID='0', position=np.array([-0.1,0.5,3]), goal=np.array([2,0.5,3]))
    pf = PanelFlow()
    flow = pf.Flow_Velocity_Calculation(v1,obstacles)
    print(flow)


