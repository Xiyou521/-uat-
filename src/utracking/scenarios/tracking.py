import re
import numpy as np
from utracking.core import World, Agent, Landmark
from utracking.scenario import BaseScenario
from utracking.tracking.target_pf import random_levy, Target
from utracking.tracking.target_ls import TargetLS

def euclidean_distance(e1_pos, e2_pos):
    return np.sqrt(np.sum(np.square(e1_pos - e2_pos)))


class Scenario(BaseScenario):
    
    def make_world(
        self,
        num_agents=3,
        num_landmarks=3,
        landmark_depth=15.,
        landmark_movable = False,
        agent_vel=0.3,
        landmark_vel=0.05,
        max_vel=0.2, 
        random_vel=True, 
        movement='linear', 
        pf_method = False, 
        rew_err_th=0.0003, 
        rew_dis_th=0.3,
        sense_range=0.5,
        max_range = 1.,
        only_closer_agent_pos=True,
        min_init_dist = 0.2,
        max_init_dist = 0.6,
        difficulty='easy',
        obs_entity_mode=False
    ):

        world = World()
        # set any world properties first
        world.dim_c = 2
        world.num_agents = num_agents
        world.num_landmarks = num_landmarks
        world.collaborative = True
        # add agents
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.04
            agent.max_a_speed = 3.1415
            agent.state.vel = agent_vel
            agent.s_range = sense_range
        # add landmarks
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        world.landmarks_best_estimations = [Landmark() for _ in range(num_landmarks)]
        for i in range(world.num_landmarks):
            # update real landmarks
            landmark = world.landmarks[i]
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = landmark_movable
            landmark.movement = movement
            # and predicted
            landmark = world.landmarks_best_estimations[i]
            landmark.name = 'landmark_estimation %d' % (i)
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.002
                
        # make initial conditions
        world.cov = np.ones(num_landmarks)/30.
        world.error = np.ones(num_landmarks)
        
        # world.vel_ocean_current = 0.05
        # world.angle_ocean_current = np.pi/2.*3.
        
        self.agent_vel = agent_vel
        self.landmark_vel = landmark_vel
        
        #benchmark variables
        self.agent_outofworld = 0
        self.landmark_collision = 0
        self.agent_collision = 0

        #Scenario initial conditions
        self.landmark_depth = landmark_depth
        self.max_landmark_depth = landmark_depth
        self.ra = (np.random.rand(1)*np.pi*2.).item(0) #initial landmark direction
        self.movement = movement
        self.pf_method = pf_method
        self.rew_err_th = rew_err_th
        self.rew_dis_th = rew_dis_th
        self.max_range = max_range

        #variables for observations and reset
        self.max_vel = max_vel
        self.random_vel = random_vel
        self.only_closer_agent_pos = only_closer_agent_pos
        self.obs_entity_mode = obs_entity_mode
        self.min_init_dist = min_init_dist
        self.max_init_dist = max_init_dist
        
        # difficultiy parameters
        world.damping = landmark_vel/2.

        assert difficulty in ['control','easy', 'medium', 'hard']
        
        if difficulty == 'control':
            self.range_droping = 0.
            self.max_vel_ocean_current = 0.
        elif difficulty == 'easy':
            self.range_droping = 0.05
            self.max_vel_ocean_current = self.agent_vel*0.1
        elif difficulty == 'medium':
            self.range_droping = 0.25
            self.max_vel_ocean_current = self.agent_vel*0.3
        elif difficulty == 'hard':
            self.range_droping = 0.45
            self.max_vel_ocean_current = self.agent_vel*0.666

        # world settings for tracking
        world.pf_method = self.pf_method
        world.max_range = self.max_range

        #make initial world current 
        world.vel_ocean_current = np.random.rand(1).item(0)*self.max_vel_ocean_current #initial random strength
        world.angle_ocean_current = (np.random.rand(1)*np.pi*2.).item(0) #initial landmark direction

        self.reset_world(world)
        
        return world

    def reset_world(self, world):

        agent_positions, landmark_positions = self.get_init_pos(world)

        # set random initial states
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = agent_positions[i]
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.p_vel_old = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
            agent.state.a_vel = 0.
            agent.state.p_pos_origin = agent.state.p_pos.copy()
            # intialize the landmark estimations for each agent
            agent.landmarks_estimated = [Target() if self.pf_method else TargetLS() for _ in range(world.num_landmarks)]
            agent.landmark_predictions = np.full((world.num_landmarks, world.dim_p), agent.state.p_pos)
            agent.slants = np.full(world.num_landmarks, np.inf)

        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
            """
            #ivan approach
            dis = np.random.uniform(0.04, 1.)
            rad = np.random.uniform(0, np.pi*2)
            landmark.state.p_pos = agent.state.p_pos + np.array([np.cos(rad),np.sin(rad)])*dis
            """
            landmark.state.p_pos = landmark_positions[i]
            landmark.state.p_vel = np.zeros(world.dim_p)
            
        for landmark in world.landmarks_best_estimations:
            landmark.color = np.array([0.55, 0.0, 0.0])
            landmark.state.p_pos = world.agents[0].state.p_pos
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.best_slant = np.inf
        
        
        #initialize the ocean current at random
        world.vel_ocean_current = np.random.rand(1).item(0)*self.max_vel_ocean_current #initial random strength
        world.angle_ocean_current = (np.random.rand(1)*np.pi*2.).item(0) #initial landmark direction
        
        #benchmark variables
        self.agent_outofworld = 0
        self.landmark_collision = 0
        self.agent_collision = 0

        # global random direction for turning movemnt 
        rd = (np.random.rand(1)*np.pi*2.).item(0)
        for landmark in world.landmarks:
            #take a random velocity
            if self.random_vel == True:
                landmark.landmark_vel = np.random.rand(1).item(0)*self.max_vel
            else:
                landmark.landmark_vel = self.landmark_vel
            landmark.max_speed = landmark.landmark_vel
            #take a random direction (the same for all if turning movoment)
            if self.movement == 'turning':
                landmark.ra = rd
                landmark.ra_sign = np.random.choice([-1,1]) # in which direction the landmark is going to turn
            else:
                landmark.ra = (np.random.rand(1)*np.pi*2.).item(0) #initial landmark direction
            landmark.landmark_depth = self.landmark_depth

        #compute initial distances
        world.update_distances()
        # update initial slants
        for agent in world.agents:
            world.update_agent_state(agent)

        # set done to false
        self.done_state = False

        # restart time step count
        world.t = 0


    def get_init_pos(self, world):
        # initalize random positions withing min and max distance
        xy_agents = np.zeros((world.num_agents, world.dim_p))
        for i in range(world.num_agents):
            if i==0:
                xy_agents[i] = np.random.uniform(-0.5, 0.5, world.dim_p)
            else:
                x_dist  = np.random.uniform(self.min_init_dist, self.max_init_dist)
                y_range = np.sqrt(self.max_init_dist**2-x_dist**2)
                y_dist  = np.random.uniform(-y_range, y_range)
                xy_agents[i] = xy_agents[i-1] + [x_dist, y_dist]

        # reassign randomly each position to the agents
        pos_per_agent = np.random.choice(world.num_agents, world.num_agents, replace=False)
        agent_positions = xy_agents[pos_per_agent]

        # set the position of the landmarks at a rand distance (withing min-max boundaries) from a rand agent
        if self.movement == 'turning':
            # landmarks span around the same spot
            x_max, y_max = agent_positions.max(axis=0)
            landmark_positions = np.zeros((world.num_landmarks, world.dim_p))
            landmark_positions[:, 0] = np.random.uniform(x_max+self.min_init_dist, x_max+self.min_init_dist+0.2, world.num_landmarks)
            landmark_positions[:, 1] = np.random.uniform(y_max+self.min_init_dist, y_max+self.min_init_dist+0.2, world.num_landmarks)

        #² TODO: increase randomness of span by allowing distance to be east, south, north, west (now only north)
        elif self.movement == 'levy':
            land_per_agent = np.random.choice(world.num_landmarks, world.num_agents)
            landmark_dist  = np.random.uniform(self.min_init_dist, world.max_range*2, (world.num_landmarks,world.dim_p))
            landmark_positions = agent_positions[land_per_agent] + landmark_dist
        else:
            land_per_agent = np.random.choice(world.num_landmarks, world.num_agents)
            landmark_dist  = np.random.uniform(self.min_init_dist, world.max_range*0.6, (world.num_landmarks,world.dim_p))
            landmark_positions = agent_positions[land_per_agent] + landmark_dist
        
        return agent_positions, landmark_positions
        

    def benchmark_data(self, world):
        stats = {}
        for i, landmark in enumerate(world.landmarks):
            stats[f'{landmark.name}_error'] = min(100, world.error[i])
            stats[f'{landmark.name}_pos']  = landmark.state.p_pos
            stats[f'{landmark.name}_pred'] = world.landmarks_best_estimations[i].state.p_pos
            stats[f'{landmark.name}_lost'] = int(min([agent.slants[i] for agent in world.agents]) > self.max_range)
            stats[f'{landmark.name}_min_dist']  = world.min_distances[landmark.name]
        stats['agent_outofworld']   = self.agent_outofworld
        stats['landmark_collision'] = self.landmark_collision
        stats['agent_collision']    = self.agent_collision

        return stats


    def is_collision(self, agent1, agent2, world):
        dist = world.entity_distances[f'{agent1.name}-{agent2.name}']
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False
    
    def reward(self, world):
        # The reward is computed in respect to the landmarks
                
        rew = 0.
        for i, landmark in enumerate(world.landmarks):

            # positive reward if the best predictions of each agent is below the err threshold 
            min_err = min([euclidean_distance(agent.landmark_predictions[i], landmark.state.p_pos) for agent in world.agents])
            if min_err > self.rew_err_th:
                rew += 0.01*(0.004-min_err)

            # positive reward if the closest agent to this landmark is below the err threshold 
            min_dist = world.min_distances[landmark.name]
            if min_dist < self.rew_dis_th: 
                rew += 1.
            else: 
                rew += 0.01*(0.7-min_dist)

            # negative reward if the closest agent is so far from it that cannot recieve signals
            if min_dist > np.sqrt(self.max_range**2-(landmark.landmark_depth/1000.)**2):
                rew -= 0.1
            elif min_dist > self.max_range*2:
                rew -= 100.
                self.done_state = True
                self.agent_outofworld += 1
            
            # negative reward if the closest agent is colliding to the landmark
            if min_dist < 0.02: #is collision
                rew -= 10.
                self.done_state = True
                self.landmark_collision += 1
        
        # check for agent collisions
        for i in range(len(world.agents)-1):
            #if world.agents[].collide:
            for j in range(i+1, len(world.agents)):
                if self.is_collision(world.agents[i], world.agents[j], world):
                    rew -= 10.
                    self.agent_collision += 1
                    self.done_state = True
        return rew


    def state(self, world):
        """Get the position, velocity and depth of each entity in the current state"""
        num_entities = world.num_agents+world.num_landmarks
        num_features = 6
        # agent features: (x, y, vel_x, vel_y, 0, 1)
        # landmark features: (x, y, vel_x, vel_y, detph, 0)

        feats = np.zeros(num_features*num_entities)
        
        # then other agents 
        i = 0
        for a in world.agents:
            pos = a.state.p_pos
            vel = a.state.p_vel
            depth = 0.
            feats[i:i+num_features] = [pos[0],pos[1],vel[0],vel[1],depth,1.]
            i += num_features

        # and finally, landmarks
        for landmark in world.landmarks:
            pos = landmark.state.p_pos
            vel = landmark.state.p_vel
            detph = landmark.landmark_depth/1000.
            feats[i:i+num_features] = [pos[0],pos[1],vel[0],vel[1],detph,0.]
            i += num_features

        return feats

    def observation(self, agent, world):

        if self.obs_entity_mode:
            """Entity approach"""
            # 6 features for each entity: (pos_x, pos_y, depth, distance, agent, self)
            # agent 1 for agent 1:    (x_a1, y_a1, 0, 0, 1, 1)
            # agent 2 for agent 1:    (dx(a1,a2), dy(a1,a2), 0, distance, 1, 0)
            # landmark 1 for agent 1: (dx(a1,pred_l1), dy(a1,pred_l1), l1_depth, slant, 0, 0)
            num_entities = world.num_agents+world.num_landmarks
            num_features = 6

            feats = np.zeros(num_features*num_entities)
            
            # then other agents 
            i = 0
            for a in world.agents:
                if a is agent:
                    feats[i:i+num_features] = [agent.state.p_pos[0], agent.state.p_pos[1], 0., 0., 1., 1.]
                else:
                    dist  = world.entity_distances[f'{agent.name}-{a.name}']
                    # include other agent observations only if they are in the sense range and comm channel not dropped
                    if dist < agent.s_range and not(world.comm_drops[f'{agent.name}-{a.name}']):
                        pos = a.state.p_pos - agent.state.p_pos
                    else:
                        pos = np.zeros(2)
                        dist = 0
                    feats[i:i+num_features] = [pos[0],pos[1],0.,dist,1.,0.]
                i += num_features

            # and finally, landmarks
            for j, landmark in enumerate(world.landmarks):
                slant = agent.slants[j]
                # include the prediction and the slant in the observation only if communication channel not dropped
                if slant < world.max_range and not(world.comm_drops[f'{agent.name}-{landmark.name}']):
                    pos = agent.landmark_predictions[j] - agent.state.p_pos   
                else:
                    pos  = np.zeros(2)
                    slant = 0
                feats[i:i+num_features] = [pos[0],pos[1],landmark.landmark_depth/1000.,slant, 0., 0.]
                i += num_features

            return feats

        else:
            """Spread approach"""
            # get positions of all entities in this agent's reference frame
            entity_pos = np.zeros((world.num_landmarks, world.dim_p))
            entity_range = np.full(world.num_landmarks, world.max_range)
            entity_depth = np.zeros(world.num_landmarks)
            for i, landmark in enumerate(world.landmarks):
                
                entity_pos[i] = agent.landmark_predictions[i] - agent.state.p_pos
                entity_depth[i] = landmark.landmark_depth/1000.
                
                slant = agent.slants[i]
                # include the slant in the observation only if communication channel not dropped
                if slant < world.max_range or world.comm_drops[f'{agent.name}-{landmark.name}']:
                    entity_range[i] = slant


            if self.only_closer_agent_pos:
                best_dist = np.inf
                #comm_aux = np.zeros((world.num_landmarks, world.dim_p))
                other_pos = np.zeros(world.dim_p)

                for other in world.agents:
                    if other is not agent:

                        dist = world.entity_distances[f'{agent.name}-{other.name}']
                        # include other agent observations only if they are in the sense range and comm channel not dropped
                        if dist < best_dist and dist < agent.s_range and not(world.comm_drops[f'{agent.name}-{other.name}']):
                            #comm_aux = other.landmark_predictions - other.state.p_pos
                            other_pos = other.state.p_pos - agent.state.p_pos
                            best_dist = dist

                return np.concatenate((agent.state.p_vel, agent.state.p_pos, entity_pos.ravel(), other_pos, entity_range, entity_depth))
            else:
                """Spread approach augmented with all the agents positions"""
                other_pos = np.zeros((world.num_agents-1, world.dim_p))
                i = 0
                for other in world.agents:
                    if other is not agent:
                        dist = world.entity_distances[f'{agent.name}-{other.name}']
                        # include other agent observations only if they are in the sense range and comm channel not dropped
                        if dist < agent.s_range and not(world.comm_drops[f'{agent.name}-{other.name}']):
                            #comm_aux = other.landmark_predictions - other.state.p_pos
                            other_pos[i] = other.state.p_pos - agent.state.p_pos
                        i += 1

                return np.concatenate((agent.state.p_vel, agent.state.p_pos, entity_pos.ravel(), other_pos.ravel(), entity_range, entity_depth))


                
    
    def done(self):
        return self.done_state




"""
def state(self, world):
#Get the position, velocity and depth of each entity in the current state
s = np.zeros(world.num_agents*4 + world.num_landmarks*5)
i = 0
for agent in world.agents:
    s[i:i+2] = agent.state.p_pos
    s[i+2:i+4] = agent.state.p_vel
    i += 4
for landmark in world.landmarks:
    s[i:i+2] = landmark.state.p_pos
    s[i+2:i+4] = landmark.state.p_vel
    s[i+4] = landmark.landmark_depth/1000.
    i += 5
return s


# Reward based on the assumption that an agent can be reward only if it's close to a single entity
# first the min distance is computed for each landmark, with no repetitions of agents
# watch out: assumes that each agent should follow a single landmark (they are even)
agent_land_dist = {k: v for k, v in world.entity_distances.items() if 'agent' in k and 'landmark' in k}
sorted_dist     = {k: v for k, v in sorted(agent_land_dist.items(), key=lambda item: item[1])}
checked_agents  = set()
min_distances   = {}
for k, v in sorted_dist.items():
    a, l = k.split('-')
    if a not in checked_agents and l not in min_distances.keys():
        min_distances[l] = v
        checked_agents.add(a)


rew = 0.
for i, landmark in enumerate(world.landmarks):

    # positive reward if the best predictions of each agent is below the err threshold 
    min_err = min([euclidean_distance(agent.landmark_predictions[i], landmark.state.p_pos) for agent in world.agents])
    if min_err > self.rew_err_th:
        rew += 0.01*(0.004-min_err)

    # positive reward if the closest agent to this landmark is below the err threshold 
    min_dist = min_distances[landmark.name]
    if min_dist < self.rew_dis_th: 
        rew += 1.
    else: 
        rew += 0.01*(0.7-min_dist)

    # negative reward if the closest agent is so far from it that cannot recieve signals
    if min_dist > np.sqrt(self.max_range**2-(landmark.landmark_depth/1000.)**2):
        rew -= 0.1
    elif min_dist > self.max_range*2:
        rew -= 100.
        self.done_state = True
        self.agent_outofworld += 1
    
    # negative reward if the closest agent is colliding to the landmark
    if min_dist < 0.02: #is collision
        rew -= 10.
        self.done_state = True
        self.landmark_collision += 1

# check for agent collisions
for i in range(len(world.agents)-1):
    #if world.agents[].collide:
    for j in range(i+1, len(world.agents)):
        if self.is_collision(world.agents[i], world.agents[j], world):
            rew -= 10.
            self.agent_collision += 1
            self.done_state = True
return rew
"""


"""
def reward(self, agent, world):
        global done_state
        done_state = False
        # Agents are rewarded based on landmarks_estimated covariance_vals, penalized for collisions
        rew = 0.
        
        for i,l in enumerate(agent.landmarks_estimated):
            
            try:
                if self.pf_method == True:
                    world.cov[i] = np.sqrt((l.pf.covariance_vals[0])**2+(l.pf.covariance_vals[1])**2)/10.
                    predicted_position = np.array([l.pfxs[0],l.pfxs[2]]) #Using PF
                else:
                    predicted_position = np.array([l.lsxs[-1][0],l.lsxs[-1][2]])
            except:
                predicted_position = agent.state.p_pos
            
            error = euclidean_distance(predicted_position, world.landmarks[i].state.p_pos)
            
            #REWARD: Based on target estimation error, for each target
            if error<self.rew_err_th:
                rew += 1.
            else:
                rew += 0.01*(0.004-error)
            
            #REWARD: Based on the landmark, if there is at least an agent close to it -> reward
            min_dist = min([world.entity_distances[f'{a.name}-{world.landmarks[i].name}'] for a in world.agents])
            if min_dist < self.rew_dis_th: #other tests
                rew += 1.
            else:
                rew += 0.01*(0.7-min_dist)

        # compute the distance` between the agent and the closest landmark   
        min_dist = min([world.entity_distances[f'{agent.name}-{l.name}'] for l in world.landmarks])

        if min_dist > self.max_range:
            rew -= 1
        elif min_dist > self.max_range*2:
            rew -= 100
            done_state = True
            self.agent_outofworld += 1


        if min_dist > self.max_range: #agent out of range
            dist_from_origin = euclidean_distance(agent.state.p_pos, agent.state.p_pos_origin)
            if dist_from_origin > self.max_range*2. : #agent outside the world
                rew -= 100
                done_state = True
                self.agent_outofworld += 1
            else:
                rew -= 0.1
        else:
            # the agent is close to the target, and therefore, we save its position as new origin.
            agent.state.p_pos_origin = agent.state.p_pos.copy()
            
        if min_dist < 0.02: #is collision
            rew -= 1.
            done_state = True
            self.landmark_collision += 1
        
        if agent.collide:
            for a in world.agents:
                if a is agent: continue
                if self.is_collision(a, agent):
                    rew -= 10.
                    self.agent_collision += 1
                    done_state = True
        return rew
"""