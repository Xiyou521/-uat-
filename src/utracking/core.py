import re
import numpy as np
from utracking.tracking.target_pf import random_levy

def euclidean_distance(e1_pos, e2_pos):
    return np.sqrt(np.sum(np.square(e1_pos - e2_pos)))

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        self.p_pos_origin = None
        # physical velocity
        self.p_vel = None
        self.p_vel_old = None
        self.a_vel = None
        self.vel = None

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        self.max_a_speed = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()
        # action
        self.action = Action()
        # physical motor noise amount
        self.u_noise = None
        # velocity,direction,depth
        self.landmark_vel = 0.
        self.ra = 0.
        self.ra_sign = 1. # sign of the change of direcion
        self.landmark_depth = 0.
        # landmark type of movement
        self.movement = 'linear'
        # for estimated landmarks, keep track of the closest agent
        self.best_slant = np.inf

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # sense range
        self.s_range = 0.5
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # each agent has a target algorithm for each landmark
        self.landmarks_estimated = []
        # each agent predicts the position of the landmarks
        self.landmark_predictions = []
        # each agent estimates every landmarks positions
        self.slants = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.landmarks_best_estimations = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.1 #0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        # if world is collaborative
        self.num_agents = 3
        self.num_landmarks = 3
        self.collaborative = True
        self.angle = []
        # sea currents
        self.vel_ocean_current = 0.
        self.angle_ocean_current = 0.
        # distances and drop communication between entities
        self.entity_distances = {}
        self.min_distances = {}
        self.comm_drops = {}
        # tracking settings
        self.pf_method = False
        self.range_droping = 0.
        self.max_range = 1.

        # current step
        self.t = 0

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set action for landmarks and intialize best slant to inf
        for i in range(self.num_landmarks):
            self.set_landmark_action(self.landmarks[i])
            self.landmarks_best_estimations[i].best_slant = np.inf
        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update entities distances
        self.update_distances()
        # update agent state (communication and slants)
        for agent in self.agents:
            self.update_agent_state(agent)
        for agent in self.agents:
            self.update_agent_predictions(agent)
        self.t += 1

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,entity in enumerate(self.entities):
            if 'agent' in entity.name:
                if entity.movable:
                    noise = np.random.randn(*entity.action.u.shape) * entity.u_noise if entity.u_noise else 0.0
                    p_force[i] = entity.action.u + noise 
            if 'landmark' in entity.name:
                if entity.movable:
                    noise = np.random.randn(*entity.action.u.shape) * entity.u_noise if entity.u_noise else 0.0
                    p_force[i] = entity.action.u + noise 
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            
            if 'landmark' in entity.name:
                #if entity is a landmark (x-y force applyied independently)
                if (p_force[i] is not None):
                    entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
                # entity.max_speed = .3
                if entity.max_speed is not None:
                    speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                    if speed > entity.max_speed:
                        entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                      np.square(entity.state.p_vel[1])) * entity.max_speed
                entity.state.p_pos += entity.state.p_vel * self.dt
            
            if 'agent' in entity.name:
                #if entity is an agnet (constant velocity, increment of angle)
                '''This is the new approach designed by Ivan'''
                #First position of p_vel is the angular velocity which is used to increase the angle of the agent
                if (p_force[i] is not None):
                    # entity.state.a_vel += (p_force[i].item(0) / entity.mass) * self.dt 
                    # if entity.max_a_speed is not None:
                    #     if entity.state.a_vel > entity.max_a_speed:
                    #         entity.state.a_vel = entity.max_a_speed + 0.
                    #     if entity.state.a_vel < -entity.max_a_speed:
                    #         entity.state.a_vel = -entity.max_a_speed + 0.
                    #increment angle based on angle velocity: New improvement for science
                    # self.angle[i] += entity.state.a_vel * self.dt
                    #increment angle based on a simple inc: Old way for CASE and IROS
                    self.angle[i] += p_force[i].item(0)*0.3 #multiply by 0.1 to set radius limit at 100m minimum
                    if self.angle[i] > np.pi*2.:
                        self.angle[i] -= np.pi*2.
                    if self.angle[i] < -np.pi*2:
                        self.angle[i] += np.pi*2
                #The second position of p_vel is the liniar velocity of the agent
                # vel = entity.state.p_vel[1]+0.
                vel = entity.state.vel #seting this velocity (0.1) and considering that the dt is equal to 0.1, means that we have a new position each 10m.
                #Finally, we increase the position (aka the agent) using the new angle and velocity.
                pos_error = False
                if pos_error == True:
                    error_in_m = 3
                    add_error = np.random.randn(2)*error_in_m/1000.
                else:
                    add_error = 0.
                entity.state.p_pos += np.array([vel*np.cos(self.angle[i]),vel*np.sin(self.angle[i])]) * self.dt + add_error
                # entity.state.p_pos += np.array([entity.state.p_vel.item(0),entity.state.p_vel.item(1)]) * self.dt + add_error
                entity.state.p_vel = np.array([vel*np.cos(self.angle[i]),vel*np.sin(self.angle[i])])    
                ocean_current = True
                if ocean_current == True:
                    entity.state.p_pos += np.array([self.vel_ocean_current*np.cos(self.angle_ocean_current),self.vel_ocean_current*np.sin(self.angle_ocean_current)]) * self.dt
                    # we don't need to modify the vel of the agent, in that way, we conserve its real direction. And not the direction of the current,
                    # with can be extrapolated from the position of the agent, its next position, and its direction.
                    # entity.state.p_vel += np.array([self.vel_ocean_current*np.cos(self.angle_ocean_current),self.vel_ocean_current*np.sin(self.angle_ocean_current)]) 
                
    def update_agent_state(self, agent):        

        for i, landmark in enumerate(self.landmarks):
            slant_range = self.entity_distances[f'{landmark.name}-{agent.name}']
            target_depth = landmark.landmark_depth/1000. #normalize the target depth
            slant_range = np.sqrt(slant_range**2+target_depth**2) #add target depth to the range measurement
            # Add some systematic error in the measured range
            slant_range *= 1.01 # where 0.99 = 1% of sound speed difference = 1495 m/s
            # Add some noise in the measured range
            slant_range += np.random.uniform(-0.001, +0.001)
            # Return to a planar range
            slant_range = np.sqrt(abs(slant_range**2-target_depth**2))
            agent.slants[i] = slant_range

        """
        # communication are the target predictions
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise
        """

    def update_agent_predictions(self, agent):

        # update the target estimators and predicted positions of the landmarks
        for i, landmark in enumerate(self.landmarks):

            # get the target estimator
            l = agent.landmarks_estimated[i]

            #1: update the predictions using the agent slant and the slants of the closest agents
            update = False
            for other in self.agents:
                # always update for the current agent
                if other is agent:
                    slant_range = agent.slants[i]
                    valid_range = True
                    curr_agent = agent
                # for the other agents, update if they are into the sense range of the agent
                else: 
                    dist = self.entity_distances[f'{agent.name}-{other.name}']
                    slant_range = other.slants[i]
                    if dist < agent.s_range and slant_range < agent.slants[i]*1.5:
                        valid_range = True
                        curr_agent = other
                    else: 
                        valid_range = False

                # the range is valid only if it is in the max range and its not dropped
                valid_range = valid_range and (
                    slant_range < self.max_range and not (self.comm_drops[f'{curr_agent.name}-{landmark.name}'] or self.comm_drops[f'{agent.name}-{curr_agent.name}'])
                )

                if valid_range:
                    update = True
                    if self.pf_method == True:
                        #2a: Update the PF
                        # TODO: modify PF so that it gets update only once as LS
                        add_pos_error = False
                        if add_pos_error == True:
                            l.updatePF(dt=0.04, new_range=True, z=slant_range, myobserver=[curr_agent.state.p_pos[0]+np.random.randn(1).item(0)*3/1000.,0.,curr_agent.state.p_pos[1]+np.random.randn(1).item(0)*3/1000.,0.], update=True)
                        else:
                            l.updatePF(dt=0.04, new_range=True, z=slant_range, myobserver=[curr_agent.state.p_pos[0],0.,curr_agent.state.p_pos[1],0.], update=True)
                    else:
                        #2b: Add the slant to the target estimator
                        l.add_range(z=slant_range, myobserver=curr_agent.state.p_pos)

            # if at least a range was inserted, update the model
            if update and self.pf_method==False:
                l.update(myobserver=agent.state.p_pos, dt=0.04)

            try:
                if self.pf_method == True:
                    predicted_position = np.array([l.pfxs[0],l.pfxs[2]]) #Using PF
                else:
                    predicted_position = l.get_pred() #Using LS
            except:
                #An error will be produced if its the initial time and no good range measurement has been conducted yet. In this case, we supose that the target 
                #is at the same position of the agent.
                predicted_position = agent.state.p_pos.copy()

            agent.landmark_predictions[i] = predicted_position

            # update the best estimations if this agent is the closest to the landmark
            if agent.slants[i] < self.landmarks_best_estimations[i].best_slant and not(self.comm_drops[f'{curr_agent.name}-{landmark.name}']):
                self.landmarks_best_estimations[i].best_slant = agent.slants[i]
                self.landmarks_best_estimations[i].state.p_pos = predicted_position
                self.error[i] = euclidean_distance(landmark.state.p_pos, predicted_position)
            

    def set_landmark_action(self, landmark):
        # Update the action if the landmark is movable
        if landmark.movable:
            if landmark.movement == 'linear':
                #linear movement
                u_force = landmark.landmark_vel
                landmark.action.u = np.array([np.cos(landmark.ra)*u_force,np.sin(landmark.ra)*u_force])

            elif landmark.movement == 'turning':
                u_force = landmark.landmark_vel
                if self.t > 100 and self.t < 150 and np.random.random()<0.05: # modify direction with prob 0.1 between 80 and 150th time steps
                    landmark.ra += landmark.ra_sign*np.random.uniform(0, 0.4)
                landmark.action.u = np.array([np.cos(landmark.ra)*u_force,np.sin(landmark.ra)*u_force])
            
            elif landmark.movement == 'random':
                # random movement
                landmark.action.u = np.random.randn(2)/2.
            
            elif landmark.movement == 'levy':
                #random walk Levy movement
                beta = 1.9 #must be between 1 and 2
                landmark.action.u = random_levy(beta)
                if landmark.state.p_pos[0] > 0.8:
                    landmark.action.u[0] = -abs(landmark.action.u[0])
                if landmark.state.p_pos[0] < -0.8:
                    landmark.action.u[0] = abs(landmark.action.u[0])
                if landmark.state.p_pos[1] > 0.8:
                    landmark.action.u[1] = -abs(landmark.action.u[1])
                if landmark.state.p_pos[1] < -0.8:
                    landmark.action.u[1] = abs(landmark.action.u[1])

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]


    
    def update_distances(self):
        """Compute the distances between the world entities and the closest agent to each landmark"""

        def euclidean_distance(e1_pos, e2_pos):
            return np.sqrt(np.sum(np.square(e1_pos - e2_pos)))

        """Update the distances between the entities"""
        for i in range(len(self.entities)-1):
            # default (diagonal) 
            self.entity_distances[f'{self.entities[i].name}-{self.entities[i].name}'] = 0.
            self.comm_drops[f'{self.entities[i].name}-{self.entities[i].name}'] = False

            for j in range(i+1, len(self.entities)):

                # distances 
                dist = euclidean_distance(self.entities[i].state.p_pos, self.entities[j].state.p_pos)
                self.entity_distances[f'{self.entities[i].name}-{self.entities[j].name}'] = dist
                self.entity_distances[f'{self.entities[j].name}-{self.entities[i].name}'] = dist

                # communication drops
                drop = np.random.rand() < self.range_droping
                self.comm_drops[f'{self.entities[i].name}-{self.entities[j].name}'] = drop
                self.comm_drops[f'{self.entities[j].name}-{self.entities[i].name}'] = drop


        """Update the closest landmark to each agent (without repetition)"""
        # First the min distance is computed for each landmark, with no repetitions of agents
        # watch out: assumes that each agent should follow a single landmark (they are even)
        agent_land_dist = {k: v for k, v in self.entity_distances.items() if re.match('agent [0-9]+-landmark [0-9]+', k)}
        sorted_dist     = {k: v for k, v in sorted(agent_land_dist.items(), key=lambda item: item[1])}
        checked_agents  = set()
        self.min_distances   = {}
        for k, v in sorted_dist.items():
            a, l = k.split('-')
            if a not in checked_agents and l not in self.min_distances.keys():
                self.min_distances[l] = v
                checked_agents.add(a)
