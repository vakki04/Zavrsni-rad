"""
Q-Learning agent implementation for warehouse navigation task.
"""

import numpy as np
import random
from collections import deque
from config import ALPHA, GAMMA, EPS_START, EPS_END, EPS_DECAY, MAX_STEPS

class QLearningAgent:
    """Q-Learning agent for autonomous warehouse navigation."""
    
    def __init__(self, n_states, n_actions):
        self.Q = np.zeros((n_states, n_actions))
        self.n_actions = n_actions
        self.epsilon = EPS_START
        self.episode = 0
        
        self.rewards_history = []
        self.steps_history   = []
        self.epsilon_history = []
        self.success_history = []
        self.avg100 = []
        self.recent = deque(maxlen=100)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s2, done):
        best_next = 0 if done else np.max(self.Q[s2])
        self.Q[s, a] += ALPHA * (r + GAMMA * best_next - self.Q[s, a])

    def decay_epsilon(self):
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    def run_episode(self, start, goal, pickup_pos, actions, state_id, valid, rewards):
        """Metoda koja provodi jednu epizodu treninga agenta."""
        r, c = start
        has_item = False
        s = state_id(r, c, has_item)
        visited = set([s])
        
        # Spremamo korake za simulaciju
        path_history = [(r, c, has_item)]
        total_reward = 0.0
        success = False
        step = 0

        for step in range(MAX_STEPS):
            a = self.choose_action(s)
            dr, dc = actions[a]
            nr, nc = r + dr, c + dc
            
            if not valid(nr, nc):
                reward = rewards['obstacle']
                ns = s
                nr, nc = r, c
            else:
                if (nr, nc) == pickup_pos and not has_item:
                    has_item = True
                    reward = rewards['pickup']
                elif (nr, nc) == goal and has_item:
                    reward = rewards['goal']
                    ns = state_id(nr, nc, has_item)
                    self.update(s, a, reward, ns, True)
                    success = True
                    total_reward += reward             
                    path_history.append((nr, nc, True))
                    break
                elif (nr, nc) == goal and not has_item:
                    reward = rewards['obstacle']
                    ns = s
                    nr, nc = r, c
                else:
                    reward = rewards['step']
                
                ns = state_id(nr, nc, has_item)
                if ns in visited:
                    reward += rewards['revisit']
                visited.add(ns)
                
            self.update(s, a, reward, ns, False)
            s, r, c = ns, nr, nc
            
            total_reward += reward
            path_history.append((r, c, has_item))

        self.decay_epsilon()
        self.episode += 1
        
        self.rewards_history.append(total_reward)
        self.steps_history.append(step + 1)
        self.epsilon_history.append(self.epsilon)
        self.success_history.append(1 if success else 0)
        self.recent.append(total_reward)
        self.avg100.append(np.mean(self.recent))

        return total_reward, step + 1, success, path_history

    def get_best_path(self, start, goal, pickup_pos, actions, state_id, valid):
        """Vraća najbolji put pronađen dosadašnjim učenjem."""
        from config import MAX_STEPS
        r, c = start
        has_item = False
        path = [(r, c)]
        visited = set([(r, c, has_item)])
        
        for _ in range(MAX_STEPS):
            s = state_id(r, c, has_item)
            a = int(np.argmax(self.Q[s]))
            dr, dc = actions[a]
            nr, nc = r + dr, c + dc
            
            if not valid(nr, nc): break
            if (nr, nc) == pickup_pos and not has_item: has_item = True
            
            key = (nr, nc, has_item)
            if key in visited: break
            
            path.append((nr, nc))
            visited.add(key)
            r, c = nr, nc
            if (r, c) == goal and has_item: break
        
        return path