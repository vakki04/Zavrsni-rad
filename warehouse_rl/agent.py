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
        # Initialize Q-value table: maps (state, action) pairs to estimated rewards
        self.Q = np.zeros((n_states, n_actions))
        self.n_actions = n_actions
        # Epsilon controls exploration vs exploitation (starts at 1.0, decreases over time)
        self.epsilon = EPS_START
        # Track number of completed episodes
        self.episode = 0
        # Store metrics for analysis and visualization
        self.rewards_history = []     # Total reward per episode
        self.steps_history   = []     # Number of steps per episode
        self.epsilon_history = []     # Exploration rate over time
        self.success_history = []     # Whether episode reached goal (0 or 1)
        self.avg100 = []              # Running average of last 100 rewards
        self.recent = deque(maxlen=100)  # Last 100 episode rewards for averaging

    def choose_action(self, state):
        """Epsilon-greedy action selection: explore (random) or exploit (best known)."""
        # With probability epsilon, pick a random action (exploration)
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        # Otherwise, pick the action with highest Q-value for this state (exploitation)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s2, done):
        """Update Q-value using the Q-Learning equation: Q(s,a) ← Q(s,a) + α(r + γ*max(Q(s',a')) - Q(s,a))"""
        # Calculate target value: immediate reward + discounted future reward (if not terminal state)
        target = r if done else r + GAMMA * np.max(self.Q[s2])
        # Update Q-value by moving towards the target, scaled by learning rate ALPHA
        self.Q[s, a] += ALPHA * (target - self.Q[s, a])

    def decay_epsilon(self):
        """Reduce exploration rate gradually: encourage exploitation of learned knowledge over time."""
        # Multiply epsilon by decay rate (0.997), ensuring it doesn't go below minimum threshold
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)

    def run_episode(self, pickup_pos, grid, actions, state_id, valid, rewards):
        """
        Execute one complete training episode: navigate from start to goal.
        
        Process: 1) Pick up item at pickup_pos, 2) Deliver to goal
        The agent learns which actions lead to higher rewards in each state.
        
        Args:
            pickup_pos: (row, col) position of item to collect
            grid: warehouse map (0=walkable, 1=obstacle)
            actions: list of action deltas [(dr,dc), ...] for [up,down,left,right]
            state_id: function to convert (row, col, has_item) to unique state ID
            valid: function checking if position is walkable and in bounds
            rewards: dict with reward values for different events
        
        Returns:
            (total_reward, steps_taken, success) - success=True if reached goal with item
        """
        from config import R_PICKUP, R_GOAL, R_OBSTACLE, R_STEP, R_REVISIT, MAX_STEPS
        
        # Initialize episode: start at position (1,1), goal at bottom-right
        start = (1, 1)
        goal = (grid.shape[0] - 2, grid.shape[1] - 2)
        
        # Reset agent state for new episode
        r, c = start
        has_item = False
        s = state_id(r, c, has_item)  # Encode current state
        visited = set([s])  # Track visited states to penalize revisits
        total_reward = 0.0
        success = False
        step = 0

        # Execute up to MAX_STEPS actions in this episode
        for step in range(MAX_STEPS):
            # Choose action using epsilon-greedy strategy
            a = self.choose_action(s)
            dr, dc = actions[a]
            nr, nc = r + dr, c + dc

            # Check if next position is valid (not a wall, in bounds)
            if not valid(nr, nc):
                # Hit an obstacle: penalize but stay in current position
                reward = rewards['obstacle']
                ns = s
                nr, nc = r, c  # Don't move
            else:
                # Valid move: check what's at this position
                if (nr, nc) == pickup_pos and not has_item:
                    # Pick up the item
                    has_item = True
                    reward = rewards['pickup']
                elif (nr, nc) == goal and has_item:
                    # Reached goal WITH item: episode successful!
                    reward = rewards['goal']
                    ns = state_id(nr, nc, has_item)
                    self.update(s, a, reward, ns, True)  # Terminal state
                    total_reward += reward
                    success = True
                    step += 1
                    break  # Episode complete
                elif (nr, nc) == goal and not has_item:
                    # Reached goal WITHOUT item: penalize (must retry)
                    reward = rewards['obstacle']
                    ns = s
                    nr, nc = r, c  # Don't move
                else:
                    # Normal movement: small penalty for each step (encourages efficiency)
                    reward = rewards['step']

                # Create new state ID and calculate composite reward
                ns = state_id(nr, nc, has_item)
                if ns in visited:
                    # Penalize revisiting same position (encourages exploration)
                    reward += rewards['revisit']
                visited.add(ns)

            # Learn from this transition: update Q-value
            self.update(s, a, reward, ns, False)  # Not terminal (unless goal+item reached above)
            total_reward += reward
            # Move to next state for next iteration
            s, r, c = ns, nr, nc

        # Episode complete: update learning statistics
        self.decay_epsilon()  # Gradually shift towards exploitation
        self.episode += 1
        self.rewards_history.append(total_reward)
        self.steps_history.append(step + 1)
        self.epsilon_history.append(self.epsilon)
        self.success_history.append(1 if success else 0)
        self.recent.append(total_reward)
        self.avg100.append(np.mean(self.recent))  # 100-episode moving average

        return total_reward, step + 1, success

    def get_best_path(self, pickup_pos, grid, actions, state_id, valid):
        """
        Extract the greedy path (highest Q-values at each step) after training is complete.
        
        This shows what the agent learned: the best route it discovered to pick up 
        the item and deliver to goal. Uses learned Q-values (no random exploration).
        
        Args:
            pickup_pos: (row, col) position of item to collect
            grid: warehouse map (0=walkable, 1=obstacle)
            actions: list of action deltas [(dr,dc), ...]
            state_id: function to convert (row, col, has_item) to state ID
            valid: function checking if position is walkable and in bounds
        
        Returns:
            List of (row, col) coordinates representing the optimal path
        """
        from config import MAX_STEPS
        
        start = (1, 1)
        goal = (grid.shape[0] - 2, grid.shape[1] - 2)
        
        # Start new episode
        r, c = start
        has_item = False
        path = [(r, c)]  # Track the path taken
        visited = set([(r, c, has_item)])  # Prevent infinite loops
        
        # Follow greedy policy for up to MAX_STEPS
        for _ in range(MAX_STEPS):
            # Get current state and select best action (highest Q-value, no randomness)
            s = state_id(r, c, has_item)
            a = int(np.argmax(self.Q[s]))  # Always pick best action
            dr, dc = actions[a]
            nr, nc = r + dr, c + dc
            
            # Check if next move is valid
            if not valid(nr, nc):
                break  # Hit obstacle, path complete
            
            # Check if we're at the pickup location
            if (nr, nc) == pickup_pos and not has_item:
                has_item = True  # Pick up the item
            
            # Check for cycles (already visited this state)
            key = (nr, nc, has_item)
            if key in visited:
                break  # Stop if repeating states
            
            # Add to path and continue
            path.append((nr, nc))
            visited.add(key)
            r, c = nr, nc
            
            # Episode complete if reached goal with item
            if (r, c) == goal and has_item:
                break
        
        return path
