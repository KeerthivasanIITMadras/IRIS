import time
import pygame
import numpy as np

pygame.init()

class GraphReach:
    def __init__(self, size):
        self.nodeLoc = {}
        self.startLoc = [648, 50]
        self.goalLoc = [648, 950]
        self.state_idx = 0
        self.state = self.startLoc
        self.lines = []
        self.centerLoc = {}
        self.centerFlag = 1
        self.choice = ['center', 'random']
        self.trajectory = {
            'observations': [],
            'actions': [],
            'next_observations': [],
            'rewards': [],
            'terminals': []
        }

        self.screen = pygame.display.set_mode(size=size)
        pygame.display.set_caption("GraphReach")
        self.clock = pygame.time.Clock()
        self.running = True
        self.speed = 20
    
    def render(self):
        pos = pygame.Vector2((self.screen.get_height()-200)/5, (self.screen.get_width()-200)/5)

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            self.screen.fill("black")
            nodes = []
            for i in range(1, 6):
                for j in range(1, 6):
                    if j == 3:
                        self.centerLoc[i] = ([j*pos[1], i*pos[0]])
                    nodes.append([j*pos[1], i*pos[0]])
                    pygame.draw.circle(self.screen, (0, 255, 255), [j*pos[1], i*pos[0]], 10)
                self.nodeLoc[i] = nodes
                nodes = []

            
            pygame.draw.circle(self.screen, "red", [648, 50], 10)
            pygame.draw.circle(self.screen, "green", [648, 950], 10)

            for line in self.lines:
                pygame.draw.line(self.screen, "yellow", line[0], line[1])

            self.step(state=self.state)
            
            pygame.display.flip()
            self.clock.tick(120)
            time.sleep(0.05)

        pygame.quit()
    
    def action(self):
        prob = np.random.choice(self.choice, 1, p=[0.7, 0.3])

        if prob == "center" and self.centerFlag != 6:
            next_state = self.centerLoc[self.centerFlag]
            next_idx = self.centerFlag
            self.centerFlag += 1
            
        elif prob == "random" and self.centerFlag != 6:
            li = list(range(self.centerFlag, 6))
            level_idx = np.random.choice(li, 1, p=[1/len(li)]*len(li))[0]
            while True:
                node_idx = np.random.choice([1, 2, 3, 4, 5], 1, p=[0.25, 0.25, 0., 0.25, 0.25])[0]
                next_state = self.nodeLoc[level_idx][node_idx-1]

                if next_state != self.state:
                    break
            next_idx = level_idx
        
        elif self.centerFlag == 6:
            next_idx = 6
            next_state = self.goalLoc

        return next_idx, next_state, prob

    def step(self, state):
        next_idx, next_state, prob = self.action()
        reward = self.reward(next_state)

        terminal = next_state == self.goalLoc

        self.trajectory['observations'].append(np.array(state))
        self.trajectory['actions'].append(self.choice.index(prob))
        self.trajectory['next_observations'].append(np.array(next_state))
        self.trajectory['rewards'].append(reward)
        self.trajectory['terminals'].append(terminal)

        print(f'State: {self.state}, Action: {prob}, Next State: {next_state}, Reward: {reward}, Transition: {[self.state_idx, next_idx]}')
        
        self.move_circle(state, next_state)

        self.lines.append((state, next_state))
        self.state_idx = next_idx
        self.state = next_state

        if self.state_idx == 6:
            print("Reached Goal")
            self.running = False
    
    def reward(self, state):
        if state in self.centerLoc.values():
            reward = 0
        elif state == self.goalLoc:
            reward = 1
        else:
            reward = 0
        return reward
    
    def move_circle(self, start, end):
        start_vec = pygame.Vector2(start)
        end_vec = pygame.Vector2(end)
        direction = (end_vec - start_vec).normalize()
        distance = start_vec.distance_to(end_vec)

        steps = int(distance/ self.speed)
        for i in range(steps):
            current_pos = start_vec + direction*(i*self.speed)
            self.screen.fill("black")

            for i in range(1, 6):
                for j in range(1, 6):
                    pygame.draw.circle(self.screen, (0, 255, 255), self.nodeLoc[i][j-1], 10)
            pygame.draw.circle(self.screen, "red", self.startLoc, 10)
            pygame.draw.circle(self.screen, "green", self.goalLoc, 10)

            pygame.draw.circle(self.screen, "white", (int(current_pos.x), int(current_pos.y)), 10)
            pygame.display.flip()
            self.clock.tick(120)
            time.sleep(0.05)

def generate_dataset(num_demo):
    dataset ={
        'observations': [],
        'actions': [],
        'next_observations': [],
        'rewards': [],
        'terminals': []
    }

    for _ in range(num_demo):
        gr = GraphReach(size=(1280, 1000))
        gr.render()

        dataset['observations'].extend(gr.trajectory['observations'])
        dataset['actions'].extend(gr.trajectory['actions'])
        dataset['next_observations'].extend(gr.trajectory['next_observations'])
        dataset['rewards'].extend(gr.trajectory['rewards'])
        dataset['terminals'].extend(gr.trajectory['terminals'])

    for key in dataset:
        dataset[key] = np.array(dataset[key])
        
    np.savez_compressed('graphreach_dataset.npz', **dataset)
    print(f"Dataset of {num_demo} demonstrations saved to 'graphreach_dataset.npz'.")

if __name__ == "__main__":
    generate_dataset(250)