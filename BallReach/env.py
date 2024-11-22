import time
import pygame
import numpy as np

pygame.init()

class BallReach:
    def __init__(self, size):
        self.nodeLoc = {}
        self.startLoc = [360, 50]
        self.goalLoc = [360, 950]
        self.playerLoc = [[125, 210], [525, 210], [205, 450], [465, 450], [120, 700], [550, 700]]
        self.state_idx = 0
        self.state = self.startLoc
        self.trajectory = {
            'observations': [],
            'actions': [],
            'next_observations': [],
            'rewards': [],
            'terminals': []
        }

        self.screen = pygame.display.set_mode(size=size)
        pygame.display.set_caption("BallReach")
        self.clock = pygame.time.Clock()
        self.running = True
        self.speed = 20
        self.time_factor = 100

        self.background = pygame.image.load("ground.jpg").convert()
        self.background = pygame.transform.scale(self.background, (720, 1000))

        self.ball = pygame.image.load("ball.png")
        self.ball = pygame.transform.scale(self.ball, (30, 30))

        self.player = pygame.image.load("player.png")
        self.player = pygame.transform.scale(self.player, (50, 50))
    
    def render(self):
        pos = pygame.Vector2((self.screen.get_height()-200)/5, (self.screen.get_width()-200)/5)

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
            
            self.screen.fill((0, 200, 40))
            self.screen.blit(self.background, (0, 0))
            
            pygame.draw.circle(self.screen, "red", [648, 50], 10)
            pygame.draw.circle(self.screen, "green", [648, 950], 10)

            self.step(state=self.state)
            
            pygame.display.flip()
            self.clock.tick(120)
            time.sleep(0.05)

        pygame.quit()
    
    def action(self):
        if np.random.rand() < 0.6:
            dx = 0
        else:
            dx = np.random.uniform(-1, 1)
        dy = np.random.uniform(0.3, 0.8)
        return np.array([dx, dy])

    def step(self, state):
        action = self.action()
        next_state = state + action*self.time_factor

        next_state[0] = np.clip(next_state[0], 0, self.screen.get_width())
        next_state[1] = np.clip(next_state[1], 0, self.screen.get_height())
        
        reward = self.reward(action)
        terminal = next_state[1] > 920

        if terminal:
            reward += 1
            print("Reached Goal")
            self.running = False

        self.trajectory['observations'].append(state.copy())
        self.trajectory['actions'].append(action)
        self.trajectory['next_observations'].append(next_state.copy())
        self.trajectory['rewards'].append(reward)
        self.trajectory['terminals'].append(terminal)

        print(f'State: {self.state}, Action: {action}, Next State: {next_state}, Reward: {reward}')
        
        state_list = [float(coord) for coord in state]
        next_state_list = [float(coord) for coord in next_state]

        self.move_circle(state_list, next_state_list)
        self.state = next_state

    
    def reward(self, action):
        dx, dy = action
        reward = -abs(dx)
        return reward
    
    def move_circle(self, start, end):
        start_vec = pygame.Vector2(start)
        end_vec = pygame.Vector2(end)
        direction = (end_vec - start_vec).normalize()
        distance = start_vec.distance_to(end_vec)

        steps = int(distance/ self.speed)
        for i in range(steps):
            current_pos = start_vec + direction*(i*self.speed)
            self.screen.fill((0, 200, 40))
            self.screen.blit(self.background, (0, 0))

            pygame.draw.circle(self.screen, "red", self.startLoc, 10)
            pygame.draw.circle(self.screen, "green", self.goalLoc, 10)

            for player in self.playerLoc:
                self.screen.blit(self.player, (player[0], player[1]))

            # pygame.draw.circle(self.screen, "white", (int(current_pos.x), int(current_pos.y)), 10)
            self.screen.blit(self.ball, (int(current_pos.x), int(current_pos.y)))
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
        gr = BallReach(size=(720, 1000))
        gr.render()

        dataset['observations'].extend(gr.trajectory['observations'])
        dataset['actions'].extend(gr.trajectory['actions'])
        dataset['next_observations'].extend(gr.trajectory['next_observations'])
        dataset['rewards'].extend(gr.trajectory['rewards'])
        dataset['terminals'].extend(gr.trajectory['terminals'])

    for key in dataset:
        dataset[key] = np.array(dataset[key])
        
    np.savez_compressed('ballreach_dataset_continous.npz', **dataset)
    print(f"Dataset of {num_demo} demonstrations saved to 'ballreach_dataset.npz'.")

if __name__ == "__main__":
    generate_dataset(250)
    # gr = BallReach(size=(720, 1000))
    # gr.render()