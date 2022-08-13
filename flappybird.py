import pygame
import neat
import os
import time
import random

pygame.font.init()
gen=0
WINDOW_HEIGHT = 800
WINDOW_WIDTH = 500
STAT_FONT = pygame.font.SysFont("comicsans", 50)




# To animate the bird flapping we take the images of bird in different positions


CHARACTER_IMAGES = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
                    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))),
                    pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]
OBSTCALES_IMAGES = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
GROUND_IMAGES = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BACKGROUND_IMAGES = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))


class Character:
    IMG = CHARACTER_IMAGES
    max_rotation = 25
    rotation_velocity = 20
    animation_time = 5

    def __init__(self, x, y):
        self.x = x         # x-coordinate
        self.y = y         # y-coordinate
        self.tilt = 0
        self.vel = 0
        self.tick_count = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMG[0]

    def jump(self):
        self.vel = -10.5     # Neagtive velocity for moving upwards because in pyGame origin is at Top-left
        self.tick_count = 0 
        self.height = self.y

    def mov(self):
        self.tick_count += 1

        d = self.vel * self.tick_count + 1.5 * self.tick_count ** 2  # Displacement
        if d > 16:     # This is a Fail-safe if the bird is moving more that 16 pixels up or down we make saturate to this value
            d = 16     # Terminal Velocity
        if d < 0:
            d -= 2
        self.y = self.y + d
        if d < 0 or self.y < self.height + 50:
            if self.tilt < self.max_rotation:
                self.tilt = self.max_rotation
        else:
            if self.tilt > -90:
                self.tilt -= self.rotation_velocity

    def draw(self, win):
        self.img_count += 1
        if self.img_count < self.animation_time:
            self.img = self.IMG[0]
        elif self.img_count < self.animation_time * 2:
            self.img = self.IMG[1]
        elif self.img_count < self.animation_time * 3:
            self.img = self.IMG[2]
        elif self.img_count < self.animation_time * 4:
            self.img = self.IMG[1]
        elif self.img_count == self.animation_time * 4 + 1:
            self.img = self.IMG[0]
            self.img_count = 0
        if self.tilt <= -80:
            self.img = self.IMG[1]
            self.img_count = self.animation_time * 2
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):
        return pygame.mask.from_surface(self.img)


class Obstacles:
    GAP = 200
    vel = 5

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.top = 0
        self.bottom = 0
        self.top_obstacle = pygame.transform.flip(OBSTCALES_IMAGES, False, True)
        self.bottom_obstacle = OBSTCALES_IMAGES
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)
        self.top = self.height - self.top_obstacle.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x -= self.vel

    def draw(self, win):
        win.blit(self.top_obstacle, (self.x, self.top))
        win.blit(self.bottom_obstacle, (self.x, self.bottom))
 

    def collide(self, bird):
        bird_mask = bird.get_mask()
        top_obstacle_mask = pygame.mask.from_surface(self.top_obstacle)
        bottom_obstacle_mask = pygame.mask.from_surface(self.bottom_obstacle)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))
        b_point = bird_mask.overlap(bottom_obstacle_mask, bottom_offset)
        t_point = bird_mask.overlap(top_obstacle_mask, top_offset)
        if b_point or t_point:
            return True
        return False


class Base:
    vel = 5  # SAME VELOCITY AS PIPE VELOCITY
    width = GROUND_IMAGES.get_width()
    img = GROUND_IMAGES

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.width

    def move(self):
        self.x1 -= self.vel
        self.x2 -= self.vel
        if self.x1 + self.width < 0:
            self.x1 = self.x2 + self.width
        if self.x2 + self.width < 0:
            self.x2 = self.x1 + self.width

    def draw(self, win):
        win.blit(self.img, (self.x1, self.y))
        win.blit(self.img, (self.x2, self.y))


def draw_window(win, birds, obstacles, base, score,gen):
    win.blit(BACKGROUND_IMAGES, (0, 0))
    for pipe in obstacles:
        pipe.draw(win)
    text = STAT_FONT.render("Score: " + str(score), 1, (166, 0, 228))
    win.blit(text, (WINDOW_WIDTH - 10 - text.get_width(), 10))
    text = STAT_FONT.render("GEN: " + str(gen), 1, (166, 0, 228))
    win.blit(text, (10, 10))
    base.draw(win)
    for bird in birds:
        bird.draw(win)
    pygame.display.update()


def main(genomes, config):
    global gen
    gen+=1
    birds = []
    ge = []
    nets = []
    # bird=Character(230,350)
    for __, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        birds.append(Character(230, 350))
        g.fitness = 0
        ge.append(g)
    base = Base(730)
    pipes = [Obstacles(600)]
    win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    run = True
    score = 0
    clock = pygame.time.Clock()
    while run:
        clock.tick(40)
        rem = []
        add_pipe = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].top_obstacle.get_width():
                pipe_ind = 1
        else:
            run = False
            break

        for x, bird in enumerate(birds):  # give each bird a fitness of 0.1 for each frame it stays alive
            ge[x].fitness += 0.1
            bird.mov()
            output = nets[birds.index(bird)].activate(
                (bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))
            if output[0] > 0.5:
                bird.jump()
            base.move()
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    ge[x].fitness -= 1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)
                if not pipe.passed and pipe.x < bird.x:
                    pipe.passed = True
                    add_pipe = True
            if pipe.x + pipe.top_obstacle.get_width() < 0:
                rem.append(pipe)
            pipe.move()

        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Obstacles(700))
        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height()-10 >730 or bird.y < -50:
                ge[x].fitness -= 1
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)
        for r in rem:
            pipes.remove(r)


        draw_window(win, birds, pipes, base, score,gen)


def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    winner = p.run(main, 50)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config-feedforward.txt")
    run(config_path)
