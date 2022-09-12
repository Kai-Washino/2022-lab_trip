import pygame
import random
 
BLACK = (0, 0, 0)
RED = (255, 0, 0)
WHITE = (255, 255, 255)

pygame.init()
screen = pygame.display.set_mode((700, 480))
myclock = pygame.time.Clock()
 
flag=0
x_paddle=250
x_ball = 10
y_ball = 10
vector_x = 5
vector_y = 5
score = 0
while flag==0:
  for event in pygame.event.get():
    if event.type==pygame.QUIT: flag=1
  screen.fill(BLACK)
 
  #パドルを描画
  press = pygame.key.get_pressed()
  if(press[pygame.K_LEFT] and x_paddle>0): x_paddle-=5
  if(press[pygame.K_RIGHT] and x_paddle<600): x_paddle+=5
  rect = pygame.Rect(x_paddle, 400, 100, 30)
  pygame.draw.rect(screen, RED, rect)
   
  #ボールを描画
  if(y_ball==390 and x_ball>=(x_paddle-5) and x_ball<=(x_paddle+95)):
    vector_y = -5
    score += 1
  if(y_ball>=500):
    x_ball = random.randrange(700)
    y_ball = 10
    vector_x = 5 * (random.randrange(0, 3, 2) - 1)
    vector_y = 5
    score=0
  if(y_ball<=0): vector_y = 5 
  if(x_ball>=700): vector_x = -5
  if(x_ball<=0): vector_x = 5
  x_ball += vector_x
  y_ball += vector_y
  pygame.draw.circle(screen, WHITE, (x_ball, y_ball), 10)
 
  #スコアを表示
  font = pygame.font.SysFont(None, 80)
  score_text = font.render(str(score), True, (0,255,255))
  screen.blit(score_text, (630,10))
 
  pygame.display.flip()
  myclock.tick(60)
 
pygame.quit()