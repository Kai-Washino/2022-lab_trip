import pygame
import random
 

def main():
  BLACK = (0, 0, 0)
  RED = (255, 0, 0)
  WHITE = (255, 255, 255)

  pygame.init()
  screen = pygame.display.set_mode((700, 480))
  myclock = pygame.time.Clock()
 
  flag=0
  x_paddle=250


  score = 0
  BALLSIZE = 30
  BALLSPEED = 5
  BALLQUANTITY = 2
  ball_array = []
  for i in range(BALLQUANTITY):
    ball_array.append([[10, 10], [10, 10]])
  #ball_array[ボールの番号][ボールのベクトル，ボールの位置][x,y]

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

    # 障害物を描画

    ball_array = make_obstacle(240, 300, ball_array, screen, WHITE,BALLSIZE, BALLQUANTITY)
    ball_array = make_obstacle(210, 200, ball_array, screen, WHITE,BALLSIZE, BALLQUANTITY)
    ball_array = make_obstacle(110, 350, ball_array, screen, WHITE,BALLSIZE, BALLQUANTITY)
    ball_array = make_obstacle(610, 220, ball_array, screen, WHITE,BALLSIZE, BALLQUANTITY)
    ball_array = make_obstacle(580, 320, ball_array, screen, WHITE,BALLSIZE, BALLQUANTITY)

    #ボールを描画

    #当たり判定
    for i in range(BALLQUANTITY):
      if(ball_array[i][1][1]==390 and ball_array[i][1][0]>=(x_paddle-5) and ball_array[i][1][0]<=(x_paddle+95)):
        ball_array[i][1][0] = random.randrange(700)
        ball_array[i][1][1] = 10
        ball_array[i][0][0] = BALLSPEED * (random.randrange(0, 3, 2) - 1)
        ball_array[i][0][1] = BALLSPEED
        score += 1

      #終了判定
      if(ball_array[i][1][1]>500):
        ball_array[i][1][0] = random.randrange(700)
        ball_array[i][1][1] = 10
        ball_array[i][0][0] = BALLSPEED* (random.randrange(0, 3, 2) - 1)
        ball_array[i][0][1] = BALLSPEED

      #壁に反射
      if(ball_array[i][1][0]>=700): ball_array[i][0][0] = -BALLSPEED
      if(ball_array[i][1][0]<=0): ball_array[i][0][0] = BALLSPEED
      if(ball_array[i][1][1]<=0): ball_array[i][0][1] = BALLSPEED
      ball_array[i][1][0] += ball_array[i][0][0]
      ball_array[i][1][1] += ball_array[i][0][1]
      pygame.draw.circle(screen, WHITE, (ball_array[i][1][0], ball_array[i][1][1]), BALLSIZE)
  
    #スコアを表示
    font = pygame.font.SysFont(None, 80)
    score_text = font.render(str(score), True, (0,255,255))
    screen.blit(score_text, (630,10))
  
    pygame.display.flip()
    myclock.tick(60)
  
  pygame.quit()

#障害物作成と反射
def make_obstacle(x_place, y_place, ball_array, screen, WHITE,BALLSIZE, BALLQUANTITY):
  pygame.draw.circle(screen, WHITE, (x_place,y_place), 10)
  for i in range(BALLQUANTITY):
    if(ball_array[i][1][1] < y_place + BALLSIZE and ball_array[i][1][1] > y_place - BALLSIZE  and ball_array[i][1][0] <  x_place + BALLSIZE  and ball_array[i][1][0] >  x_place - BALLSIZE ):
      if(ball_array[i][1][1] < y_place - BALLSIZE + 15 or ball_array[i][1][1] > y_place + BALLSIZE - 15):
        ball_array[i][0][1] = ball_array[i][0][1] * -1
        print("yが変わった")
      else:
        ball_array[i][0][0] = ball_array[i][0][0] * -1
        print("xが変わった")
  return ball_array

if __name__ == '__main__':
  main()