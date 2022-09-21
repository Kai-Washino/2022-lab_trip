import pygame
import random
import time
 

def main():
  BLACK = (0, 0, 0)
  RED = (255, 0, 0)
  WHITE = (255, 255, 255)

  pygame.init()
  screen = pygame.display.set_mode((700, 480))
  myclock = pygame.time.Clock()
  pygame.mixer.init(frequency = 44100)    # 初期設定
  bound = pygame.mixer.Sound("materials//ビヨォン.wav")

  tsukamoto_goal = pygame.mixer.Sound("materials//tsukamoto_goal.wav")
  terada_goal = pygame.mixer.Sound("materials//terada_goal.wav")
  ohnishi_goal = pygame.mixer.Sound("materials//ohnishi_goal.wav")
  tsuchida_goal = pygame.mixer.Sound("materials//tsuchida_goal.wav")
  gold_goal = pygame.mixer.Sound("materials//gold_goal.wav")

  tsukamoto = pygame.image.load("materials//tsukamoto.png")
  terada = pygame.image.load("materials//terada.png")
  ohnishi = pygame.image.load("materials//ohnishi.png")
  tsuchida = pygame.image.load("materials//tsuchida.png")
  gold = pygame.image.load("materials//gold.png")
  fever = pygame.image.load("materials//fever.jpg")
  

  flag=0
  x_paddle=250


  score = 0
  ball_count = 0
  BALLSIZE = 30
  BALLSPEED = 10
  BALLQUANTITY = 2
  ball_array = []
  fever_time = False

  for i in range(BALLQUANTITY):
    ball_array.append([[10, 10], [10, 10], [ohnishi, ohnishi_goal]])
  #ball_array[ボールの番号][ボールの画像の名前，ボールのベクトル，ボールの位置][x,y もしくは画像，音]


  while flag==0:
    for event in pygame.event.get():
      if event.type==pygame.QUIT: flag=1
    screen.fill(BLACK)
  
    #パドルを描画
    press = pygame.key.get_pressed()
    if(press[pygame.K_LEFT] and x_paddle>0): x_paddle-=10
    if(press[pygame.K_RIGHT] and x_paddle<600): x_paddle+=10
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
      if(ball_array[i][1][1]==390 and ball_array[i][1][0]>=(x_paddle-10) and ball_array[i][1][0]<=(x_paddle+95)):
        ball_count += 1
        print(ball_count)
        ball_array[i][2][1].play(0)

        if(ball_array[i][2][0] == gold and not fever_time):
          print("金")
          temp = BALLQUANTITY
          BALLQUANTITY = 50
          for i in range(BALLQUANTITY- temp):
            ball_array.append([[10, 10], [10, 10], [ohnishi, ohnishi_goal]])
          screen.blit(fever, (100, 100))
          fever_time = True
          time.sleep(3)
        elif(ball_array[i][2][0] == tsukamoto): score += 10
        elif(ball_array[i][2][0] == terada): score += 5
        else: score += 1

        ball_array[i][1][0] = random.randrange(700)
        ball_array[i][1][1] = 10
        ball_array[i][0][0] = BALLSPEED * (random.randrange(0, 3, 2) - 1)
        ball_array[i][0][1] = BALLSPEED
        

        if(ball_count%3 == 0 and not fever_time):
          ball_array[i][2][0] = gold
          ball_array[i][2][1] = gold_goal
        elif(ball_count%20 ==0):
          ball_array[i][2][0] = tsukamoto
          ball_array[i][2][1] = tsukamoto_goal
        elif(ball_count%15 ==0):
          ball_array[i][2][0] = terada
          ball_array[i][2][1] = terada_goal
        elif(ball_count%2 ==0):
          ball_array[i][2][0] = ohnishi
          ball_array[i][2][1] = ohnishi_goal
        else:
          ball_array[i][2][0] = tsuchida
          ball_array[i][2][1] = tsuchida_goal
        

      #終了判定
      if(ball_array[i][1][1]>500):
        ball_array[i][1][0] = random.randrange(700)
        ball_array[i][1][1] = 10
        ball_array[i][0][0] = BALLSPEED* (random.randrange(0, 3, 2) - 1)
        ball_array[i][0][1] = BALLSPEED
        ball_count += 1

        if(ball_count%50 ==0):
          ball_array[i][2][0] = gold
          ball_array[i][2][1] = gold_goal
        elif(ball_count%20 ==0):
          ball_array[i][2][0] = tsukamoto
          ball_array[i][2][1] = tsukamoto_goal
        elif(ball_count%15 ==0):
          ball_array[i][2][0] = terada
          ball_array[i][2][1] = terada_goal
        elif(ball_count%2 ==0):
          ball_array[i][2][0] = ohnishi
          ball_array[i][2][1] = ohnishi_goal
        else:
          ball_array[i][2][0] = tsuchida
          ball_array[i][2][1] = tsuchida_goal

      #壁に反射
      if(ball_array[i][1][0]>=700): 
        ball_array[i][0][0] = -BALLSPEED
        bound.play(0)
      if(ball_array[i][1][0]<=0): 
        ball_array[i][0][0] = BALLSPEED
        bound.play(0)
      if(ball_array[i][1][1]<=0):
        ball_array[i][0][1] = BALLSPEED
        bound.play(0)
      ball_array[i][1][0] += ball_array[i][0][0]
      ball_array[i][1][1] += ball_array[i][0][1]
      # pygame.draw.circle(screen, WHITE, (ball_array[i][1][0], ball_array[i][1][1]), BALLSIZE)

      screen.blit(ball_array[i][2][0], (ball_array[i][1][0] - BALLSIZE, ball_array[i][1][1] - BALLSIZE))
  
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
      else:
        ball_array[i][0][0] = ball_array[i][0][0] * -1
  return ball_array

if __name__ == '__main__':
  main()