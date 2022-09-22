# ピンボールランナーを作成
# mediapipeで体の位置を認識しておいて，鼻の位置or肩の位置で映像を動かす

# ピンボールランナーを作成
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from PIL import Image

import pygame
import sys
import random
import time

global nose_position
global left_shoulder
global right_shoulder
global right_elbow
global left_elbow
global right_hand
global left_hand
global right_waist
global left_waist
global right_knee
global left_knee
global right_ankle
global left_ankle


 


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    # parser.add_argument('--upper_body_only', action='store_true')  # 0.8.3 or less
    parser.add_argument('--unuse_smooth_landmarks', action='store_true')
    parser.add_argument('--enable_segmentation', action='store_true')
    parser.add_argument('--smooth_segmentation', action='store_true')
    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=1)
    parser.add_argument("--min_detection_confidence",
                        help='face mesh min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='face mesh min_tracking_confidence',
                        type=int,
                        default=0.5)
    parser.add_argument("--segmentation_score_th",
                        help='segmentation_score_threshold',
                        type=float,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true')
    parser.add_argument('--plot_world_landmark', action='store_true')

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    # upper_body_only = args.upper_body_only
    smooth_landmarks = not args.unuse_smooth_landmarks
    enable_segmentation = args.enable_segmentation
    smooth_segmentation = args.smooth_segmentation
    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    segmentation_score_th = args.segmentation_score_th

    use_brect = args.use_brect
    plot_world_landmark = args.plot_world_landmark

    # ゲーム準備 ###############################################################
    global nose_position
    global left_shoulder
    global right_shoulder
    global right_elbow
    global left_elbow
    global right_hand
    global left_hand
    global right_waist
    global left_waist
    global right_knee
    global left_knee
    global right_ankle
    global left_ankle

    nose_position = [0, 0]
    left_shoulder = [0, 0]
    right_shoulder = [0, 0]
    right_elbow = [0, 0]
    left_elbow = [0, 0]
    right_hand = [0, 0]
    left_hand = [0, 0]
    right_waist = [0, 0]
    left_waist = [0, 0]
    right_knee = [0, 0]
    left_knee = [0, 0]
    right_ankle = [0, 0]
    left_ankle = [0, 0]


    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    WHITE = (255, 255, 255)

    pygame.init()
    screen = pygame.display.set_mode((700, 480), pygame.FULLSCREEN)
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
    feverend = pygame.image.load("materials//feverend.jpg")
    
    x_paddle=250

    font = pygame.font.SysFont(None, 80)
    font_big = pygame.font.SysFont(None, 200)

    score = 0
    BALLSIZE = 30
    BALLSPEED = 10
    BALLQUANTITY = 2
    TIMELIMIT = 90
    ball_count = 0
    ball_array = []
    fever_flag = False
    game_start = False

    for i in range(BALLQUANTITY):
        ball_array.append([[10, 10], [10, 10], [ohnishi, ohnishi_goal]])
    #ball_array[ボールの番号][ボールの画像の名前，ボールのベクトル，ボールの位置][x,y もしくは画像，音]
    
    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        # upper_body_only=upper_body_only,
        model_complexity=model_complexity,
        smooth_landmarks=smooth_landmarks,
        #enable_segmentation=enable_segmentation,    ###
        #smooth_segmentation=smooth_segmentation,    ###
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # World座標プロット ########################################################
    if plot_world_landmark:
        import matplotlib.pyplot as plt
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

    starttime = time.time()
    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True

        # Face Mesh ###########################################################
        face_landmarks = results.face_landmarks
        if face_landmarks is not None:
            # 外接矩形の計算
            brect = calc_bounding_rect(debug_image, face_landmarks)
            # 描画
            debug_image = draw_face_landmarks(debug_image, face_landmarks)
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)

        # Pose ###############################################################
        if enable_segmentation and results.segmentation_mask is not None:
            # セグメンテーション
            mask = np.stack((results.segmentation_mask, ) * 3,
                            axis=-1) > segmentation_score_th
            bg_resize_image = np.zeros(image.shape, dtype=np.uint8)
            bg_resize_image[:] = (0, 255, 0)
            debug_image = np.where(mask, debug_image, bg_resize_image)

        pose_landmarks = results.pose_landmarks
        if pose_landmarks is not None:
            # 外接矩形の計算
            brect = calc_bounding_rect(debug_image, pose_landmarks)
            # 描画
            debug_image = draw_pose_landmarks(
                debug_image,
                pose_landmarks,
                # upper_body_only,
            )
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)

        # Pose:World座標プロット #############################################
        if plot_world_landmark:
            if results.pose_world_landmarks is not None:
                plot_world_landmarks(
                    plt,
                    ax,
                    results.pose_world_landmarks,
                )

        # Hands ###############################################################
        left_hand_landmarks = results.left_hand_landmarks
        right_hand_landmarks = results.right_hand_landmarks

        # キー処理(ESC：終了) #################################################
        press = pygame.key.get_pressed()
        if(press[pygame.K_SPACE]):
            game_start = True
        if(press[pygame.K_ESCAPE]):
            break

        # ゲーム ###############################################################
        for event in pygame.event.get():
            if event.type==pygame.QUIT: flag=1
        rest_time = int(TIMELIMIT - time.time() + starttime)
        screen.fill(BLACK)
        if(not game_start):
            starttime = time.time()
            score_text = font_big.render("SPACE", True, (0,255,255))
            screen.blit(score_text, (200,100))
        
        elif(fever_flag and  (time.time() - fever_start < 3 or time.time() - fever_start > 18)): #フィーバータイム
            fever_time = time.time() - fever_start
            if(fever_time < 3):
                screen.blit(fever, (100, 100))
                temp = BALLQUANTITY
                BALLQUANTITY = 50
                for i in range(BALLQUANTITY- temp):
                    ball_array.append([[10, 10], [10, 10], [ohnishi, ohnishi_goal]])
            elif(fever_time > 18):
                screen.blit(feverend, (100, 100))
                BALLQUANTITY = 2
                if(fever_time > 21):
                    fever_flag = False
        elif(rest_time <= 0):
            score_text = font_big.render("score " + str(score), True, (0,255,255))
            screen.blit(score_text, (200,100))
        else:
            
            #パドルを描画
            x_paddle = nose_position[0]
            rect = pygame.Rect(x_paddle, 400, 100, 30)
            pygame.draw.rect(screen, RED, rect)

            left_top = [left_shoulder,left_elbow, left_hand]
            right_top = [right_shoulder,right_elbow, right_hand]
            left_bottom = [left_waist, left_knee, left_ankle]
            right_bottom = [right_waist, right_knee, right_ankle]
            body_position = [nose_position] + left_top + right_top + left_bottom + right_bottom
            body_position = list(body_position)
            for i in range(len(body_position)):
                body_position[i][0] = body_position[i][0]*0.2
                body_position[i][1] = body_position[i][1]*0.2
                body_position[i][1] = body_position[i][1] + 300

            pygame.draw.line(screen, RED, body_position[0], body_position[1], 3)
            pygame.draw.line(screen, RED, body_position[0], body_position[4], 3)
            pygame.draw.line(screen, RED, body_position[1], body_position[7], 3)
            pygame.draw.line(screen, RED, body_position[4], body_position[10], 3)

            for i in range(2):
                pygame.draw.line(screen, RED, body_position[i+1], body_position[i+2], 3)
                pygame.draw.line(screen, RED, body_position[i+4], body_position[i+5], 3)
                pygame.draw.line(screen, RED, body_position[i+7], body_position[i+8], 3)
                pygame.draw.line(screen, RED, body_position[i+10], body_position[i+11], 3)

            
            # 障害物を描画

            ball_array = make_obstacle(240, 300, ball_array, screen, WHITE,BALLSIZE, BALLQUANTITY)
            ball_array = make_obstacle(210, 200, ball_array, screen, WHITE,BALLSIZE, BALLQUANTITY)
            ball_array = make_obstacle(110, 350, ball_array, screen, WHITE,BALLSIZE, BALLQUANTITY)
            ball_array = make_obstacle(610, 220, ball_array, screen, WHITE,BALLSIZE, BALLQUANTITY)
            ball_array = make_obstacle(580, 320, ball_array, screen, WHITE,BALLSIZE, BALLQUANTITY)

            #当たり判定
            for i in range(BALLQUANTITY):
                if(ball_array[i][1][1]==390 and ball_array[i][1][0]>=(x_paddle-10) and ball_array[i][1][0]<=(x_paddle+95)):
                    ball_count += 1
                    ball_array[i][2][1].play(0)

                    if(ball_array[i][2][0] == gold and not fever_flag):
                        fever_flag = True
                        fever_start = time.time()
                        starttime = starttime + 6
                    elif(ball_array[i][2][0] == tsukamoto): score += 10
                    elif(ball_array[i][2][0] == terada): score += 5
                    else: score += 1

                    ball_array[i][1][0] = random.randrange(700)
                    ball_array[i][1][1] = 10
                    ball_array[i][0][0] = BALLSPEED * (random.randrange(0, 3, 2) - 1)
                    ball_array[i][0][1] = BALLSPEED
                    

                    if(ball_count%3 == 0 and not fever_flag):
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

                    if(ball_count%3 ==0 and not fever_flag):
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
            score_text = font.render("score " + str(score), True, (0,255,255))
            screen.blit(score_text, (470,50))

            #残り時間を表示
            time_text = font.render("time " + str(rest_time), True, (0,255,255))
            screen.blit(time_text, (470,10))

        pygame.display.flip()
        myclock.tick(60)

    


    cap.release()
    cv.destroyAllWindows()
            


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]




def draw_face_landmarks(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append((landmark_x, landmark_y))

        cv.circle(image, (landmark_x, landmark_y), 1, (0, 255, 0), 1)

    if len(landmark_point) > 0:
        # 参考：https://github.com/tensorflow/tfjs-models/blob/master/facemesh/mesh_map.jpg

        # 左眉毛(55：内側、46：外側)
        cv.line(image, landmark_point[55], landmark_point[65], (0, 255, 0), 2)
        cv.line(image, landmark_point[65], landmark_point[52], (0, 255, 0), 2)
        cv.line(image, landmark_point[52], landmark_point[53], (0, 255, 0), 2)
        cv.line(image, landmark_point[53], landmark_point[46], (0, 255, 0), 2)

        # 右眉毛(285：内側、276：外側)
        cv.line(image, landmark_point[285], landmark_point[295], (0, 255, 0),
                2)
        cv.line(image, landmark_point[295], landmark_point[282], (0, 255, 0),
                2)
        cv.line(image, landmark_point[282], landmark_point[283], (0, 255, 0),
                2)
        cv.line(image, landmark_point[283], landmark_point[276], (0, 255, 0),
                2)

        # 左目 (133：目頭、246：目尻)
        cv.line(image, landmark_point[133], landmark_point[173], (0, 255, 0),
                2)
        cv.line(image, landmark_point[173], landmark_point[157], (0, 255, 0),
                2)
        cv.line(image, landmark_point[157], landmark_point[158], (0, 255, 0),
                2)
        cv.line(image, landmark_point[158], landmark_point[159], (0, 255, 0),
                2)
        cv.line(image, landmark_point[159], landmark_point[160], (0, 255, 0),
                2)
        cv.line(image, landmark_point[160], landmark_point[161], (0, 255, 0),
                2)
        cv.line(image, landmark_point[161], landmark_point[246], (0, 255, 0),
                2)

        cv.line(image, landmark_point[246], landmark_point[163], (0, 255, 0),
                2)
        cv.line(image, landmark_point[163], landmark_point[144], (0, 255, 0),
                2)
        cv.line(image, landmark_point[144], landmark_point[145], (0, 255, 0),
                2)
        cv.line(image, landmark_point[145], landmark_point[153], (0, 255, 0),
                2)
        cv.line(image, landmark_point[153], landmark_point[154], (0, 255, 0),
                2)
        cv.line(image, landmark_point[154], landmark_point[155], (0, 255, 0),
                2)
        cv.line(image, landmark_point[155], landmark_point[133], (0, 255, 0),
                2)

        # 右目 (362：目頭、466：目尻)
        cv.line(image, landmark_point[362], landmark_point[398], (0, 255, 0),
                2)
        cv.line(image, landmark_point[398], landmark_point[384], (0, 255, 0),
                2)
        cv.line(image, landmark_point[384], landmark_point[385], (0, 255, 0),
                2)
        cv.line(image, landmark_point[385], landmark_point[386], (0, 255, 0),
                2)
        cv.line(image, landmark_point[386], landmark_point[387], (0, 255, 0),
                2)
        cv.line(image, landmark_point[387], landmark_point[388], (0, 255, 0),
                2)
        cv.line(image, landmark_point[388], landmark_point[466], (0, 255, 0),
                2)

        cv.line(image, landmark_point[466], landmark_point[390], (0, 255, 0),
                2)
        cv.line(image, landmark_point[390], landmark_point[373], (0, 255, 0),
                2)
        cv.line(image, landmark_point[373], landmark_point[374], (0, 255, 0),
                2)
        cv.line(image, landmark_point[374], landmark_point[380], (0, 255, 0),
                2)
        cv.line(image, landmark_point[380], landmark_point[381], (0, 255, 0),
                2)
        cv.line(image, landmark_point[381], landmark_point[382], (0, 255, 0),
                2)
        cv.line(image, landmark_point[382], landmark_point[362], (0, 255, 0),
                2)

        # 口 (308：右端、78：左端)
        cv.line(image, landmark_point[308], landmark_point[415], (0, 255, 0),
                2)
        cv.line(image, landmark_point[415], landmark_point[310], (0, 255, 0),
                2)
        cv.line(image, landmark_point[310], landmark_point[311], (0, 255, 0),
                2)
        cv.line(image, landmark_point[311], landmark_point[312], (0, 255, 0),
                2)
        cv.line(image, landmark_point[312], landmark_point[13], (0, 255, 0), 2)
        cv.line(image, landmark_point[13], landmark_point[82], (0, 255, 0), 2)
        cv.line(image, landmark_point[82], landmark_point[81], (0, 255, 0), 2)
        cv.line(image, landmark_point[81], landmark_point[80], (0, 255, 0), 2)
        cv.line(image, landmark_point[80], landmark_point[191], (0, 255, 0), 2)
        cv.line(image, landmark_point[191], landmark_point[78], (0, 255, 0), 2)

        cv.line(image, landmark_point[78], landmark_point[95], (0, 255, 0), 2)
        cv.line(image, landmark_point[95], landmark_point[88], (0, 255, 0), 2)
        cv.line(image, landmark_point[88], landmark_point[178], (0, 255, 0), 2)
        cv.line(image, landmark_point[178], landmark_point[87], (0, 255, 0), 2)
        cv.line(image, landmark_point[87], landmark_point[14], (0, 255, 0), 2)
        cv.line(image, landmark_point[14], landmark_point[317], (0, 255, 0), 2)
        cv.line(image, landmark_point[317], landmark_point[402], (0, 255, 0),
                2)
        cv.line(image, landmark_point[402], landmark_point[318], (0, 255, 0),
                2)
        cv.line(image, landmark_point[318], landmark_point[324], (0, 255, 0),
                2)
        cv.line(image, landmark_point[324], landmark_point[308], (0, 255, 0),
                2)

    return image


def draw_pose_landmarks(
    image,
    landmarks,
    # upper_body_only,
    visibility_th=0.5,
):
    global nose_position
    global left_shoulder
    global right_shoulder
    global right_elbow
    global left_elbow
    global right_hand
    global left_hand
    global right_waist
    global left_waist
    global right_knee
    global left_knee
    global right_ankle
    global left_ankle

    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

        if landmark.visibility < visibility_th:
            continue

        if index == 0:  # 鼻
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            nose_position = [int(landmark_x), int(landmark_y)]

        if index == 1:  # 右目：目頭
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 右目：瞳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 右目：目尻
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 左目：目頭
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 5:  # 左目：瞳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 左目：目尻
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 右耳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 左耳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 9:  # 口：左端
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 口：左端
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 右肩
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            right_shoulder =  [int(landmark_x), int(landmark_y)]
        if index == 12:  # 左肩
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            left_shoulder = [int(landmark_x), int(landmark_y)]
        if index == 13:  # 右肘
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            right_elbow  = [int(landmark_x), int(landmark_y)]
        if index == 14:  # 左肘
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            left_elbow  = [int(landmark_x), int(landmark_y)]
        if index == 15:  # 右手首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            right_hand  = [int(landmark_x), int(landmark_y)]
        if index == 16:  # 左手首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            left_hand  = [int(landmark_x), int(landmark_y)]
        if index == 17:  # 右手1(外側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 左手1(外側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 右手2(先端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 左手2(先端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 21:  # 右手3(内側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 22:  # 左手3(内側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 23:  # 腰(右側)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            right_waist  = [int(landmark_x), int(landmark_y)]
        if index == 24:  # 腰(左側)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            left_waist  = [int(landmark_x), int(landmark_y)]
        if index == 25:  # 右ひざ
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            right_knee  = [int(landmark_x), int(landmark_y)]
        if index == 26:  # 左ひざ
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            left_knee  = [int(landmark_x), int(landmark_y)]
        if index == 27:  # 右足首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            right_ankle  = [int(landmark_x), int(landmark_y)]
        if index == 28:  # 左足首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            left_ankle  = [int(landmark_x), int(landmark_y)]
        if index == 29:  # 右かかと
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 30:  # 左かかと
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 31:  # 右つま先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 32:  # 左つま先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)

        # if not upper_body_only:
        if True:
            cv.putText(image, "z:" + str(round(landmark_z, 3)),
                       (landmark_x - 10, landmark_y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                       cv.LINE_AA)

    if len(landmark_point) > 0:
        # 右目
        if landmark_point[1][0] > visibility_th and landmark_point[2][
                0] > visibility_th:
            cv.line(image, landmark_point[1][1], landmark_point[2][1],
                    (0, 255, 0), 2)
        if landmark_point[2][0] > visibility_th and landmark_point[3][
                0] > visibility_th:
            cv.line(image, landmark_point[2][1], landmark_point[3][1],
                    (0, 255, 0), 2)

        # 左目
        if landmark_point[4][0] > visibility_th and landmark_point[5][
                0] > visibility_th:
            cv.line(image, landmark_point[4][1], landmark_point[5][1],
                    (0, 255, 0), 2)
        if landmark_point[5][0] > visibility_th and landmark_point[6][
                0] > visibility_th:
            cv.line(image, landmark_point[5][1], landmark_point[6][1],
                    (0, 255, 0), 2)

        # 口
        if landmark_point[9][0] > visibility_th and landmark_point[10][
                0] > visibility_th:
            cv.line(image, landmark_point[9][1], landmark_point[10][1],
                    (0, 255, 0), 2)

        # 肩
        if landmark_point[11][0] > visibility_th and landmark_point[12][
                0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[12][1],
                    (0, 255, 0), 2)

        # 右腕
        if landmark_point[11][0] > visibility_th and landmark_point[13][
                0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[13][1],
                    (0, 255, 0), 2)
        if landmark_point[13][0] > visibility_th and landmark_point[15][
                0] > visibility_th:
            cv.line(image, landmark_point[13][1], landmark_point[15][1],
                    (0, 255, 0), 2)

        # 左腕
        if landmark_point[12][0] > visibility_th and landmark_point[14][
                0] > visibility_th:
            cv.line(image, landmark_point[12][1], landmark_point[14][1],
                    (0, 255, 0), 2)
        if landmark_point[14][0] > visibility_th and landmark_point[16][
                0] > visibility_th:
            cv.line(image, landmark_point[14][1], landmark_point[16][1],
                    (0, 255, 0), 2)

        # 右手
        if landmark_point[15][0] > visibility_th and landmark_point[17][
                0] > visibility_th:
            cv.line(image, landmark_point[15][1], landmark_point[17][1],
                    (0, 255, 0), 2)
        if landmark_point[17][0] > visibility_th and landmark_point[19][
                0] > visibility_th:
            cv.line(image, landmark_point[17][1], landmark_point[19][1],
                    (0, 255, 0), 2)
        if landmark_point[19][0] > visibility_th and landmark_point[21][
                0] > visibility_th:
            cv.line(image, landmark_point[19][1], landmark_point[21][1],
                    (0, 255, 0), 2)
        if landmark_point[21][0] > visibility_th and landmark_point[15][
                0] > visibility_th:
            cv.line(image, landmark_point[21][1], landmark_point[15][1],
                    (0, 255, 0), 2)

        # 左手
        if landmark_point[16][0] > visibility_th and landmark_point[18][
                0] > visibility_th:
            cv.line(image, landmark_point[16][1], landmark_point[18][1],
                    (0, 255, 0), 2)
        if landmark_point[18][0] > visibility_th and landmark_point[20][
                0] > visibility_th:
            cv.line(image, landmark_point[18][1], landmark_point[20][1],
                    (0, 255, 0), 2)
        if landmark_point[20][0] > visibility_th and landmark_point[22][
                0] > visibility_th:
            cv.line(image, landmark_point[20][1], landmark_point[22][1],
                    (0, 255, 0), 2)
        if landmark_point[22][0] > visibility_th and landmark_point[16][
                0] > visibility_th:
            cv.line(image, landmark_point[22][1], landmark_point[16][1],
                    (0, 255, 0), 2)

        # 胴体
        if landmark_point[11][0] > visibility_th and landmark_point[23][
                0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[23][1],
                    (0, 255, 0), 2)
        if landmark_point[12][0] > visibility_th and landmark_point[24][
                0] > visibility_th:
            cv.line(image, landmark_point[12][1], landmark_point[24][1],
                    (0, 255, 0), 2)
        if landmark_point[23][0] > visibility_th and landmark_point[24][
                0] > visibility_th:
            cv.line(image, landmark_point[23][1], landmark_point[24][1],
                    (0, 255, 0), 2)

        if len(landmark_point) > 25:
            # 右足
            if landmark_point[23][0] > visibility_th and landmark_point[25][
                    0] > visibility_th:
                cv.line(image, landmark_point[23][1], landmark_point[25][1],
                        (0, 255, 0), 2)
            if landmark_point[25][0] > visibility_th and landmark_point[27][
                    0] > visibility_th:
                cv.line(image, landmark_point[25][1], landmark_point[27][1],
                        (0, 255, 0), 2)
            if landmark_point[27][0] > visibility_th and landmark_point[29][
                    0] > visibility_th:
                cv.line(image, landmark_point[27][1], landmark_point[29][1],
                        (0, 255, 0), 2)
            if landmark_point[29][0] > visibility_th and landmark_point[31][
                    0] > visibility_th:
                cv.line(image, landmark_point[29][1], landmark_point[31][1],
                        (0, 255, 0), 2)

            # 左足
            if landmark_point[24][0] > visibility_th and landmark_point[26][
                    0] > visibility_th:
                cv.line(image, landmark_point[24][1], landmark_point[26][1],
                        (0, 255, 0), 2)
            if landmark_point[26][0] > visibility_th and landmark_point[28][
                    0] > visibility_th:
                cv.line(image, landmark_point[26][1], landmark_point[28][1],
                        (0, 255, 0), 2)
            if landmark_point[28][0] > visibility_th and landmark_point[30][
                    0] > visibility_th:
                cv.line(image, landmark_point[28][1], landmark_point[30][1],
                        (0, 255, 0), 2)
            if landmark_point[30][0] > visibility_th and landmark_point[32][
                    0] > visibility_th:
                cv.line(image, landmark_point[30][1], landmark_point[32][1],
                        (0, 255, 0), 2)
    return image


def plot_world_landmarks(
    plt,
    ax,
    landmarks,
    visibility_th=0.5,
):
    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_point.append(
            [landmark.visibility, (landmark.x, landmark.y, landmark.z)])

    face_index_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    right_arm_index_list = [11, 13, 15, 17, 19, 21]
    left_arm_index_list = [12, 14, 16, 18, 20, 22]
    right_body_side_index_list = [11, 23, 25, 27, 29, 31]
    left_body_side_index_list = [12, 24, 26, 28, 30, 32]
    shoulder_index_list = [11, 12]
    waist_index_list = [23, 24]

    # 顔
    face_x, face_y, face_z = [], [], []
    for index in face_index_list:
        point = landmark_point[index][1]
        face_x.append(point[0])
        face_y.append(point[2])
        face_z.append(point[1] * (-1))

    # 右腕
    right_arm_x, right_arm_y, right_arm_z = [], [], []
    for index in right_arm_index_list:
        point = landmark_point[index][1]
        right_arm_x.append(point[0])
        right_arm_y.append(point[2])
        right_arm_z.append(point[1] * (-1))

    # 左腕
    left_arm_x, left_arm_y, left_arm_z = [], [], []
    for index in left_arm_index_list:
        point = landmark_point[index][1]
        left_arm_x.append(point[0])
        left_arm_y.append(point[2])
        left_arm_z.append(point[1] * (-1))

    # 右半身
    right_body_side_x, right_body_side_y, right_body_side_z = [], [], []
    for index in right_body_side_index_list:
        point = landmark_point[index][1]
        right_body_side_x.append(point[0])
        right_body_side_y.append(point[2])
        right_body_side_z.append(point[1] * (-1))

    # 左半身
    left_body_side_x, left_body_side_y, left_body_side_z = [], [], []
    for index in left_body_side_index_list:
        point = landmark_point[index][1]
        left_body_side_x.append(point[0])
        left_body_side_y.append(point[2])
        left_body_side_z.append(point[1] * (-1))

    # 肩
    shoulder_x, shoulder_y, shoulder_z = [], [], []
    for index in shoulder_index_list:
        point = landmark_point[index][1]
        shoulder_x.append(point[0])
        shoulder_y.append(point[2])
        shoulder_z.append(point[1] * (-1))

    # 腰
    waist_x, waist_y, waist_z = [], [], []
    for index in waist_index_list:
        point = landmark_point[index][1]
        waist_x.append(point[0])
        waist_y.append(point[2])
        waist_z.append(point[1] * (-1))
            
    ax.cla()
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    ax.scatter(face_x, face_y, face_z)
    ax.plot(right_arm_x, right_arm_y, right_arm_z)
    ax.plot(left_arm_x, left_arm_y, left_arm_z)
    ax.plot(right_body_side_x, right_body_side_y, right_body_side_z)
    ax.plot(left_body_side_x, left_body_side_y, left_body_side_z)
    ax.plot(shoulder_x, shoulder_y, shoulder_z)
    ax.plot(waist_x, waist_y, waist_z)
    
    plt.pause(.001)

    return


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image

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
