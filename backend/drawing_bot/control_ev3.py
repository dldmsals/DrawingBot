#!/usr/bin/env pybricks-micropython
from pybricks.hubs      import EV3Brick
from pybricks.ev3devices import Motor, TouchSensor
from pybricks.parameters import Port, Stop
from pybricks.tools     import wait
import math
import ujson

# ===== 하드웨어 초기화 =====
ev3            = EV3Brick()
x_motor        = Motor(Port.D)
y_motor1       = Motor(Port.B)
y_motor2       = Motor(Port.C)
z_motor        = Motor(Port.A)

x_stop_sensor  = TouchSensor(Port.S2)
y_stop_sensor  = TouchSensor(Port.S4)

# ===== 상수 정의 =====
PEN_DOWN_ANGLE = 110
PEN_UP_ANGLE   =   0

 
X_GEAR_RATIO = 1.0
Y_GEAR_RATIO = 1.0

MOTOR_MAX_X = 400
MOTOR_MAX_Y = 1100
MOTOR_MIN   = 0

AXIS_MAX_X = MOTOR_MAX_X / X_GEAR_RATIO
AXIS_MAX_Y = MOTOR_MAX_Y / Y_GEAR_RATIO

# ===== 유틸 함수 =====
def pen_down(speed=100):
    z_motor.run_target(speed, PEN_DOWN_ANGLE, Stop.HOLD)

def pen_up(speed=100):
    z_motor.run_target(speed, PEN_UP_ANGLE, Stop.HOLD)

def move_to(axis_x, axis_y, speed=150, tol=2):
    ax = max(0, min(AXIS_MAX_X, axis_x))
    ay = max(0, min(AXIS_MAX_Y, axis_y))
    mx = ax * X_GEAR_RATIO
    my = ay * Y_GEAR_RATIO
    mx = max(MOTOR_MIN, min(MOTOR_MAX_X, mx))
    my = max(MOTOR_MIN, min(MOTOR_MAX_Y, my))
    x_motor.run_target(speed, mx, Stop.HOLD, wait=False)
    y_motor1.run_target(speed, my, Stop.HOLD, wait=False)
    y_motor2.run_target(speed, my, Stop.HOLD, wait=False)
    while abs(x_motor.angle() - mx) > tol or abs(y_motor1.angle() - my) > tol:
        wait(10)

def initial_setup(speed=200):
    # X축 홈 찾기
    x_motor.run(-speed)
    while not x_stop_sensor.pressed():
        wait(10)
    x_motor.stop(Stop.BRAKE)

    # Y축 홈 찾기
    y_motor1.run(-speed)
    y_motor2.run(-speed)
    while not y_stop_sensor.pressed():
        wait(10)
    y_motor1.stop(Stop.BRAKE)
    y_motor2.stop(Stop.BRAKE)

# ===== 메인 실행부 =====
if __name__ == "__main__":
    initial_setup()
    x_motor.reset_angle(0)
    y_motor1.reset_angle(0)
    y_motor2.reset_angle(0)
    z_motor.reset_angle(PEN_UP_ANGLE)

    with open("drawing_paths_stream.json", "r") as f:
        for line in f:
            path = ujson.loads(line)
            pen_up()
            move_to(*path[0])
            pen_down()
            for x, y in path[1:]:
                move_to(x, y)
            pen_up()

    ev3.speaker.beep()
