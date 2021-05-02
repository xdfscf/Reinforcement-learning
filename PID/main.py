import time

class PID:
    """PID Controller
    """

    def __init__(self, P=0.2, I=0.0, D=0.0, current_time=None):

        self.Kp = P
        self.Ki = I
        self.Kd = D

        self.sample_time = 0.00
        self.current_time = current_time if current_time is not None else time.time()
        self.last_time = self.current_time

        self.clear()

    def clear(self):
        """Clears PID computations and coefficients"""
        self.SetPoint = 0.0

        self.PTerm = 0.0
        self.ITerm = 0.0
        self.DTerm = 0.0
        self.last_error = 0.0

        # Windup Guard
        self.int_error = 0.0
        self.windup_guard = 20.0

        self.output = 0.0

    def update(self, feedback_value, current_time=None):
        """Calculates PID value for given reference feedback
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        .. figure:: images/pid_1.png
           :align:   center
           Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)
        """
        error = self.SetPoint - feedback_value

        self.current_time = current_time if current_time is not None else time.time()
        delta_time = 1
        delta_error = error - self.last_error


        self.PTerm = self.Kp * error
        self.ITerm += error * delta_time

        if (self.ITerm < -self.windup_guard):
            self.ITerm = -self.windup_guard
        elif (self.ITerm > self.windup_guard):
            self.ITerm = self.windup_guard

        self.DTerm = 0.0
        if delta_time > 0:
            self.DTerm = delta_error / delta_time

        # Remember last time and last error for next calculation
        self.last_time = self.current_time
        self.last_error = error

        self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

    def setKp(self, proportional_gain):
        """Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
        self.Kp = proportional_gain

    def setKi(self, integral_gain):
        """Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
        self.Ki = integral_gain

    def setKd(self, derivative_gain):
        """Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
        self.Kd = derivative_gain

    def setWindup(self, windup):
        """Integral windup, also known as integrator windup or reset windup,
        refers to the situation in a PID feedback controller where
        a large change in setpoint occurs (say a positive change)
        and the integral terms accumulates a significant error
        during the rise (windup), thus overshooting and continuing
        to increase as this accumulated error is unwound
        (offset by errors in the other direction).
        The specific problem is the excess overshooting.
        """
        self.windup_guard = windup

    def setSampleTime(self, sample_time):
        """PID that should be updated at a regular interval.
        Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
        """
        self.sample_time = sample_time

import numpy as np
import matplotlib.pyplot as plt
import random as rn
import environment


targetT = 30
P = 10
I = 1
D = 1
seeds=[30,42,50]
sets=[[10,1,1],
      [8,1,1],
      [6,1,1],
      [10,0.5,1],
      [10,1.5,1],
      [10,2,1],
      [10,1,0.5],
      [10,1,1.5],
      [10,1,2]]
for i in seeds:
    print(i)
    for a in sets:
        print(a)
        pid = PID(a[0],a[1],a[2])
        pid.SetPoint = targetT
        pid.setSampleTime(1)



        np.random.seed(i)
        rn.seed(12345)
        env = environment.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)


        new_month = np.random.randint(0, 12)
        env.reset(new_month=new_month)
        train = False
        env.train = train
        game_over = False
        current_state, _, _ = env.observe()
        timestep = 0
        # STARTING THE LOOP OVER ALL THE TIMESTEPS (1 Timestep = 1 Minute) IN ONE EPOCH
        direction=0
        episodic_reward=0
        temperature =current_state[0]
        times=[]
        temperatures=[]
        temperatures1=[]
        actions=[]
        while timestep <= 12 * 30 * 24 * 60:


            pid.update(temperature)
            action =pid.output
            action=max(min( int(action), 7.5 ),-7.5)
            actions.append(action)
            temperatures.append(temperature)
            temperatures1.append(35)
            #print(targetT, temperature, action)
            if  action<0:
                direction = -1
            else:
                direction = 1
            energy_ai = abs(action)
            # UPDATING THE ENVIRONMENT AND REACHING THE NEXT STATE
            next_state, reward, game_over = env.update_env(direction, energy_ai,
                                                       (new_month + int(timestep / (30 * 24 * 60))) % 12)
            times.append(timestep)
            timestep += 1
            temperature=next_state[0]
            # End this episode when `done` is True

            if timestep % 40 == -1:
                plt.subplot(211)
                plt.plot(times, temperatures,'r-x', label='temperature of data center')
                plt.plot(times, temperatures1, 'g-x', label='Preset temperature')
                plt.xlabel("time step")
                plt.ylabel("Temperature")
                plt.legend()
                plt.subplot(212)
                plt.plot(times, actions,'g-^', label='The temperature changed by PID algorithm')
                plt.xlabel("time step")
                plt.ylabel("Action")
                plt.legend()
                plt.show()
                '''
        plt.plot(x, np.sin(x), )  # 画线并添加图例legend
        plt.plot(x, np.cos(x), 'g-^', label='Cos(x)')  # 画线并添加图例legend
          # 展示图例legend
        plt.xlabel('Rads')  # 给 x 轴添加坐标轴信息
        plt.ylabel('Amplitude')  # 给 y 轴添加坐标轴信息
        plt.title('Sin and Cos Waves')  # 添加图片标题
        # plt.axis('off')   # 关闭坐标轴的显示
        plt.show()
         '''
        print("\n")
        print("Total Energy spent with an AI: {:.0f}".format(env.total_energy_ai))
        print("Total Energy spent with no AI: {:.0f}".format(env.total_energy_noai))
        print("ENERGY SAVED: {:.0f} %".format((env.total_energy_noai - env.total_energy_ai) / env.total_energy_noai * 100))
        print("overflow rate "+ str(env.overflow))
        print("\n")
