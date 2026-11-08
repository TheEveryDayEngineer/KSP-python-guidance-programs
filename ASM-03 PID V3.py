
# ASM-03 PID V3.py
'''
This script implements a PID guidance system for the ASM-03 missile in Kerbal Space Program using kRPC.
It uses 4 PID controllers to manage pitch, target pitch, roll, and yaw towards the target.
This is Version 3, which adds an initial roll phase immediately after launch to orient the missile towards the target before pitching over.
The PID constants have been tuned for better performance as well. 

Version completed on 04-05-2026

'''
import krpc
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math


conn = krpc.connect()
vessel = conn.space_center.active_vessel
target = conn.space_center.target_vessel
pitch = vessel.flight().pitch

# Constants #########################################################################################
CRUISE_THROTTLE = 0.5
CRUISE_ALTITUDE = 50

# Functions #########################################################################################

# PID control function to maintain desired altitude
def PID_control(desired_value, current_value,
                kp, ki, kd,
                max_integral, min_integral,
                max_output, min_output,
                last_time, integral, previous_error):

    # compute dt
    now = time.time()
    if last_time is None:
        dt = 0.1  # or whatever your expected loop rate is (e.g., 0.01 for 100Hz)
    else:
        dt = now - last_time

    # error term
    error = desired_value - current_value

    # integral term
    integral += error * dt
    integral = max(min(integral, max_integral), min_integral)

    # derivative term
    derivative = (error - previous_error) / dt if dt > 0 else 0

    # PID output
    output = kp * error + ki * integral + kd * derivative

    # clamp
    output = max(min(output, max_output), min_output)

    # return output and updated state
    return output, now, integral, error


###################################################################################################################################
#                                               Launch Phase with initial roll                                                    #
###################################################################################################################################
def launch_phase():
    print("Launching missile...")

    target_position = target.position(vessel.reference_frame) # gets target position

    vessel.control.throttle = 1
    vessel.control.activate_next_stage() # Activate first stage

    # Short roll control loop immediately after launch to roll toward the target
    launch_roll_kp = 0.05
    launch_roll_ki = 0.0
    launch_roll_kd = 0.015
    launch_roll_max_integral = 10
    launch_roll_min_integral = -10
    launch_roll_max_output = 1
    launch_roll_min_output = -1

    launch_roll_last_time = None
    launch_roll_integral = 0
    launch_roll_previous_error = 0

    roll_start = time.perf_counter()
    print("Performing initial roll towards target...")

    while time.perf_counter() - roll_start < 5:
      target_position = target.position(vessel.reference_frame)
      tx, ty, tz = target_position
      launch_angle_to_target = -1 *angle_from_positive_y_axis(tx, tz)  # Calculate angle to target in horizontal plane

      control_roll, launch_roll_last_time, launch_roll_integral, launch_roll_previous_error = PID_control(
        0, launch_angle_to_target,
        kp=launch_roll_kp, ki=launch_roll_ki, kd=launch_roll_kd,
        max_integral=launch_roll_max_integral, min_integral=launch_roll_min_integral,
        max_output=launch_roll_max_output, min_output=launch_roll_min_output,
        last_time=launch_roll_last_time, integral=launch_roll_integral, previous_error=launch_roll_previous_error)

      vessel.control.roll = control_roll 
      time.sleep(0.05)

    print("Initial roll complete. Proceeding to pitch over...")
    vessel.control.roll = 0  # Stop roll control after initial phase

    #Pitch over after the initial roll
    vessel.control.pitch = -1
    time.sleep(4)
    vessel.control.pitch = 0
    vessel.control.roll = 0

###################################################################################################################################
#                                               Cruise phase with PID control                                                     #
###################################################################################################################################

def cruise_phase():
    program_start = time.perf_counter() # marks the start time of the program

    print("Entering cruise phase...")
    vessel.control.throttle = CRUISE_THROTTLE  # Set throttle to a cruising value

    # cruise phase Settings
    # Pitch control PID constants
    pitch_kp = 0.025
    pitch_ki = 0.025
    pitch_kd = 0.003
    pitch_max_integral = 20
    pitch_min_integral = -100
    pitch_max_output = 1
    pitch_min_output = -1

    pitch_last_time = None
    pitch_integral = 0
    pitch_previous_error = 0

    # pitch target PID constants
    target_pitch_kp = 0.5
    target_pitch_ki = 0.03
    target_pitch_kd = 0.3
    target_pitch_max_integral = 100
    target_pitch_min_integral = -100
    target_pitch_max_output = 10
    target_pitch_min_output = -5

    target_pitch_last_time = 0
    target_pitch_integral = 0
    target_pitch_previous_error = 0

    # Roll PID constants
    roll_kp = 0.01
    roll_ki = 0
    roll_kd = 0
    roll_max_integral = 100
    roll_min_integral = -100
    roll_max_output = 0.5
    roll_min_output = -0.5

    roll_last_time = 0
    roll_integral = 0
    roll_previous_error = 0

    # to target PID constants
    to_target_kp = 0.02
    to_target_ki = 0.05
    to_target_kd = 0.005
    to_target_max_integral = 20
    to_target_min_integral = -20
    to_target_max_output = 1
    to_target_min_output = -1

    to_target_last_time = None
    to_target_integral = 0
    to_target_previous_error = 0

    # Data logging lists
    time_stamp = []
    integral_data = []
    control_pitch_data = []
    pitch_data = []
    target_pitch_data = []
    pitch_previous_error_data = []

    while True:
        alt = vessel.flight().surface_altitude # gets altitude in meters
        pitch = vessel.flight().pitch # gets pitch in degrees
        roll = vessel.flight().roll # gets roll in degrees
        speed = vessel.flight().true_air_speed # gets speed in m/s
        target_position = target.position(vessel.reference_frame) # gets target position

        if distance_to_target(target_position[0], target_position[1]) < 418:
          break

        #pitch target PID
        target_pitch, target_pitch_last_time, target_pitch_integral, target_pitch_previous_error = PID_control(CRUISE_ALTITUDE, alt, kp=target_pitch_kp, ki=target_pitch_ki,
                                            kd=target_pitch_kd, max_integral=target_pitch_max_integral,
                                              min_integral=target_pitch_min_integral, max_output=target_pitch_max_output,
                                                min_output=target_pitch_min_output, last_time=target_pitch_last_time, integral=target_pitch_integral,
                                                  previous_error=target_pitch_previous_error)

        
        # control Pitch PID
        control_pitch, pitch_last_time, pitch_integral, pitch_previous_error = PID_control(target_pitch, pitch, kp=pitch_kp, ki=pitch_ki,
                                            kd=pitch_kd, max_integral=pitch_max_integral,
                                              min_integral=pitch_min_integral, max_output=pitch_max_output,
                                                min_output=pitch_min_output, last_time=pitch_last_time, integral=pitch_integral,
                                                  previous_error=pitch_previous_error)
        # Apply pitch control
        vessel.control.pitch = control_pitch

        # Roll PID
        control_roll, roll_last_time, roll_integral, roll_previous_error = PID_control(0, roll, kp=roll_kp, ki=roll_ki,
                                           kd=roll_kd, max_integral=roll_max_integral,
                                             min_integral=roll_min_integral, max_output=roll_max_output,
                                               min_output=roll_min_output, last_time=roll_last_time, integral=roll_integral,
                                                 previous_error=roll_previous_error)
        # Apply roll control
        vessel.control.roll = control_roll

        # guidence to target
        if (time.perf_counter() - program_start) >= 10:
          target_x, target_y, target_z = target_position #unpacking reference frame values
          # Calculate angle to target in horizontal plane
          angle_to_target = angle_from_positive_y_axis(target_x, target_y)
          print("Angle to target: ", angle_to_target)

          # yaw PID
          control_yaw, to_target_last_time, to_target_integral, to_target_previous_error = PID_control(0.2, angle_to_target, kp=to_target_kp, ki=to_target_ki,
                                            kd=to_target_kd, max_integral=to_target_max_integral,
                                              min_integral=to_target_min_integral, max_output=to_target_max_output,
                                                min_output=to_target_min_output, last_time=to_target_last_time, integral=to_target_integral,
                                                  previous_error=to_target_previous_error)
          # Apply yaw control
          vessel.control.yaw = control_yaw

        # Log data
        time_stamp.append(time.perf_counter() - program_start)
        integral_data.append(pitch_integral)
        control_pitch_data.append(control_pitch)
        pitch_data.append(pitch)
        pitch_previous_error_data.append(pitch_previous_error)
        

        if time.perf_counter() - program_start >= 10000: # plots and breaks after 10000 seconds
            # Create subplots  
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                subplot_titles=("Target pitch", "Control Pitch Over Time", "Pitch Error"))

            # Add integral data
            fig.add_trace(go.Scatter(x=time_stamp, y=integral_data, mode='lines', name='Target Pitch'), row=1, col=1)

            # Add control pitch data
            fig.add_trace(go.Scatter(x=time_stamp, y=control_pitch_data, mode='lines', name='Control Pitch'), row=2, col=1)

            # Add pitch data
            fig.add_trace(go.Scatter(x=time_stamp, y=pitch_data, mode='lines', name='Pitch'), row=2, col=1)
            fig.add_trace(go.Scatter(x=time_stamp, y=pitch_previous_error_data, mode='lines', name='Pitch Error'), row=2, col=1)
            fig.show()
            break
        
        #print("control_pitch: ", control_pitch, "          actual pitch: ", pitch)
        time.sleep(0.05)  # Small delay to prevent excessive CPU usage

#Returns the angle in degrees from the positive y-axis to the point (x, y), measured counter-clockwise.############################################################################################
def angle_from_positive_y_axis(x: float, y: float) -> float:
    """
    Returns the angle in degrees from the positive y-axis to the point (x, y),
    measured counter-clockwise.
    
    Examples:
        (0, 1)  -> 0°    (straight up, on positive y-axis)
        (1, 0)  -> 90°   (positive x-axis)
        (0, -1) -> 180°  (negative y-axis)
        (-1, 0) -> 270°  (negative x-axis)
    
    Returns value in range [0, 360).
    """
    # math.atan2(y, x) gives angle from positive x-axis to (x, y)
    angle_from_x = math.degrees(math.atan2(y, x))
    
    # Convert to angle from positive y-axis (counter-clockwise)
    angle_from_y = (angle_from_x - 90) % 360
    if angle_from_y > 180:
        angle_from_y -= 360  # Convert to range [-180, 180)
    
    return angle_from_y

# calcs distance to target ############################################################################################
def distance_to_target(x,y):
    return math.sqrt(x**2 + y**2)

###################################################################################################################################
#                                                   Terminal Phase                                                                #
###################################################################################################################################
def terminal_phase():
    print("Entering terminal phase...")
    while True:
      vessel.control.pitch = -1

###################################################################################################################################
#                                                       Main Execution                                                            #
###################################################################################################################################
print("hello world")
launch_phase()

time.sleep(5)  # Wait for some time before entering cruise phase

cruise_phase()

terminal_phase()

