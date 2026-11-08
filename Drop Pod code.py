import krpc
import time

conn = krpc.connect()
vessel = conn.space_center.active_vessel
speed = vessel.flight().true_air_speed
alt = vessel.flight().surface_altitude

print(speed)
print('Landing system ONLINE')

while alt > 5:
    time.sleep(.1)
    alt = vessel.flight().surface_altitude
    speed = vessel.flight().true_air_speed 
    
    if alt <= 1000:
        vessel.control.gear = True

    if alt <= 650 and speed > 20:
        vessel.control.activate_next_stage()
        vessel.control.throttle = 1

    if speed < 20 and speed > 10:
        vessel.control.throttle = 0.3
    elif speed < 10:
        vessel.control.throttle = 0.2

    if alt < 4:
        vessel.control.cargo_bays = True


vessel.control.throttle = 0
vessel.control.cargo_bays = True 
        


