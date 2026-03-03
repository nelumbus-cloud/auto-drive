import carla
import time
import random
from config import get_args
from perception.sensor_manager import SensorManager
from planning.planner import Planner
from control.controller import Controller

def main():
    config = get_args()
    
    client = carla.Client(config.host, config.port)
    # client.set_timeout(30.0)  # Generous timeout for heavy inference workload
    world = client.get_world()
    original_settings = world.get_settings()

    # Enable synchronous mode with a fixed time step
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.1  # 10 FPS simulation
    world.apply_settings(settings)
    
    # Setup Traffic Manager in synchronous mode to match
    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)

    blueprint_library = world.get_blueprint_library()
    bp = blueprint_library.filter('model3')[0]
    
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = random.choice(spawn_points)
    
    vehicle = world.spawn_actor(bp, spawn_point)
    
    # Enable autopilot for ego vehicle
    vehicle.set_autopilot(True, tm.get_port())

    # Spawn some traffic
    actors = [vehicle]
    traffic_points = world.get_map().get_spawn_points()
    random.shuffle(traffic_points)
    
    # Spawn 5 cars
    spawned_cars = 0
    for tp in traffic_points:
        if spawned_cars >= 5:
            break
        if tp.location.distance(spawn_point.location) < 50.0 and tp.location.distance(spawn_point.location) > 5.0:
            v_bp = random.choice(blueprint_library.filter('vehicle.*'))
            v = world.try_spawn_actor(v_bp, tp)
            if v:
                actors.append(v)
                v.set_autopilot(True, tm.get_port())
                spawned_cars += 1
    
    # Spawn 5 pedestrians
    p_bps = blueprint_library.filter('walker.pedestrian.*')
    for i in range(5):
        p_bp = random.choice(p_bps)
        for _ in range(10):
            p_tp = world.get_random_location_from_navigation()
            if p_tp and p_tp.distance(spawn_point.location) < 30.0:
                p = world.try_spawn_actor(p_bp, carla.Transform(p_tp))
                if p:
                    actors.append(p)
                    break

    try:
        # Initialize modules
        perception = SensorManager(world, vehicle, config=config)
        planner = Planner()
        control = Controller(vehicle)

        # Setup sensors
        perception.spawn_cameras(n=4)
        perception.spawn_lidar()
        perception.spawn_gnss()
        perception.spawn_imu()

        print("System ready. Driving with Autopilot and YOLOPv2...")
        
        for _ in range(600): # Run for ~60 seconds at 10 FPS
            try:
                world.tick()
            except RuntimeError:
                pass  # Skip transient CARLA timeouts

        time.sleep(2.0) # Wait for final frames to be processed

    finally:
        print("Cleaning up...")
        if 'perception' in locals():
            perception.destroy_all()
        for actor in actors:
            if actor.is_alive:
                actor.destroy()
        # Restore original world settings
        world.apply_settings(original_settings)
        print("Done.")

if __name__ == '__main__':
    main()
