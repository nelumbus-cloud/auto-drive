import carla
import numpy as np
import cv2
import os
import threading

from perception.yolopv2_detector import YOLOPv2Detector

class SensorManager:
    def __init__(self, world, vehicle, config=None):
        self.world = world
        self.vehicle = vehicle
        self.config = config
        self.blueprint_library = world.get_blueprint_library()
        self.sensors = []
        self.camera_data = {}
        self.detector = None
        self.video_writers = {}
        self._lock = threading.Lock()
        self._active = True
        if config and config.obstacle_detection:
            self.detector = YOLOPv2Detector(config.obstacle_detection_model_paths)

    def spawn_cameras(self, n=4):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '90')

        # Simple arrangement: Front, Back, Left, Right
        transforms = [
            carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(yaw=0)),    # Front
            carla.Transform(carla.Location(x=-1.5, z=2.4), carla.Rotation(yaw=180)), # Back
            carla.Transform(carla.Location(y=-0.8, z=2.4), carla.Rotation(yaw=-90)), # Left
            carla.Transform(carla.Location(y=0.8, z=2.4), carla.Rotation(yaw=90))    # Right
        ]

        for i in range(min(n, len(transforms))):
            cam = self.world.spawn_actor(camera_bp, transforms[i], attach_to=self.vehicle)
            # Initialize video writer
            output_dir = self.config.output_dir if self.config else "outputs"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            video_name = os.path.join(output_dir, f'camera_{i}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writers[i] = cv2.VideoWriter(video_name, fourcc, 10.0, (1280, 720))
            
            cam.listen(lambda image, idx=i: self._camera_callback(image, idx))
            self.sensors.append(cam)
            print(f"Spawned Camera {i} (Recording to {video_name})")

    def spawn_lidar(self):
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.5))
        lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        lidar.listen(lambda data: self._lidar_callback(data))
        self.sensors.append(lidar)
        print("Spawned LiDAR")

    def spawn_gnss(self):
        gnss_bp = self.blueprint_library.find('sensor.other.gnss')
        gnss = self.world.spawn_actor(gnss_bp, carla.Transform(), attach_to=self.vehicle)
        gnss.listen(lambda data: self._gnss_callback(data))
        self.sensors.append(gnss)
        print("Spawned GNSS")

    def spawn_imu(self):
        imu_bp = self.blueprint_library.find('sensor.other.imu')
        imu = self.world.spawn_actor(imu_bp, carla.Transform(), attach_to=self.vehicle)
        imu.listen(lambda data: self._imu_callback(data))
        self.sensors.append(imu)
        print("Spawned IMU")

    def _camera_callback(self, image, idx):
        with self._lock:
            if not self._active:
                return
                
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        
        # Inference with YOLOPv2
        if self.detector is not None:
            drivable, lanes, detections = self.detector.infer(array)
            if self.config and (self.config.visualize_lane_detection or self.config.visualize_detected_obstacles):
                array = self.detector.visualize(array, drivable, lanes, detections)

        self.camera_data[idx] = array
        
        # Write to video
        with self._lock:
            if self._active and idx in self.video_writers:
                # Force resize to match VideoWriter initialization (1280x720)
                if (array.shape[1], array.shape[0]) != (1280, 720):
                    array = cv2.resize(array, (1280, 720))
                self.video_writers[idx].write(array)

    def _lidar_callback(self, data):
        pass

    def _gnss_callback(self, data):
        pass

    def _imu_callback(self, data):
        pass

    def destroy_all(self):
        with self._lock:
            self._active = False
        for s in self.sensors:
            if s.is_alive:
                s.stop()
                s.destroy()
        with self._lock:
            for w in self.video_writers.values():
                w.release()
            self.video_writers = {}
        self.sensors = []
