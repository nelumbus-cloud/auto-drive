import carla

class Controller:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        print("Initializing Controller")

    def apply_control(self, decision):
        if decision == "FORWARD":
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.0))
        else:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
