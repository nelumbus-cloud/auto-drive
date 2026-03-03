import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Autonomous Driving Deployment Platform Configuration")

    # CARLA settings
    parser.add_argument("--host", default="127.0.0.1", help="IP of the host server")
    parser.add_argument("--port", type=int, default=2000, help="TCP port to listen to")
    parser.add_argument("--timeout", type=float, default=10.0, help="Timeout for connection")

    # DETECTION
    parser.add_argument("--obstacle_detection", action="store_true", default=True, help="enable/disable obstacle detection")
    parser.add_argument("--obstacle_detection_model_paths", default="perception/pretrained-models/yolopv2.pt", help="path to detection model")
    parser.add_argument("--obstacle_detection_min_score_threshold", type=float, default=0.5, help="min score threshold for detection")
    parser.add_argument("--perfect_obstacle_detection", action="store_true", help="use ground truth for detection")
    parser.add_argument("--dynamic_obstacle_distance_threshold", type=float, default=50.0, help="threshold for dynamic obstacles (m)")
    parser.add_argument("--static_obstacle_distance_threshold", type=float, default=70.0, help="threshold for static obstacles (m)")

    # TRACKING
    parser.add_argument("--obstacle_tracking", action="store_true", help="enable/disable tracking")
    parser.add_argument("--tracker_type", choices=["da_siam_rpn", "deep_sort", "center_track"], default="deep_sort")
    parser.add_argument("--tracking_num_steps", type=int, default=10, help="history limit")
    parser.add_argument("--min_matching_iou", type=float, default=0.5)
    parser.add_argument("--obstacle_track_max_age", type=int, default=3)

    # TRAFFIC LIGHTS
    parser.add_argument("--traffic_light_detection", action="store_true", help="enable/disable traffic light detection")
    parser.add_argument("--traffic_light_det_model_path", default="")
    parser.add_argument("--traffic_light_det_min_score_threshold", type=float, default=0.5)

    # LANES
    parser.add_argument("--lane_detection", action="store_true", default=True, help="enable/disable lane detection")
    parser.add_argument("--lane_detection_type", choices=["lanenet", "canny", "yolopv2"], default="yolopv2")
    parser.add_argument("--perfect_lane_detection", action="store_true")

    # SEGMENTATION
    parser.add_argument("--segmentation", action="store_true", default=True, help="enable/disable drivable area segmentation")
    parser.add_argument("--segmentation_model_path", default="perception/pretrained-models/yolopv2.pt")

    # VISUALIZATION
    parser.add_argument("--visualize_detected_obstacles", action="store_true", default=True)
    parser.add_argument("--visualize_detected_traffic_lights", action="store_true")
    parser.add_argument("--visualize_lane_detection", action="store_true", default=True)
    parser.add_argument("--output_dir", default="visualization", help="Directory to save output videos/images")

    return parser.parse_args()
