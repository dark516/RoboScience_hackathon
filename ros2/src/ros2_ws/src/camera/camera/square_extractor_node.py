import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D, PoseArray, Pose
from std_msgs.msg import Header


class SquareExtractor(Node):
    def __init__(self):
        super().__init__('square_extractor')

        # Create publishers
        self.robot_pose_publisher = self.create_publisher(Pose2D, '/robot/pose', 10)
        self.obstacles_publisher = self.create_publisher(PoseArray, '/obstacles', 10)
        self.red_base_publisher = self.create_publisher(Pose2D, '/red_base', 10)
        self.blue_base_publisher = self.create_publisher(Pose2D, '/blue_base', 10)

        self.points = []
        self.red_point = None
        self.blue_point = None
        self.square_size = 130  # Size of the colored squares around points
        self.output_size = 500
        self.stage = 1  # 1: selecting 4 corners, 2: selecting red/blue points, 3: Detection mode
        self.current_frame = None
        self.warp_matrix = None

        # Store detected objects
        self.detected_red_objects = []
        self.detected_blue_objects = []
        self.detected_arucos = []

        # Default HSV values for color detection (from your settings)
        self.red_lower1 = np.array([0, 165, 62])
        self.red_upper1 = np.array([15, 255, 255])
        self.red_lower2 = np.array([157, 57, 121])
        self.red_upper2 = np.array([180, 154, 202])
        self.blue_lower = np.array([90, 89, 62])
        self.blue_upper = np.array([132, 255, 239])

        # Minimum area threshold
        self.min_area = 100

        # ArUco detection parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Detection flags
        self.show_color_settings = False

        # Timer for publishing (10Hz)
        self.publish_timer = self.create_timer(0.1, self.publish_data)  # 10Hz

    def publish_data(self):
        """Publish all data to ROS2 topics"""
        # Only publish when we're in detection mode and have data
        if self.stage != 3:
            return

        # Publish robot pose from ArUco marker
        if self.detected_arucos:
            # Use the first detected ArUco marker for robot pose
            aruco = self.detected_arucos[0]
            pose_msg = Pose2D()
            pose_msg.x = float(aruco['center'][0])
            pose_msg.y = float(aruco['center'][1])
            pose_msg.theta = float(np.radians(aruco['heading']))  # Convert to radians
            self.robot_pose_publisher.publish(pose_msg)

        # Publish obstacles (red and blue objects)
        obstacles_msg = PoseArray()
        obstacles_msg.header = Header()
        obstacles_msg.header.stamp = self.get_clock().now().to_msg()
        obstacles_msg.header.frame_id = "camera_frame"

        # Add red objects (z=1)
        for center in self.detected_red_objects:
            pose = Pose()
            pose.position.x = float(center[0])
            pose.position.y = float(center[1])
            pose.position.z = 0.0  # Red objects
            obstacles_msg.poses.append(pose)

        # Add blue objects (z=0)
        for center in self.detected_blue_objects:
            pose = Pose()
            pose.position.x = float(center[0])
            pose.position.y = float(center[1])
            pose.position.z = 1.0  # Blue objects
            obstacles_msg.poses.append(pose)

        self.obstacles_publisher.publish(obstacles_msg)

        # Publish red base (heading straight left = 180 degrees = π radians)
        if self.red_point:
            red_base_msg = Pose2D()
            red_base_msg.x = float(self.red_point[0])
            red_base_msg.y = float(self.red_point[1])
            red_base_msg.theta = np.pi  # 180 degrees = π radians (straight left)
            self.red_base_publisher.publish(red_base_msg)

        # Publish blue base (heading straight right = 0 degrees = 0 radians)
        if self.blue_point:
            blue_base_msg = Pose2D()
            blue_base_msg.x = float(self.blue_point[0])
            blue_base_msg.y = float(self.blue_point[1])
            blue_base_msg.theta = 0.0  # 0 degrees = 0 radians (straight right)
            self.blue_base_publisher.publish(blue_base_msg)

    def create_trackbars(self):
        """Create trackbars for adjusting color detection parameters"""
        cv2.namedWindow("Color Settings")

        # Set default values from your settings
        cv2.createTrackbar("Red1 Lower H", "Color Settings", 0, 180, self.nothing)
        cv2.createTrackbar("Red1 Upper H", "Color Settings", 15, 180, self.nothing)
        cv2.createTrackbar("Red1 Lower S", "Color Settings", 120, 255, self.nothing)
        cv2.createTrackbar("Red1 Upper S", "Color Settings", 255, 255, self.nothing)
        cv2.createTrackbar("Red1 Lower V", "Color Settings", 70, 255, self.nothing)
        cv2.createTrackbar("Red1 Upper V", "Color Settings", 255, 255, self.nothing)

        # Red color range 2 trackbars
        cv2.createTrackbar("Red2 Lower H", "Color Settings", 170, 180, self.nothing)
        cv2.createTrackbar("Red2 Upper H", "Color Settings", 180, 180, self.nothing)
        cv2.createTrackbar("Red2 Lower S", "Color Settings", 120, 255, self.nothing)
        cv2.createTrackbar("Red2 Upper S", "Color Settings", 255, 255, self.nothing)
        cv2.createTrackbar("Red2 Lower V", "Color Settings", 56, 255, self.nothing)
        cv2.createTrackbar("Red2 Upper V", "Color Settings", 255, 255, self.nothing)

        # Blue color range trackbars
        cv2.createTrackbar("Blue Lower H", "Color Settings", 90, 180, self.nothing)
        cv2.createTrackbar("Blue Upper H", "Color Settings", 132, 180, self.nothing)
        cv2.createTrackbar("Blue Lower S", "Color Settings", 90, 255, self.nothing)
        cv2.createTrackbar("Blue Upper S", "Color Settings", 255, 255, self.nothing)
        cv2.createTrackbar("Blue Lower V", "Color Settings", 72, 255, self.nothing)
        cv2.createTrackbar("Blue Upper V", "Color Settings", 239, 255, self.nothing)

        # Area threshold trackbar
        cv2.createTrackbar("Min Area", "Color Settings", 100, 1000, self.nothing)

    def nothing(self, x):
        """Dummy function for trackbar callback"""
        pass

    def get_trackbar_values(self):
        """Get current values from all trackbars"""
        # Red range 1
        red1_lh = cv2.getTrackbarPos("Red1 Lower H", "Color Settings")
        red1_uh = cv2.getTrackbarPos("Red1 Upper H", "Color Settings")
        red1_ls = cv2.getTrackbarPos("Red1 Lower S", "Color Settings")
        red1_us = cv2.getTrackbarPos("Red1 Upper S", "Color Settings")
        red1_lv = cv2.getTrackbarPos("Red1 Lower V", "Color Settings")
        red1_uv = cv2.getTrackbarPos("Red1 Upper V", "Color Settings")

        # Red range 2
        red2_lh = cv2.getTrackbarPos("Red2 Lower H", "Color Settings")
        red2_uh = cv2.getTrackbarPos("Red2 Upper H", "Color Settings")
        red2_ls = cv2.getTrackbarPos("Red2 Lower S", "Color Settings")
        red2_us = cv2.getTrackbarPos("Red2 Upper S", "Color Settings")
        red2_lv = cv2.getTrackbarPos("Red2 Lower V", "Color Settings")
        red2_uv = cv2.getTrackbarPos("Red2 Upper V", "Color Settings")

        # Blue range
        blue_lh = cv2.getTrackbarPos("Blue Lower H", "Color Settings")
        blue_uh = cv2.getTrackbarPos("Blue Upper H", "Color Settings")
        blue_ls = cv2.getTrackbarPos("Blue Lower S", "Color Settings")
        blue_us = cv2.getTrackbarPos("Blue Upper S", "Color Settings")
        blue_lv = cv2.getTrackbarPos("Blue Lower V", "Color Settings")
        blue_uv = cv2.getTrackbarPos("Blue Upper V", "Color Settings")

        # Area threshold
        min_area = cv2.getTrackbarPos("Min Area", "Color Settings")

        # Update the color ranges
        self.red_lower1 = np.array([red1_lh, red1_ls, red1_lv])
        self.red_upper1 = np.array([red1_uh, red1_us, red1_uv])
        self.red_lower2 = np.array([red2_lh, red2_ls, red2_lv])
        self.red_upper2 = np.array([red2_uh, red2_us, red2_uv])
        self.blue_lower = np.array([blue_lh, blue_ls, blue_lv])
        self.blue_upper = np.array([blue_uh, blue_us, blue_uv])
        self.min_area = max(10, min_area)  # Ensure minimum area is at least 10

    def mouse_callback_stage1(self, event, x, y, flags, param):
        """Mouse callback for selecting 4 corners"""
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            print(f"Corner point {len(self.points)} selected: ({x}, {y})")

            # Draw the point on the image
            cv2.circle(self.current_frame, (x, y), 5, (0, 255, 0), -1)
            if len(self.points) > 1:
                cv2.line(self.current_frame, self.points[-2], self.points[-1], (0, 255, 0), 2)

            # If 4 points are selected, automatically proceed
            if len(self.points) == 4:
                cv2.line(self.current_frame, self.points[3], self.points[0], (0, 255, 0), 2)
                print("All 4 corners selected! Processing transformation...")

    def mouse_callback_stage2(self, event, x, y, flags, param):
        """Mouse callback for selecting red and blue points"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.red_point is None:
                self.red_point = (x, y)
                print(f"Red point selected: ({x}, {y})")
            elif self.blue_point is None:
                self.blue_point = (x, y)
                print(f"Blue point selected: ({x}, {y})")
                print("Both points selected! Starting detection...")

    def get_square_corners(self, point, image_shape):
        """Calculate square corners around a point, ensuring they stay within image bounds"""
        x, y = point
        half_size = self.square_size // 2

        # Calculate square boundaries
        x1 = max(0, x - half_size)
        y1 = max(0, y - half_size)
        x2 = min(image_shape[1] - 1, x + half_size)
        y2 = min(image_shape[0] - 1, y + half_size)

        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return corners

    def draw_transparent_square(self, image, corners, color, alpha=0.3):
        """Draw a semi-transparent square on the image"""
        overlay = image.copy()
        pts = np.array(corners, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        # Draw border
        cv2.polylines(image, [pts], True, color, 2)

    def order_points(self, pts):
        """Sort points in the order: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    def calculate_perspective_transform(self):
        """Calculate perspective transform matrix for the selected points"""
        if len(self.points) != 4:
            return None

        pts = np.array(self.points, dtype="float32")
        ordered_pts = self.order_points(pts)

        dst_pts = np.array([
            [0, 0],
            [self.output_size - 1, 0],
            [self.output_size - 1, self.output_size - 1],
            [0, self.output_size - 1]
        ], dtype="float32")

        matrix = cv2.getPerspectiveTransform(ordered_pts, dst_pts)
        return matrix

    def extract_square_region(self, frame):
        """Apply perspective transform to extract the square region"""
        if self.warp_matrix is None:
            return frame

        warped = cv2.warpPerspective(frame, self.warp_matrix,
                                     (self.output_size, self.output_size))
        return warped

    def save_coordinates(self):
        """Save all coordinates to variables and print them"""
        if self.red_point and self.blue_point:
            # Get square corners for red point
            red_corners = self.get_square_corners(self.red_point, (self.output_size, self.output_size))

            # Get square corners for blue point
            blue_corners = self.get_square_corners(self.blue_point, (self.output_size, self.output_size))

            print("\n=== SAVED COORDINATES ===")
            print(f"Red point coordinates: {self.red_point}")
            print(f"Red square corners: {red_corners}")
            print(f"Blue point coordinates: {self.blue_point}")
            print(f"Blue square corners: {blue_corners}")
            print("========================\n")

            # You can also return these values if needed
            return {
                'red_point': self.red_point,
                'red_corners': red_corners,
                'blue_point': self.blue_point,
                'blue_corners': blue_corners
            }
        return None

    def heading_from_vector(self, dx, dy):
        """Calculate heading from a vector in degrees"""
        return np.degrees(np.arctan2(dy, dx))

    def wrap_angle_deg(self, angle):
        """Wrap angle to 0-360 degrees"""
        return angle % 360

    def heading_from_aruco_corners(self, corners):
        """Calculate heading from ArUco marker corners"""
        pts = corners.reshape(-1, 2).astype(np.float32)
        TL, TR = pts[0], pts[1]
        dx, dy = TR[0] - TL[0], TR[1] - TL[1]
        return self.heading_from_vector(dx, dy)

    def detect_color_objects(self, frame, exclusion_areas=[]):
        """Detect red and dark blue objects in the frame, excluding specified areas"""
        # Convert to HSV color space for better color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks for red and blue using current trackbar values
        red_mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        red_mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)

        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)

        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
        blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

        # Create exclusion mask
        if exclusion_areas:
            exclusion_mask = np.ones_like(red_mask) * 255
            for area in exclusion_areas:
                pts = np.array(area, np.int32)
                cv2.fillPoly(exclusion_mask, [pts], 0)

            # Apply exclusion mask
            red_mask = cv2.bitwise_and(red_mask, exclusion_mask)
            blue_mask = cv2.bitwise_and(blue_mask, exclusion_mask)

        # Find contours for red objects
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_centers = []

        for contour in red_contours:
            # Filter by area to remove small noise
            area = cv2.contourArea(contour)
            if area > self.min_area:
                # Calculate center of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    red_centers.append((cx, cy))

        # Find contours for blue objects
        blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blue_centers = []

        for contour in blue_contours:
            # Filter by area to remove small noise
            area = cv2.contourArea(contour)
            if area > self.min_area:
                # Calculate center of the contour
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    blue_centers.append((cx, cy))

        # Update the detected objects
        self.detected_red_objects = red_centers
        self.detected_blue_objects = blue_centers

        return red_mask, blue_mask, red_centers, blue_centers

    def detect_aruco_markers(self, frame):
        """Detect ArUco markers in the frame and return their info"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)

        marker_info = []

        if ids is not None:
            for i in range(len(ids)):
                marker_id = ids[i][0]
                marker_corners = corners[i][0]

                # Calculate center of marker
                center_x = int(np.mean(marker_corners[:, 0]))
                center_y = int(np.mean(marker_corners[:, 1]))

                # Calculate heading
                heading = self.heading_from_aruco_corners(marker_corners)

                marker_info.append({
                    'id': marker_id,
                    'corners': marker_corners,
                    'center': (center_x, center_y),
                    'heading': heading
                })

                print(f"Detected ArUco marker ID: {marker_id}, Heading: {heading:.2f}°")

        return corners, ids, marker_info

    def draw_aruco_markers(self, frame, corners, ids, marker_info):
        """Draw detected ArUco markers on the frame with heading lines"""
        if ids is not None:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Add ID labels and heading lines
            for i, marker_id in enumerate(ids):
                id_str = str(marker_id[0])
                center_x = int(np.mean(corners[i][0][:, 0]))
                center_y = int(np.mean(corners[i][0][:, 1]))

                # Draw ID text
                cv2.putText(frame, f"ID:{id_str}", (center_x - 20, center_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Draw heading line
                if i < len(marker_info) and 'heading' in marker_info[i]:
                    heading = marker_info[i]['heading']
                    # Convert heading to radians for trigonometric functions
                    heading_rad = np.radians(heading)
                    # Calculate endpoint of the heading line
                    line_length = 50
                    end_x = int(center_x + line_length * np.cos(heading_rad))
                    end_y = int(center_y + line_length * np.sin(heading_rad))

                    # Draw heading line
                    cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y),
                                    (0, 255, 255), 2, tipLength=0.3)

                    # Draw heading text
                    cv2.putText(frame, f"{heading:.1f}°", (center_x + 25, center_y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    def draw_color_objects(self, frame, red_centers, blue_centers):
        """Draw detected red and blue objects on the frame"""
        # Draw red objects
        for center in red_centers:
            cv2.circle(frame, center, 10, (0, 0, 255), 2)  # Red circle
            cv2.putText(frame, "Red", (center[0] + 15, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw blue objects
        for center in blue_centers:
            cv2.circle(frame, center, 10, (255, 0, 0), 2)  # Blue circle
            cv2.putText(frame, "Blue", (center[0] + 15, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    def run(self):
        cap = cv2.VideoCapture(0)

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTO_WB, 0.0)
        cap.set(cv2.CAP_PROP_EXPOSURE, 200)
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 5000)
        cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
        cap.set(cv2.CAP_PROP_SHARPNESS, 7)
        cap.set(cv2.CAP_PROP_SATURATION, 100)
        cap.set(cv2.CAP_PROP_CONTRAST, 10)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("=== STAGE 1 ===")
        print("Click on 4 points to define the square region")
        print("The transformation will happen automatically after 4 points")

        cv2.namedWindow("Camera Feed")
        cv2.setMouseCallback("Camera Feed", self.mouse_callback_stage1)

        # Stage 1: Select 4 corners
        while self.stage == 1:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            self.current_frame = frame.copy()

            # Draw the points and lines
            for i, point in enumerate(self.points):
                cv2.circle(self.current_frame, point, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(self.current_frame, self.points[i - 1], point, (0, 255, 0), 2)

            # Close the polygon if we have 4 points
            if len(self.points) == 4:
                cv2.line(self.current_frame, self.points[3], self.points[0], (0, 255, 0), 2)

            cv2.imshow("Camera Feed", self.current_frame)

            # Check if we have 4 points to proceed automatically
            if len(self.points) == 4:
                self.warp_matrix = self.calculate_perspective_transform()
                if self.warp_matrix is not None:
                    # Close the original window and move to stage 2
                    cv2.destroyWindow("Camera Feed")
                    self.stage = 2
                    break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

        # Stage 2: Show transformed image and select red/blue points
        print("\n=== STAGE 2 ===")
        print("Click to select RED point (first click)")
        print("Click to select BLUE point (second click)")
        print("Detection will start automatically after both points are selected")

        cv2.namedWindow("Transformed Square")
        cv2.setMouseCallback("Transformed Square", self.mouse_callback_stage2)

        while self.stage == 2:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Apply transformation
            transformed_frame = self.extract_square_region(frame)
            display_frame = transformed_frame.copy()

            # Draw red point and square if selected
            if self.red_point:
                cv2.circle(display_frame, self.red_point, 5, (0, 0, 255), -1)
                red_corners = self.get_square_corners(self.red_point, display_frame.shape)
                self.draw_transparent_square(display_frame, red_corners, (0, 0, 255), 0.3)

            # Draw blue point and square if selected
            if self.blue_point:
                cv2.circle(display_frame, self.blue_point, 5, (255, 0, 0), -1)
                blue_corners = self.get_square_corners(self.blue_point, display_frame.shape)
                self.draw_transparent_square(display_frame, blue_corners, (255, 0, 0), 0.3)

            # Add instructions to the image
            if not self.red_point:
                cv2.putText(display_frame, "Click to select RED point",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif not self.blue_point:
                cv2.putText(display_frame, "Click to select BLUE point",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            else:
                cv2.putText(display_frame, "Starting detection...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Transformed Square", display_frame)

            # Check if both points are selected to proceed automatically
            if self.red_point and self.blue_point:
                print("Both points selected! Starting detection...")
                self.stage = 3
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        # Stage 3: Detection mode (ArUco + Color objects)
        if self.stage == 3:
            print("\n=== STAGE 3 ===")
            print("Detection mode active - detecting ArUco markers and colored objects")
            print("Press 's' to open color settings")
            print("Press 'r' to return to point selection")
            print("Press 'c' to print current detected objects")
            print("Press 'p' to print current color settings")
            print("Press 'q' to quit")

        while self.stage == 3:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Apply transformation
            transformed_frame = self.extract_square_region(frame)
            display_frame = transformed_frame.copy()

            # Get current trackbar values if settings window is open
            if self.show_color_settings:
                self.get_trackbar_values()

            # Detect ArUco markers
            aruco_corners, aruco_ids, aruco_info = self.detect_aruco_markers(transformed_frame)
            self.detected_arucos = aruco_info

            # Prepare exclusion areas: red box, blue box, and ArUco markers
            exclusion_areas = []

            # Add red box area
            if self.red_point:
                red_corners = self.get_square_corners(self.red_point, transformed_frame.shape)
                exclusion_areas.append(red_corners)

            # Add blue box area
            if self.blue_point:
                blue_corners = self.get_square_corners(self.blue_point, transformed_frame.shape)
                exclusion_areas.append(blue_corners)

            # Add ArUco marker areas
            if aruco_corners is not None:
                for corner in aruco_corners:
                    # Convert corners to integer points for exclusion
                    corner_points = corner.reshape(-1, 2).astype(np.int32)
                    exclusion_areas.append(corner_points)

            # Detect red and blue objects (excluding specified areas)
            red_mask, blue_mask, red_centers, blue_centers = self.detect_color_objects(
                transformed_frame, exclusion_areas)

            # Draw detected ArUco markers with heading
            self.draw_aruco_markers(display_frame, aruco_corners, aruco_ids, aruco_info)

            # Draw detected color objects
            self.draw_color_objects(display_frame, red_centers, blue_centers)

            # Draw red and blue points and squares (for reference)
            if self.red_point:
                cv2.circle(display_frame, self.red_point, 5, (0, 0, 255), -1)
                red_corners = self.get_square_corners(self.red_point, display_frame.shape)
                self.draw_transparent_square(display_frame, red_corners, (0, 0, 255), 0.2)

            if self.blue_point:
                cv2.circle(display_frame, self.blue_point, 5, (255, 0, 0), -1)
                blue_corners = self.get_square_corners(self.blue_point, display_frame.shape)
                self.draw_transparent_square(display_frame, blue_corners, (255, 0, 0), 0.2)

            # Add stage info
            cv2.putText(display_frame, "Detection Mode - Press 's' for settings",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Show detection count
            aruco_count = len(aruco_ids) if aruco_ids is not None else 0
            cv2.putText(display_frame, f"ArUco: {aruco_count}, Red: {len(red_centers)}, Blue: {len(blue_centers)}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Transformed Square", display_frame)

            # Show mask windows only if settings are open
            if self.show_color_settings:
                cv2.imshow('red', red_mask)
                cv2.imshow('blue', blue_mask)
            else:
                # Close mask windows if they were open
                if cv2.getWindowProperty('red', cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow('red')
                if cv2.getWindowProperty('blue', cv2.WND_PROP_VISIBLE) >= 1:
                    cv2.destroyWindow('blue')

            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                self.stage = 2
                if self.show_color_settings:
                    cv2.destroyWindow("Color Settings")
                    self.show_color_settings = False
                print("Returning to point selection mode...")
                break
            elif key == ord('c'):
                print("\n=== CURRENT DETECTED OBJECTS ===")
                print(f"Red objects centers: {self.detected_red_objects}")
                print(f"Blue objects centers: {self.detected_blue_objects}")
                if self.detected_arucos:
                    print(
                        f"ArUco markers: {[{'id': marker['id'], 'heading': marker['heading']} for marker in self.detected_arucos]}")
                else:
                    print("ArUco markers: None")
                print("===============================\n")
            elif key == ord('p'):
                print("\n=== CURRENT COLOR SETTINGS ===")
                print(f"Red Range 1: Lower {self.red_lower1}, Upper {self.red_upper1}")
                print(f"Red Range 2: Lower {self.red_lower2}, Upper {self.red_upper2}")
                print(f"Blue Range: Lower {self.blue_lower}, Upper {self.blue_upper}")
                print(f"Min Area: {self.min_area}")
                print("===============================\n")
            elif key == ord('s'):
                if not self.show_color_settings:
                    self.create_trackbars()
                    self.show_color_settings = True
                    print("Color settings opened. Adjust sliders and press 's' again to close.")
                else:
                    cv2.destroyWindow("Color Settings")
                    self.show_color_settings = False
                    print("Color settings closed.")
            elif key == ord('q'):
                break

            # Process ROS2 callbacks
            rclpy.spin_once(self, timeout_sec=0.001)

        cap.release()
        cv2.destroyAllWindows()


def main():
    rclpy.init()
    node = SquareExtractor()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
