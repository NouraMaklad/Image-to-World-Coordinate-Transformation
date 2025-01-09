import numpy as np
import cv2
import glob
import pybullet as p
import pybullet_data
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class CameraCalibrator:
    def __init__(self):
        self.chessboard_size = (6, 8)
        self.square_size = 25.0  # mm
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Prepare object points
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1,2)
        self.objp = self.objp * self.square_size
        
        self.objpoints = []
        self.imgpoints = []
        
        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        
        # Store test image data
        self.test_image = None
        self.test_corners = None

    def calibrate_camera(self, calibration_path):
        """Calibrate camera using chessboard images"""
        print("Starting camera calibration...")
        images = glob.glob(calibration_path)
        
        if not images:
            print(f"No images found in {calibration_path}")
            return False
            
        print(f"Found {len(images)} calibration images")
        
        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            if img is None:
                print(f"Failed to load image: {fname}")
                continue
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            
            if ret:
                self.objpoints.append(self.objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
                self.imgpoints.append(corners2)
                
                # Draw and display corners
                calibration_img = img.copy()
                cv2.drawChessboardCorners(calibration_img, self.chessboard_size, corners2, ret)
                
                # Display image dimensions
                height, width = img.shape[:2]
                print(f"\nImage {idx+1} dimensions: {width}x{height}")
                
                cv2.imshow(f'Calibration Image {idx+1}', calibration_img)
                cv2.waitKey(500)
                
        cv2.destroyAllWindows()
        
        if len(self.objpoints) > 0:
            print("\nPerforming camera calibration...")
            ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
                self.objpoints, self.imgpoints, gray.shape[::-1], None, None)
            
            print("\nCalibration Results:")
            print("Camera Matrix:")
            print(self.camera_matrix)
            print("\nDistortion Coefficients:")
            print(self.dist_coeffs)
            
            # Calculate reprojection error
            mean_error = 0
            for i in range(len(self.objpoints)):
                imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], 
                                                self.tvecs[i], self.camera_matrix, 
                                                self.dist_coeffs)
                error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                mean_error += error
            
            print(f"\nMean reprojection error: {mean_error/len(self.objpoints)}")
            return True
        
        return False

    def analyze_test_image(self, test_image_path):
        """Analyze test image and display results"""
        self.test_image = cv2.imread(test_image_path)
        if self.test_image is None:
            print(f"Failed to load test image: {test_image_path}")
            return None
            
        gray = cv2.cvtColor(self.test_image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
        
        if ret:
            self.test_corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
            
            # Create visualization
            fig = plt.figure(figsize=(15, 5))
            
            # Original image with corners
            ax1 = fig.add_subplot(131)
            marked_img = self.test_image.copy()
            cv2.drawChessboardCorners(marked_img, self.chessboard_size, self.test_corners, ret)
            ax1.imshow(cv2.cvtColor(marked_img, cv2.COLOR_BGR2RGB))
            ax1.set_title('Detected Corners')
            
            # Undistorted image
            ax2 = fig.add_subplot(132)
            undistorted_img = cv2.undistort(self.test_image, self.camera_matrix, 
                                        self.dist_coeffs, None, self.camera_matrix)
            ax2.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
            ax2.set_title('Undistorted Image')
            
            # 3D visualization of corners
            ax3 = fig.add_subplot(133, projection='3d')
            world_points = self.get_world_coordinates(self.test_corners)
            if world_points is not None:
                ax3.scatter(world_points[:, 0], world_points[:, 1], world_points[:, 2])
                ax3.set_title('3D Corner Positions')
                ax3.set_xlabel('X (mm)')
                ax3.set_ylabel('Y (mm)')
                ax3.set_zlabel('Z (mm)')
            
            plt.tight_layout()
            plt.show()
            
            # Print corner coordinates
            print("\nTest Image Analysis:")
            print(f"Image dimensions: {self.test_image.shape[1]}x{self.test_image.shape[0]}")
            print("\nCorner Coordinates (first 5 corners):")
            for i in range(min(5, len(self.test_corners))):
                pixel_coord = self.test_corners[i][0]
                world_coord = self.pixel_to_world(pixel_coord)
                if world_coord is not None:
                    print(f"\nCorner {i+1}:")
                    print(f"Pixel coordinates: ({pixel_coord[0]:.2f}, {pixel_coord[1]:.2f})")
                    print(f"World coordinates (mm): X={world_coord[0]:.2f}, Y={world_coord[1]:.2f}, Z={world_coord[2]:.2f}")
            
            return undistorted_img
        
        print("Failed to find chessboard corners in test image")
        return None

    def pixel_to_world(self, pixel_coord):
        """Convert a single pixel coordinate to world coordinate"""
        try:
            uv = np.array([[pixel_coord[0], pixel_coord[1]]], dtype=np.float32)
            uv = cv2.undistortPoints(uv, self.camera_matrix, self.dist_coeffs)
            
            # Use last calibration image's rotation and translation
            R, _ = cv2.Rodrigues(self.rvecs[-1])
            t = self.tvecs[-1]
            
            ray = np.array([uv[0][0][0], uv[0][0][1], 1.0])
            ray = ray / np.linalg.norm(ray)
            
            # Assume Z=0 plane for chessboard
            scale = -t[2] / (R[2].dot(ray))
            world_point = t + scale * R.dot(ray)
            
            return world_point.flatten()  # Ensure 1D array
        except Exception as e:
            print(f"Error in coordinate conversion: {e}")
            return None

    def get_world_coordinates(self, pixel_points):
        """Convert pixel coordinates to world coordinates"""
        world_points = []
        for point in pixel_points:
            world_point = self.pixel_to_world(point[0])
            if world_point is not None:
                world_points.append(world_point)
        
        if world_points:
            return np.array(world_points)
        return None
    def generate_random_target(self):
        """Generate a random target point within the chessboard area"""
        if self.test_corners is not None:
            # Get the bounds of the chessboard from detected corners
            corners = self.test_corners.reshape(-1, 2)
            min_x = np.min(corners[:, 0])
            max_x = np.max(corners[:, 0])
            min_y = np.min(corners[:, 1])
            max_y = np.max(corners[:, 1])
            
            # Generate random point within bounds
            random_pixel_x = np.random.uniform(min_x, max_x)
            random_pixel_y = np.random.uniform(min_y, max_y)
            
            return np.array([random_pixel_x, random_pixel_y])
        return None
class PyBulletSimulator:
    def __init__(self, camera_matrix, dist_coeffs, calibrator, test_image):
        # Initialize PyBullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Store calibration data
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.calibrator = calibrator
        self.test_image = test_image
        
        # Create smaller ground plane
        self.create_custom_plane()
        
        # Load and scale robot
        self.robot_id = self.create_scaled_robot()
        
        # Navigation parameters
        self.target_pos = None
        self.moving = False
        self.target_marker = None
        
        # Store calibrated points
        self.calibrated_points = self.get_calibrated_points()
        self.current_point_index = 0
        
        # Setup camera view
        self.setup_camera_view()

        # Add debug parameter for selecting next point
        self.next_point_button = p.addUserDebugParameter("Move to Next Calibrated Point", 1, 0, 1)
        self.last_button_value = 1

    def create_scaled_robot(self):
        """Create a scaled-down version of the robot"""
        # Start position slightly above ground
        start_pos = [0, 0, 0.005]
        robot_id = p.loadURDF("r2d2.urdf", start_pos, 
                             globalScaling=0.02)  # Scale the entire robot
        
        # Change color for better visibility
        for joint in range(-1, p.getNumJoints(robot_id)):
            p.changeVisualShape(
                robot_id,
                joint,
                rgbaColor=[0, 0.5, 0.8, 1]
            )
        
        return robot_id

    def get_calibrated_points(self):
        """Get all calibrated points from the chessboard corners"""
        calibrated_points = []
        
        if self.calibrator.test_corners is not None:
            corners = self.calibrator.test_corners.reshape(-1, 2)
            rows, cols = self.calibrator.chessboard_size  # (6,8)
            
            # Calculate grid parameters
            grid_size = 0.2  # Total size of the grid in meters
            cell_size = grid_size / max(rows, cols)
            
            # Calculate offsets to center the grid
            offset_x = -cell_size * (cols - 1) / 2
            offset_y = -cell_size * (rows - 1) / 2
            
            # Reshape corners to match chessboard layout
            corners_grid = corners.reshape(rows, cols, 2)
            
            # Create ordered list of corners starting from top-left
            ordered_corners = []
            for row in range(rows):
                for col in range(cols):
                    ordered_corners.append(corners_grid[row, col])
            
            # Set initial robot position to match first corner
            first_corner = ordered_corners[0]
            first_world_coord = self.calibrator.pixel_to_world(first_corner)
            if first_world_coord is not None:
                x = offset_x
                y = offset_y + (rows - 1) * cell_size  # Start from top row
                initial_pos = [x, y, 0.005]
                # Reset robot position to match first corner
                p.resetBasePositionAndOrientation(
                    self.robot_id,
                    initial_pos,
                    p.getQuaternionFromEuler([0, 0, 0])
                )
            
            # Process corners in order
            for i in range(len(ordered_corners)):
                pixel_coord = ordered_corners[i]
                world_coord = self.calibrator.pixel_to_world(pixel_coord)
                
                if world_coord is not None:
                    # Calculate grid position
                    col = i % cols
                    row = i // cols
                    
                    # Calculate world position
                    x = offset_x + col * cell_size
                    y = offset_y + (rows - 1 - row) * cell_size  # Start from top
                    z = 0.005
                    
                    world_pos = [x, y, z]
                    
                    calibrated_points.append({
                        'pixel': pixel_coord,
                        'world': world_pos,
                        'world_mm': world_coord,
                        'grid_pos': (row, col)
                    })
                    
                    print(f"Corner {i}: Grid({row}, {col}) -> World({x:.3f}, {y:.3f})")
        
        print(f"\nFound {len(calibrated_points)} calibrated points")
        return calibrated_points

    def create_custom_plane(self):
        """Create a plane with grid matching chessboard corners"""
        self.plane_id = p.loadURDF("plane.urdf", [0, 0, 0])
        p.changeVisualShape(self.plane_id, -1, rgbaColor=[0.9, 0.9, 0.9, 1])
        
        # Grid parameters matching calibrated points
        rows, cols = self.calibrator.chessboard_size
        grid_size = 0.2
        cell_size = grid_size / max(rows, cols)
        
        # Calculate grid boundaries
        start_x = -cell_size * (cols - 1) / 2
        start_y = -cell_size * (rows - 1) / 2
        end_x = cell_size * (cols - 1) / 2
        end_y = cell_size * (rows - 1) / 2
        
        # Create grid lines
        for i in range(cols + 1):
            x = start_x + i * cell_size
            p.addUserDebugLine([x, start_y, 0.001], [x, end_y, 0.001], [0.5, 0.5, 0.5])
        
        for i in range(rows + 1):
            y = start_y + i * cell_size
            p.addUserDebugLine([start_x, y, 0.001], [end_x, y, 0.001], [0.5, 0.5, 0.5])

        # Add coordinate labels for debugging
        for i in range(rows):
            for j in range(cols):
                x = start_x + j * cell_size
                y = start_y + (rows - 1 - i) * cell_size  # Invert row numbering
                label = f"({i},{j})"
                p.addUserDebugText(label, [x, y, 0.002], [0, 0, 0], 0.01)

    def setup_camera_view(self):
        """Setup camera view"""
        p.resetDebugVisualizerCamera(
            cameraDistance=0.4,    # Adjusted for better view
            cameraYaw=0,          # Changed to match chessboard orientation
            cameraPitch=-60,      # More top-down view
            cameraTargetPosition=[0, 0, 0]
        )
        
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)

    def move_to_next_point(self):
        """Set next calibrated point as target"""
        if self.calibrated_points:
            point_data = self.calibrated_points[self.current_point_index]
            
            pixel_coord = point_data['pixel']
            world_coord_mm = point_data['world_mm']
            world_pos = point_data['world']
            grid_pos = point_data['grid_pos']
            
            print("\nMoving to Calibrated Point", self.current_point_index + 1)
            print(f"Grid Position: Row {grid_pos[0]}, Col {grid_pos[1]}")
            print("\nCamera Perspective (before calibration):")
            print(f"Pixel X: {pixel_coord[0]:.2f}")
            print(f"Pixel Y: {pixel_coord[1]:.2f}")
            
            print("\nWorld Coordinates (after calibration):")
            print(f"X: {world_coord_mm[0]:.2f} mm ({world_pos[0]:.4f} m)")
            print(f"Y: {world_coord_mm[1]:.2f} mm ({world_pos[1]:.4f} m)")
            print(f"Z: {world_coord_mm[2]:.2f} mm ({world_pos[2]:.4f} m)")
            
            self.target_pos = world_pos
            self.create_target_marker(world_pos)
            self.moving = True
            
            self.current_point_index = (self.current_point_index + 1) % len(self.calibrated_points)
            
            # Visualize point on image
            img_copy = self.test_image.copy()
            cv2.circle(img_copy, (int(pixel_coord[0]), int(pixel_coord[1])), 
                    5, (0, 0, 255), -1)
            cv2.imshow('Current Target Point', img_copy)
            cv2.waitKey(1)
            
    def update(self):
        """Update robot position"""
        if self.moving and self.target_pos is not None:
            current_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
            direction = np.array(self.target_pos) - np.array(current_pos)
            distance = np.linalg.norm(direction[:2])  # Only consider X-Y plane
            
            if distance > 0.001:  # Adjusted threshold
                # Calculate movement speed
                speed = min(0.002, distance * 0.2)  # Increased speed and acceleration
                direction_normalized = direction / np.linalg.norm(direction)
                new_pos = np.array(current_pos) + direction_normalized * speed
                
                # Update robot position and orientation
                p.resetBasePositionAndOrientation(
                    self.robot_id,
                    [new_pos[0], new_pos[1], 0.005],  # Fixed height
                    p.getQuaternionFromEuler([0, 0, np.arctan2(direction[1], direction[0])])
                )
                
                # Print current position and target
                print(f"\rRobot Position: ({new_pos[0]:.3f}, {new_pos[1]:.3f}) -> "
                    f"Target: ({self.target_pos[0]:.3f}, {self.target_pos[1]:.3f})", end='')
            else:
                self.moving = False
                print("\nTarget reached!")

    def create_target_marker(self, position):
        """Create or update visual marker for target"""
        if self.target_marker is not None:
            p.removeBody(self.target_marker)
        
        visual_shape_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.003,  # Small marker
            rgbaColor=[1, 0, 0, 1]
        )
        self.target_marker = p.createMultiBody(
            baseVisualShapeIndex=visual_shape_id,
            basePosition=position
        )
        
    def check_button(self):
        """Check if next point button is pressed"""
        button_value = p.readUserDebugParameter(self.next_point_button)
        if button_value != self.last_button_value:
            self.last_button_value = button_value
            self.move_to_next_point()

    def run_simulation(self):
        """Run the simulation"""
        print("\nSimulation started - Click button to move to next calibrated point")
        
        while True:
            # Check button press
            self.check_button()
            
            # Update robot position
            self.update()
            
            # Step simulation
            p.stepSimulation()
            time.sleep(1./240.)

def main():
    # Initialize calibrator
    calibrator = CameraCalibrator()
    
    # Calibrate camera
    if calibrator.calibrate_camera('/home/hager/robotics/calibration_images/*.jpeg'):
        # Analyze test image
        try:
            test_image_path = '/home/hager/robotics/test_images/chess9.jpeg'
            undistorted_img = calibrator.analyze_test_image(test_image_path)
            
            if undistorted_img is not None:
                # Run PyBullet simulation
                simulator = PyBulletSimulator(
                    calibrator.camera_matrix,
                    calibrator.dist_coeffs,
                    calibrator,
                    cv2.imread(test_image_path)
                )
                simulator.run_simulation()
        except Exception as e:
            print(f"Error during simulation: {e}")
            raise e
    else:
        print("Camera calibration failed!")

if __name__ == "__main__":
    main()