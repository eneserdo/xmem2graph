import pyzed.sl as sl

class Camera:

    def __init__(self, get_depth=True, get_pcl=True) -> None:

        # Create a Camera object
        self.zed = sl.Camera()

        # Create a InitParameters object and set configuration parameters
        self.init_params = sl.InitParameters()
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.camera_resolution = sl.RESOLUTION.HD720

        # Open the camera
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Can't open the camera!")
            exit(1)

        # Create and set RuntimeParameters after opening the camera
        self.runtime_parameters = sl.RuntimeParameters()
        self.runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
        # Setting the depth confidence parameters
        self.runtime_parameters.confidence_threshold = 50
        self.runtime_parameters.textureness_confidence_threshold = 100

        # min_depth
        self.rgb = sl.Mat()
        if get_depth:
            self.depth = sl.Mat()
        if get_pcl:
            self.point_cloud = sl.Mat()

    def getData(self):
        try:
            while True:
                # A new image is available if grab() returns SUCCESS
                if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
                    # Retrieve left image
                    self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
                    # Retrieve depth map. Depth is aligned on the left image
                    self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
                    # Retrieve colored point cloud. Point cloud is aligned on the left image.
                    self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)

                    yield self.image.get_data(), self.point_cloud

                    # point3D = point_cloud.get_value(i,j)
                    # x = point3D[0]
                    # y = point3D[1]
                    # z = point3D[2]
                    # color = point3D[3] 
        finally:
            # Close the camera
            print("\nclosing")
            self.zed.close()
