
config_base = {
    "name": "SurfaceNavigator",
    "world": "SimpleUnderwater",
    "package_name": "Ocean",
    "main_agent": "auv0",
    "agents":[
        {
            "agent_name": "auv0",
            "agent_type": "HoveringAUV",
            "sensors": [
                {
                    "sensor_type": "DynamicsSensor",
                    "socket": "COM"
                },
                {
                    "sensor_type": "RGBCamera",
                    "sensor_name": "RightCamera",
                    "socket": "CameraRightSocket",
                    "Hz": 5,
                    "configuration": {
                        "CaptureWidth": 512,
                        "CaptureHeight": 512
                    }
                },
                {
                    "sensor_type": "ImagingSonar",
                    "socket": "SonarSocket",
                    "Hz": 2,
                    "configuration": {
                        "RangeBins": 512,
                        "AzimuthBins": 512,
                        "RangeMin": 0.1,
                        "RangeMax": 40,
                        "InitOctreeRange": 100,
                        "Elevation": 20,
                        "Azimuth": 130,
                        "AzimuthStreaks": 0,
                        "ViewRegion": True
                        # "ScaleNoise": True,
                        # "AddSigma": 0.15,
                        # "MultSigma": 0.2,
                        # "RangeSigma": 0.1,
                        # "MultiPath": True
                    }
                }
            ],
            "control_scheme": 1, # PD Control Scheme
            "location": [2.0,3.0,-30.0],
            "rotation": [0, 0, 0]
        }
    ],
}

config_debug = {
    "name": "SurfaceNavigator",
    # "world": "BP_UnderWater_Main",
    # "world": "Deep_Blue_Water",
    # "world": "SonarLevelClear",
    # "world": "SonarLevel",
    "world": "ExampleLevel",
    "package_name": "Custom",
    # "world": "Dam",
    # "package_name": "Ocean",
    "main_agent": "auv0",
    "octree_min": 0.02,
    "octree_max": 5.0,
    "agents":[
        {
            "agent_name": "auv0",
            "agent_type": "HoveringAUV",
            # "agent_type": "Seeker",
            "sensors": [
                {
                    "sensor_type": "DynamicsSensor",
                    "socket": "Viewport"
                },
                {
                    "sensor_type": "ViewportCapture",
                    "sensor_name": "RightCamera",
                    "socket": "Viewport",
                    # "socket": "CameraRightSocket",
                    "Hz": 10,
                    "configuration": {
                        "CaptureWidth": 1280,
                        "CaptureHeight": 720,
                        # "ExposureCompensation": 10.5,
                        # "FovAngle": 100.0
                        # "FocalDistance": 1000.0
                        # "ExposureMethod": "AEM_MAX"
                    }
                },
                {
                    "sensor_type": "ImagingSonar",
                    "socket": "CameraRightSocket",
                    # "socket": "Viewport",
                    "Hz": 5,
                    "configuration": {
                        "RangeBins": 128,
                        # "RangeBins": 512,
                        "AzimuthBins": 128,
                        # "AzimuthBins": 512,
                        "RangeMin": 1,
                        "RangeMax": 10,
                        "InitOctreeRange": 20,
                        "Elevation": 20,
                        "Azimuth": 130,
                        # "AzimuthStreaks": -1,
                        # "ScaleNoise": true,
                        # "AddSigma": 0.15,
                        # "MultSigma": 0.2,
                        # "RangeSigma": 0.1,
                        "ViewRegion": True,
                        "ViewOctree": True
                        # "MultiPath": true
                    }
                }
            ],
            "control_scheme": 0, # Manual Control Scheme
            # "control_scheme": 1, # PD Control Scheme
            "location": [0,0,-10],
            # "location": [0,0,-55],
            "rotation": [0, 0, 160]
        }
    ],
}

config_final = {
    "name": "SurfaceNavigator",
    "world": "SonarLevelClear",
    "package_name": "Alex",
    "main_agent": "auv0",
    "octree_min": 0.02,
    "octree_max": 5.0,
    "agents":[
        {
            "agent_name": "auv0",
            "agent_type": "HoveringAUV",
            "sensors": [
                {
                    "sensor_type": "DynamicsSensor",
                    "socket": "Viewport"
                },
                {
                    "sensor_type": "ViewportCapture",
                    "sensor_name": "RightCamera",
                    "socket": "Viewport",
                    "Hz": 10,
                    "configuration": {
                        "CaptureWidth": 1280,
                        "CaptureHeight": 720,
                    }
                },
                {
                    "sensor_type": "DVLSensor",
                    "sensor_name": "DVL",
                    "socket": "DVLSocket",
                    "Hz": 5,
                    "configuration": {
                        "Elevation": 25,
                        # "DebugLines": True,
                        "MaxRange": 30
                    }
                },
                {
                    "sensor_type": "ImagingSonar",
                    "sensor_name": "SonarHigh",
                    "socket": "Viewport",
                    "Hz": 5,
                    "configuration": {
                        # "RangeBins": 256,
                        "RangeBins": 512,
                        # "AzimuthBins": 256,
                        "AzimuthBins": 512,
                        "RangeMin": 0.1,
                        "RangeMax": 10,
                        "InitOctreeRange": 20,
                        "Elevation": 12,
                        "Azimuth": 60,
                        "AzimuthStreaks": -1,
                        "ScaleNoise": True,
                        "AddSigma": 0.15,
                        "MultSigma": 0.2,
                        "RangeSigma": 0.1,
                        # "ViewRegion": True,
                        # "ViewOctree": True
                        # "MultiPath": true
                    }
                },
                {
                    "sensor_type": "ImagingSonar",
                    "sensor_name": "SonarLow",
                    # "socket": "CameraRightSocket",
                    "socket": "Viewport",
                    "Hz": 5,
                    "configuration": {
                        # "RangeBins": 256,
                        "RangeBins": 512,
                        # "AzimuthBins": 256,
                        "AzimuthBins": 512,
                        "RangeMin": 0.5,
                        "RangeMax": 40,
                        "InitOctreeRange": 20,
                        "Elevation": 20,
                        "Azimuth": 130,
                        "AzimuthStreaks": -1,
                        "ScaleNoise": True,
                        "AddSigma": 0.15,
                        "MultSigma": 0.2,
                        "RangeSigma": 0.1,
                        # "ViewRegion": True,
                        # "ViewOctree": True
                        # "MultiPath": true
                    }
                },
            ],
            "control_scheme": 0, # Manual Control Scheme
            "location": [-1.0,-8.0,-9],
            "rotation": [0, 0, 90]
        }
    ],
}

config_wo_sonar = {
    "name": "SurfaceNavigator",
    "world": "SonarLevelClear",
    "package_name": "Custom",
    "main_agent": "auv0",
    "octree_min": 0.02,
    "octree_max": 5.0,
    "agents":[
        {
            "agent_name": "auv0",
            # "agent_type": "HoveringAUV",
            "agent_type": "Seeker",
            "sensors": [
                {
                    "sensor_type": "DynamicsSensor",
                    "socket": "Viewport"
                },
                {
                    "sensor_type": "ViewportCapture",
                    "sensor_name": "RightCamera",
                    "socket": "Viewport",
                    # "socket": "CameraRightSocket",
                    "Hz": 10,
                    "configuration": {
                        "CaptureWidth": 1280,
                        "CaptureHeight": 720,
                    }
                },
                {
                    "sensor_type": "DVLSensor",
                    "sensor_name": "DVL",
                    "socket": "DVLSocket",
                    # "socket": "CameraRightSocket",
                    "Hz": 5,
                    "configuration": {
                        "Elevation": 25,
                        # "DebugLines": True,
                        "MaxRange": 30
                    }
                },
            ],
            "control_scheme": 0, # Manual Control Scheme
            "location": [-1.0,-8.0,-9],
            "rotation": [0, 0, 90]
        }
    ],
}
