import carla

"""
Town01  A small, simple town with a river and several bridges.
Town02	A small simple town with a mixture of residential and commercial buildings.
Town03	A larger, urban map with a roundabout and large junctions.
Town04	A small town embedded in the mountains with a special "figure of 8" infinite highway.
Town05	Squared-grid town with cross junctions and a bridge. It has multiple lanes per direction. Useful to perform lane changes.
Town06	Long many lane highways with many highway entrances and exits. It also has a Michigan left.
Town07	A rural environment with narrow roads, corn, barns and hardly any traffic lights.
Town08	Secret "unseen" town used for the Leaderboard challenge
Town09	Secret "unseen" town used for the Leaderboard challenge
Town10	A downtown urban environment with skyscrapers, residential buildings and an ocean promenade.
Town11	A Large Map that is undecorated. Serves as a proof of concept for the Large Maps feature.
Town12	A Large Map with numerous different regions, including high-rise, residential and rural environments.
"""
maps = [
  "Town01",
  "Town02",
  "Town03",
  "Town04",
  "Town05",
  "Town06",
  "Town07",
  "Town08",
  "Town09",
  "Town10",
  "Town11",
  "Town12"
  ]

weather = {
  "ClearNoon": carla.WeatherParameters.ClearNoon,
  "CloudyNoon": carla.WeatherParameters.CloudyNoon,
  "WetNoon": carla.WeatherParameters.WetNoon,
  "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
  "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
  "HardRainNoon": carla.WeatherParameters.HardRainNoon,
  "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
  "ClearSunset": carla.WeatherParameters.ClearSunset,
  "CloudySunset": carla.WeatherParameters.CloudySunset,
  "WetSunset": carla.WeatherParameters.WetSunset,
  "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
  "MidRainSunset": carla.WeatherParameters.MidRainSunset,
  "HardRainSunset": carla.WeatherParameters.HardRainSunset,
  "SoftRainSunset": carla.WeatherParameters.SoftRainSunset,
  "ClearNight": carla.WeatherParameters(cloudiness=0.0,
                                   precipitation=0.0,
                                   sun_altitude_angle=-20.0),
  "CloudyNight": carla.WeatherParameters(cloudiness=80.0,
                                   precipitation=0.0,
                                   sun_altitude_angle=-20.0),
  "HardRainNight": carla.WeatherParameters(cloudiness=80.0,
                                   precipitation=50.0,
                                   sun_altitude_angle=-20.0),
  "SoftRainNight": carla.WeatherParameters(cloudiness=80.0,
                                   precipitation=25.0,
                                   sun_altitude_angle=-20.0)

  }