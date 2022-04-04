import numpy as np

from os.path import join, dirname
import json

from projects import Project

p1 = Project("Who dat boy")
p1.spotify_grabber()
p1.download_helper()
