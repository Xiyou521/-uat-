
import sys

if sys.version_info[0] == 2:
	from sacred.cpuinfo import *
else:
	from sacred.cpuinfo.cpuinfo import *


