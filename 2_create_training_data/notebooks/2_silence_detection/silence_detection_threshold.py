""" silence_detection_threshold.py: 
Usage:
    silence_detection_threshold.py [-i <input.csv>] [-t <threshold>] [-hv]

Options:
    -h --help                   Print this screen and exit
    -v --version                Print the version of silence_detection_threshold.py
    -i --input <input.csv>      The input silence.csv [default: silence.csv]
    -t --threshold <threshold>  The silence threshold [default: 0.05]
"""

from docopt import docopt
import pandas as pd
import pandas as np
from io import StringIO

args = docopt(__doc__, version="0.0.1")

try:
    threshold = float(args["--threshold"])
except ValueError:
    exit(f"The threshold `{args['--threshold']}` can't be converted to a float")

try:
    df = pd.read_csv(args["--input"])
except FileNotFoundError:
    exit(f"The input file `{args['--input']}` is not a file?")

df["value"] = df["value"].apply(lambda v: 1 if v < threshold else 0)

output = StringIO()
df.to_csv(output, index=None)
output.seek(0)
print(output.read())