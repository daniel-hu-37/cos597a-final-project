import shutil
import urllib.request as request
from contextlib import closing
import tarfile
import numpy as np
import os

# first we download the Sift1M dataset
with closing(
    request.urlopen("ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz")
) as r:
    with open("./data/sift.tar.gz", "wb") as f:
        shutil.copyfileobj(r, f)

# the download leaves us with a tar.gz file, we unzip it
tar = tarfile.open("./data/sift.tar.gz", "r:gz")
tar.extractall(path="./data/")
os.remove("./data/sift.tar.gz")


# now define a function to read the fvecs file format of Sift1M dataset
def read_fvecs(fp):
    a = np.fromfile(fp, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy().view("float32")


# data we will search through
xb = read_fvecs("./data/sift/sift_base.fvecs")  # 1M samples
# also get some query vectors to search with
xq = read_fvecs("./data/sift/sift_query.fvecs")
# take just one query (there are many in sift_learn.fvecs)
xq = xq[0].reshape(1, xq.shape[1])

print()
print()
print("Files downloaded and extracted.")
print()
print("Base Shape: " + str(xb.shape))
print("Query Shape: " + str(xq.shape))
print()
print()
