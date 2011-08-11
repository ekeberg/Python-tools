from net_tools import *

# s1 = Sender('localhost', 5011, "boo_in.h5")
# s1.send()

# s2 = Sender('localhost', 5011, "foo_in.h5")
# s2.send()

# s3 = Sender('localhost', 5011, "foo_in.txt")
# s3.send()

runc = RunCommandClient('localhost', 5011)
runc.start()
