from net_tools import *

#l = Reciever(5011)
#l.start()

os.chdir(os.path.expanduser('~/Work/programs/emc/git/emc'))
runs = RunCommandServer(5011,'./emc','')
runs.start()
