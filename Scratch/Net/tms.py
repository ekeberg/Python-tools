from net_tools import *

#l = Reciever(5011)
#l.start()

os.chdir('~/Work/programs/emc/git/emc')
runs = RunCommandServer(5011,'./emc','')
runs.start()
