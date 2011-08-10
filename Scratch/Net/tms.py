import socket
import sys
import os

class Listener:
    def __init__(self, portn, filename):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind(('', portn))
        self.filename = filename
    def start(self):
        while(1):
            self.s.listen(1)
            conn, addr = self.s.accept()
            print 'client is at', addr
            flo = conn.makefile('r', 0)
            output = file(self.filename,'wp')
            output.writelines(flo.readlines())
            output.close()
            conn.close()
            print 'wrote ', self.filename

l = Listener(1234,'foo_out.h5')
l.start()        
