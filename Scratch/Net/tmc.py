import socket
import sys
import os

class Sender:
    def __init__(self, host, port, filename):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((host, port))
        self.filename = filename
    def send(self):
        flo = self.s.makefile('w', 0)
        ppn = file(self.filename,'r')
        flo.writelines(ppn.readlines())
        flo.close()
        print 'done sending'

s = Sender('localhost', 1234, 'foo_in.h5')
s.send()

