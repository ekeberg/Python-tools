import socket
import sys
import os
import threading
import subprocess

MSG_LEN = 1024

class RunCommandServer:
    def __init__(self, portn, command, args):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind(('', portn))
        self.command = command
        self.args = args
    def start(self):
        while(1):
            self.s.listen(1)
            clientsocket, addr = self.s.accept()
            print 'client is at ', addr
            self.ClientThread(clientsocket, self.command, self.args).start()

    class ClientThread(threading.Thread):
        def __init__(self, socket, command, args):
            threading.Thread.__init__(self)
            self.socket = socket
            self.command = command
            self.args = args
        def run(self):
            flo = self.socket.makefile('w',0)
            comm = pexpect.spawn(self.command, timeout=10000)
            while True:
                l = comm.readline()
                self.socket.send(l)
            flo.close()
            self.socket.close()

class RunCommandClient:
    def __init__(self, host, portn):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = host
        self.port = portn
    def start(self):
        self.socket.connect((self.host, self.port))
        # flo = self.socket.makefile('r', 0)
        # while not flo.closed:
        #     l = flo.readline()
        #     if l:
        #         sys.stdout.writelines(l)
        # flo.close()
        l = True
        while l:
            l = self.socket.recv(1024)
            print l
        self.socket.close()

class Reciever:
    def __init__(self, portn):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind(('', portn))
    def start(self):
        while(1):
            self.s.listen(3)
            clientsocket, addr = self.s.accept()
            print 'client is at', addr
            t = self.ClientThread(clientsocket)
            t.start()

    class ClientThread(threading.Thread):
        def __init__(self, socket):
            threading.Thread.__init__(self)
            self.socket = socket
        def recv_filename(self):
            msg = ''
            while len(msg) < MSG_LEN:
                chunk = self.socket.recv(MSG_LEN-len(msg))
                msg = msg + chunk
            self.filename = msg.lstrip('0')
        def run(self):
            self.recv_filename()
            print 'receiving '+self.filename
            flo = self.socket.makefile('r', 0)
            output = file("sent_"+self.filename,'wp')
            output.writelines(flo.readlines())
            output.close()
            self.socket.close()
            print 'wrote ', self.filename
            

class Sender:
    def __init__(self, host, port, filename):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.host = host
        self.port = port
        self.socket.connect((self.host, self.port))
        self.filename = filename
    def send_filename(self):
        self.socket.send(self.filename.zfill(MSG_LEN))
    def send(self):
        self.send_filename()
        flo = self.socket.makefile('w', 0)
        ppn = file(self.filename,'r')
        flo.writelines(ppn.readlines())
        flo.close()
        self.socket.close()
        print 'done sending ', self.filename
