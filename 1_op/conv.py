#!/usr/bin/python3
import os
import subprocess

def conv(n, c, h, w, f, r, s, ph, pw, sh, sw, dh, dw, args=""):
    cmd = "./conv -n %d -c %d -h %d -w %d -f %d -g %d -r %d -s %d -ph %d -pw %d -sh %d -sw %d -dh %d -dw %d " % (n, c, h, w, f, 1, r, s, ph, pw, sh, sw, dh, dw) + args
    result = subprocess.run(cmd, shell=True, capture_output=True)
    # print(result)
    pos = result.stdout.decode().find('best time')
    t = float(result.stdout[pos:].split()[-1])
    return t


def run_test(n, c, h, w, f, r, s, pp, ss, dd, test_id):
    print("test: ", test_id)
    t = conv(n, c, h, w, f, r, s, pp, pp, ss, ss, dd, dd, "-fma-math")
    print(t)
    t = conv(n, c, h, w, f, r, s, pp, pp, ss, ss, dd, dd, "-tensor-op-math-allow-conversion")
    print(t)


def expr_origin():
    run_test(128, 3, 224, 224, 64, 7, 7, 3, 2, 1, 0)
    #run_test(128, 64, 56, 56, 64, 3, 3, 1, 1, 1, 1)
    #run_test(128, 64, 56, 56, 128, 3, 3, 1, 2, 1, 2)
    #run_test(128, 128, 28, 28, 128, 3, 3, 1, 1, 1, 3)
    #run_test(128, 128, 28, 28, 256, 3, 3, 1, 2, 1, 4)
    #run_test(128, 256, 14, 14, 256, 3, 3, 1, 1, 1, 5)
    #run_test(128, 256, 14, 14, 512, 3, 3, 1, 2, 1, 6)
    #run_test(128, 512, 7, 7, 512, 3, 3, 1, 1, 1, 7)


if __name__ == "__main__":
    expr_origin()
