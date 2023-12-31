#!/usr/bin/python

import sys


def comparator(tame, wild):
    A, Ak, B, Bk = [], [], [], []
    with open(tame, 'r') as f:
        for line in f:
            if len(line) == 130:
                L = line.split()
                a = int(L[0], 16)
                b = int(L[1], 16)
                A.append(a)
                Ak.append(b)
    with open(wild, 'r') as f:
        for line in f:
            if len(line) == 130:
                L = line.split()
                a = int(L[0], 16)
                b = int(L[1], 16)
                B.append(a)
                Bk.append(b)
    result = list(set(A) & set(B))
    if len(result) > 0:
        sol_kt = A.index(result[0])
        sol_kw = B.index(result[0])
        d = Ak[sol_kt] - Bk[sol_kw]
        print('\n' + '\n' + 'SOLVED: ' + hex(d) + '\n')
        print('  Tips: 1NULY7DhzuNvSDtPkFzNo6oRTZQWBqXNE9 ' + '\n')
        file = open("Result.txt", 'a')
        file.write("----------------------------------------\n")
        file.write(hex(Ak[sol_kt] - Bk[sol_kw]) + "\n")
        file.write("----------------------------------------\n")
        file.write("Tips: 1NULY7DhzuNvSDtPkFzNo6oRTZQWBqXNE9\n")
        file.close()
        sys.exit(0)
    else:
        sys.exit(1)


def main():

    name1 = sys.argv[1]
    name2 = sys.argv[2]
    s_tame = str(name1)
    s_wild = str(name2)
    #print('\n' + 'Compare:' + s_tame + s_wild + '\n')
    comparator(s_tame, s_wild)


if __name__ == "__main__":
    main()
