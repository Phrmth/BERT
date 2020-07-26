 # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys

def main():
    # print command line arguments
#    for arg in sys.argv[1:]:
#        print(arg)
    if len(sys.argv) <= 1:
        print("no args passed")
    if len(sys.argv) > 1:
        print(f'passsed args{sys.argv[1:]} here')

if __name__ == "__main__":
    main()