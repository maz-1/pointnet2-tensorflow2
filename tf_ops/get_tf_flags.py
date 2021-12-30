#!/usr/bin/env python3

import os
import sys
import tensorflow as tf

cflags_list=tf.sysconfig.get_compile_flags()
lflags_list=tf.sysconfig.get_link_flags()

if os.name == 'nt':
    for flags_list in [cflags_list, lflags_list]:
        for i, s in enumerate(flags_list):
            if s.startswith('-I'):
                s=s.replace("-I","")
                s='/I"'+s+'"'
            if s.startswith('-D'):
                s=s.replace("-D","")
                s='/D'+s
            flags_list[i] = s
        
cflags=" ".join(cflags_list)
lflags=" ".join(lflags_list)

if len(sys.argv) == 2:
    if sys.argv[1] == 'cflags':
        print(cflags)
    elif sys.argv[1] == 'lflags':
        print(lflags)