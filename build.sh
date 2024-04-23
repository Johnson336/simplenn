#!/bin/zsh

clang -Wall -Werror -O3 -o twice twice.c -lm
clang -Wall -Werror -O3 -o nn nn.c -lm
clang -std=c99 -Wall -Werror -O3 -o adder adder.c -I/usr/local/include -L/usr/local/lib -lm -lraylib -framework Cocoa -framework IOKit -framework OpenGL
clang -std=c99 -Wall -Werror -O3 -o visual nn_visual.c -I/usr/local/include -L/usr/local/lib -lm -lraylib -framework Cocoa -framework IOKit -framework OpenGL
