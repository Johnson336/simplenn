
#clang -Wall -Werror -O3 -o twice twice.c -lm
#clang -Wall -Werror -O3 -o nn nn.c -lm
#clang -std=c99 -Wall -Werror -O3 -o adder adder.c -I/usr/local/include -L/usr/local/lib -lm -lraylib -framework Cocoa -framework IOKit -framework OpenGL

# Determine platform_os
UNAMEOS = $(shell uname)
ifeq ($(UNAMEOS),Linux)
  PLATFORM_OS = LINUX
endif
ifeq ($(UNAMEOS),Darwin)
  PLATFORM_OS = OSX
endif

CXX := clang
CXX_FLAGS := -O3 -std=c99 -lncurses -I/usr/local/include -L/usr/local/lib
LIBRARIES := -lm -lraylib

ifeq ($(PLATFORM_OS),OSX)
  LIBRARIES += -framework Cocoa -framework IOKit -framework OpenGL
endif
EXECUTABLE := adder

all: $(EXECUTABLE)

run: clean all
	clear
	./$(EXECUTABLE)

$(EXECUTABLE): adder.c
	$(CXX) $(CXX_FLAGS) $^ -o $@ $(LIBRARIES)

clean:
	-rm -r *.o $(EXECUTABLE)

#clang -std=c99 -Wall -Werror -O3 -o visual nn_visual.c -I/usr/local/include -L/usr/local/lib -lm -lraylib -framework Cocoa -framework IOKit -framework OpenGL
