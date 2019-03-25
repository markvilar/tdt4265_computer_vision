# Makefile for C++ project
#
# Directory structure:
# 	- src
# 	- build
# 	- devel
#
# Usage:
# 	- make
# 	- make clean

# Compilers
CPP = g++

# Linker
LINKERFLAG = -lm

# Phonies
.PHONY = all clean

# Directories
SRC_DIR := src
BUILD_DIR := build
DEVEL := devel

# Files
SRCS := $(wildcard *.cpp)
BINS := $(SRCS:%.cpp=%)

all: ${BINS}

%: %.o
	@echo "Checking..."
	${CPP} -c $<


clean:
	@echo "Cleaning up..."
	rm -rvf *.o ${BINS}
