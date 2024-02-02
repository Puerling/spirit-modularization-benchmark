CXX=g++
CXXFLAGS=-std=c++17 -O3 -Wall -lstdc++ -lm

SRC_DIR=./src
INCLUDE_DIR=./include

INCLUDES=-I./nanobench -I./include

HPP_FILES=$(patsubst %,$(INCLUDE_DIR)/%,natives.hpp benchmark.hpp)
SRC_FILES=$(wildcard $(SRC_DIR)/*.cpp)
OUT_FILES=$(patsubst $(SRC_DIR)/%.cpp,%.out,$(SRC_FILES))

ALL=$(OUT_FILES)


.PHONY : show all run time clean
.SILENT : show all run time

all : $(ALL)

show :
	-echo "$(ALL)"

run : $(ALL)
	-for f in $(ALL); do (set -x; "./$$f"); done

clean :
	-rm -f $(OUT_FILES) gmon.out

time : $(ALL)
	-for f in $(ALL); do echo "==========" && env TIMEFMT=$$'job\t%J\nreal\t%E\nuser\t%U\nsys\t%S\nmem(peak)\t%M' zsh -c "time ./$$f > /dev/null"; done && echo "=========="

write_locality.out : $(SRC_DIR)/write_locality.cpp $(INCLUDE_DIR)/write_locality.hpp $(HPP_FILES)
	$(CXX) $(CXXFLAGS) -ltbb $(INCLUDES) -o $@ $<

%.out : $(SRC_DIR)/%.cpp $(INCLUDE_DIR)/%.hpp $(HPP_FILES)
	$(CXX) $(CXXFLAGS) -fopenmp-simd $(INCLUDES) -o $@ $<
