CXX=g++
CXXFLAGS=-std=c++17 -O3 -Wall -lstdc++ -lm

CUDA=nvcc
CUDAFLAGS=-std=c++17 -O3 -Wall -forward-unknown-to-host-compiler --display-error-number --diag-suppress=20012 --expt-relaxed-constexpr --expt-extended-lambda

SRC_DIR=./src
INCLUDE_DIR=./include
DATA_DIR=data

INCLUDES=-I./thirdparty -I./thirdparty/Eigen -I./include

HPP_FILES=$(patsubst %,$(INCLUDE_DIR)/%,natives.hpp benchmark.hpp Managed_Allocator.hpp)
SRC_FILES=$(wildcard $(SRC_DIR)/*.cpp)
OUT_FILES=$(patsubst $(SRC_DIR)/%.cpp,%.out,$(SRC_FILES))

ALL=$(OUT_FILES) polymorphic_cuda.out


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

compact_write_locality.out : $(SRC_DIR)/compact_write_locality.cpp $(INCLUDE_DIR)/compact_write_locality.hpp $(HPP_FILES)
	$(CXX) $(CXXFLAGS) -ltbb $(INCLUDES) -o $@ $<

%_cuda.out : $(SRC_DIR)/%.cpp $(INCLUDE_DIR)/%.hpp $(HPP_FILES)
	$(CUDA) $(CUDAFLAGS) -DSPIRIT_USE_CUDA -x cu $(INCLUDES) -o $@ $<

%.out : $(SRC_DIR)/%.cpp $(INCLUDE_DIR)/%.hpp $(HPP_FILES)
	$(CXX) $(CXXFLAGS) -fopenmp-simd $(INCLUDES) -o $@ $<

$(DATA_DIR)/%.log : %.out
	-./$< | grep "^|" > $@
