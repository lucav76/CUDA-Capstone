IDIR=./src/
COMPILER=nvcc
LIBRARIES += -lcudart -lcuda
LIBRARIES += -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_videoio
LIBRARIES += -lnppisu_static -lnppif_static -lnppc_static -lculibos
LIBRARIES += -lnpps -lnppc -lnppicc -lnppidei -lnppif -lnppig -lnppim -lnppist -lnppisu -lnppitc -lcudart

COMPILER_FLAGS=-I$(IDIR) -I/usr/include/opencv4 -I/usr/local/cuda/include -I/usr/local/cuda/lib64 -I/home/coder/lib/cub/ -I/home/coder/lib/cuda-samples/Common $(LIBRARIES) --std c++14
CXXFLAGS = `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`

#INCLUDES += -I../Common/UtilNPP


.PHONY: clean build run

build: src/*.cu src/*.h
	$(COMPILER) $(COMPILER_FLAGS) src/*.cu -o luca

clean:
	rm -f luca

run:
	./luca $(ARGS)

all: clean build run