INC_DIR 	= include
KER_DIR		= kernel
SRC_DIR		= main
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include -I$(INC_DIR) -I$(KER_DIR)
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE1	    = scene1
OBJ1        = scene1.o 
EXE2	    = scene2
OBJ2        = scene2.o 
EXE3	    = scene3
OBJ3        = scene3.o 

all: scene1 scene2 scene3

$(EXE1): $(OBJ1)
	$(NVCC) $(OBJ1) -o $(EXE1) 

$(EXE2): $(OBJ2)
	$(NVCC) $(OBJ2) -o $(EXE2) 

$(EXE3): $(OBJ3)
	$(NVCC) $(OBJ3) -o $(EXE3)


$(OBJ1): $(SRC_DIR)/scene1.cu $(KER_DIR)/kernel.h
	$(NVCC) -c -o $@ $(SRC_DIR)/scene1.cu $(NVCC_FLAGS)

$(OBJ2): $(SRC_DIR)/scene2.cu $(KER_DIR)/kernel.h
	$(NVCC) -c -o $@ $(SRC_DIR)/scene2.cu $(NVCC_FLAGS)

$(OBJ3): $(SRC_DIR)/scene3.cu $(KER_DIR)/kernel.h
	$(NVCC) -c -o $@ $(SRC_DIR)/scene3.cu $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE1) $(EXE2) $(EXE3)
	rm -rf result/*
