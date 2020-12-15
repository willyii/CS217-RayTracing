INC_DIR 	= include
KER_DIR		= kernel
SRC_DIR		= test
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include -I$(INC_DIR) -I$(KER_DIR)
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = rayTracing
OBJ	        = test.o 

default: $(EXE)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) 

test.o: $(SRC_DIR)/test.cu
	$(NVCC) -c -o $@ $(SRC_DIR)/test.cu $(NVCC_FLAGS)

clean:
	rm -rf *.o $(EXE)
	rm -rf result/*