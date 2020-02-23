CC = nvcc

sssp: *.cu *.cpp *.c
	$(CC) -std=c++11 parse_graph.cpp utils.c entry_point.cu -O3 -o sssp
#parse_graph.cpp impl1.cu impl2.cu opt.cu 

clean:
	rm -f *.o sssp
