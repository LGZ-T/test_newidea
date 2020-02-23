#include <vector>
#include <iostream>
#include <fstream>

#include "utils.h"
#include "cuda_error_check.cuh"
#include "initial_graph.hpp"
#include "parse_graph.hpp"

__global__ void pulling_kernel(unsigned int *trace, GraphEdge_t* edges, unsigned int nEdges, unsigned int* d_curr, unsigned int* d_prev, int* is_changed) {

	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int threadCount = blockDim.x * gridDim.x;
	int lane = threadId & (32 - 1);
	int countWarps = threadCount % 32 ? threadCount / 32 + 1 : threadCount / 32;
	int nEdgesPerWarp = nEdges % countWarps == 0 ? nEdges / countWarps : nEdges / countWarps + 1;
	int warpId = threadId / 32;
	int beg = nEdgesPerWarp*warpId;
	int end = beg + nEdgesPerWarp - 1;

	unsigned int src = 0;
	unsigned int dest = 0;
	unsigned int weight = 0;
	unsigned int tmp = 0;
	GraphEdge_t *edge;
	unsigned int *temp;
	unsigned int j=0;
	for (int i = beg + lane; i<= end && i< nEdges; i += 32) {
		edge = edges + i;
		src = edge->src;
		temp = &(edge->src);
		trace[j] = (unsigned int)temp;
		dest = edge->dest;
		temp = &(edge->dest);
		j++;
		trace[j] = (unsigned int)temp;
		weight = edge->weight;
		temp = &(edge->weight);
		j++;
		trace[j] = (unsigned int)temp;
		j++;

		tmp = d_prev[src] + weight;
		trace[j] = src;
		j++;
		trace[j] = dest;
		j++;
		if (tmp < d_prev[dest]) {
			trace[j] = dest;
			j++;
			atomicMin(&d_curr[dest], tmp);
			*is_changed = 1;
		}else{
			trace[j] = 1111111111;
			j++;
		}
	} 
}

__device__ unsigned int min_dist(unsigned int val1, unsigned int val2) {
	if (val1 < val2) {
		return val1;
	} else {
		return val2;
	}
}

__global__ void pulling_kernel_smem(GraphEdge_t* edges, unsigned int nEdges, unsigned int nVertices, unsigned int* d_curr, unsigned int* d_prev, int* is_changed) {

	__shared__ unsigned int shared_mem[2048];

	const int nEdgesPerWarp = 64;
	int threadId = blockDim.x * blockIdx.x + threadIdx.x;
	int threadCount = blockDim.x * gridDim.x;
	int lane = threadId & (32 - 1);
	int countWarps = threadCount % 32 ? threadCount / 32 + 1 : threadCount / 32;
	int nEdgesPerIter = countWarps * nEdgesPerWarp;
	int nIters = nEdges % nEdgesPerIter ? nEdges / nEdgesPerIter + 1 : nEdges / nEdgesPerIter;

	unsigned int src = 0;
	unsigned int dest = 0;
	unsigned int weight = 0;
	GraphEdge_t *edge;
	for (int iter = 0; iter < nIters; ++iter) {
		int warpId = threadId / 32 + iter*countWarps;
		int beg = warpId * nEdgesPerWarp;
		int end = beg + nEdgesPerWarp -1;

		for (int i = beg + lane; i<= end && i< nEdges; i += 32) {
			edge = edges + i;
			src = edge->src;
			dest = edge->dest;
			weight = edge->weight;

			shared_mem[threadIdx.x] = min_dist(d_prev[src] + weight, d_prev[dest]);

			__syncthreads();

			if ((lane >= 1) && (edges[i].dest == edges[i-1].dest)) {
				shared_mem[threadIdx.x] = min_dist(shared_mem[threadIdx.x], shared_mem[threadIdx.x-1]);
			}
			if (lane >= 2 && edges[i].dest == edges[i-2].dest) {
				shared_mem[threadIdx.x] = min_dist(shared_mem[threadIdx.x], shared_mem[threadIdx.x-2]);
			}
			if (lane >= 4 && edges[i].dest == edges[i-4].dest) {
				shared_mem[threadIdx.x] = min_dist(shared_mem[threadIdx.x], shared_mem[threadIdx.x-4]);
			}
			if (lane >= 8 && edges[i].dest == edges[i-8].dest) {
				shared_mem[threadIdx.x] = min_dist(shared_mem[threadIdx.x], shared_mem[threadIdx.x-8]);
			}
			if (lane >= 16 && edges[i].dest == edges[i-16].dest) {
				shared_mem[threadIdx.x] = min_dist(shared_mem[threadIdx.x], shared_mem[threadIdx.x-16]);
			}

			__syncthreads();

			if (((i < nEdges - 1) && edges[i].dest != edges[i+1].dest) || (i % 32 == 31 || i == nEdges - 1)) {
				if (d_curr[dest] > shared_mem[threadIdx.x])
					atomicMin(&d_curr[dest], shared_mem[threadIdx.x]);
			}
		}
	}

	__syncthreads();

	int nVerticesPerWarp = nVertices % countWarps == 0 ? nVertices / countWarps : nVertices / countWarps + 1;
	int warpId = threadId / 32;
	int beg = nVerticesPerWarp*warpId;
	int end = beg + nVerticesPerWarp - 1;
	for (int i = beg + lane; i<= end && i< nVertices && *is_changed == 0; i += 32) {
		if (d_prev[i] != d_curr[i]) {
			*is_changed = 1;
		}
	}
}

void puller(GraphEdge_t* edges, unsigned int nEdges, unsigned int nVertices, unsigned int* distance, int bsize, int bcount, int isIncore, int useSharedMem) {
	GraphEdge_t *d_edges;
	unsigned int* d_distances_curr;
	unsigned int* d_distances_prev;
	int *d_is_changed;
	int h_is_changed;
	int count_iterations = 0;

	cudaMalloc((void**)&d_edges, sizeof(GraphEdge_t)*nEdges);
	cudaMalloc((void**)&d_distances_curr, sizeof(unsigned int)*nVertices);
	cudaMalloc((void**)&d_distances_prev, sizeof(unsigned int)*nVertices);
	cudaMalloc((void**)&d_is_changed, sizeof(int));

	cudaMemcpy(d_edges, edges, sizeof(GraphEdge_t)*nEdges, cudaMemcpyHostToDevice);
	cudaMemcpy(d_distances_curr, distance, sizeof(unsigned int)*nVertices, cudaMemcpyHostToDevice);
	cudaMemcpy(d_distances_prev, distance, sizeof(unsigned int)*nVertices, cudaMemcpyHostToDevice);
	std::cout << "3333333" << std::endl;
	setTime();
	int threadCount = bcount*bsize;
	int countWarps = threadCount % 32 ? threadCount / 32 + 1 : threadCount / 32;
	int nEdgesPerWarp = nEdges % countWarps == 0 ? nEdges / countWarps : nEdges / countWarps + 1;
	std::ofstream outputFile("lgzoutfile",std::ios::binary | std::ios::out|std::ios::app);
	unsigned int *trace = (unsigned int *)malloc(bcount*bsize*sizeof(unsigned int)*6*(nEdgesPerWarp/32+1));
	unsigned int *trace_gpu;
	cudaMalloc((void**)&trace_gpu,bcount*bsize*sizeof(unsigned int)*6*(nEdgesPerWarp/32+1));
	std::cout<<"444444" << std::endl;
	for (int i = 0; i < nVertices-1; ++i) {
		cudaMemset(d_is_changed, 0, sizeof(int));
		if (isIncore == 1){
			pulling_kernel<<<bcount, bsize>>>(trace_gpu, d_edges, nEdges, d_distances_curr, d_distances_curr, d_is_changed);
			cudaMemcpy(&trace, trace_gpu, bcount*bsize*sizeof(unsigned int)*6*(nEdgesPerWarp/32+1), cudaMemcpyDeviceToHost);
			outputFile.write((char *)trace,bcount*bsize*sizeof(unsigned int)*6*(nEdgesPerWarp/32+1));
		}
		else if (useSharedMem == 0){
			std::cout<<"55555555"<<std::endl;
			pulling_kernel<<<bcount, bsize>>>(trace_gpu,d_edges, nEdges, d_distances_curr, d_distances_prev, d_is_changed);
			std::cout<<"66666666"<<std::endl;
			cudaMemcpy(&trace, trace_gpu, bcount*bsize*sizeof(unsigned int)*6*(nEdgesPerWarp/32+1), cudaMemcpyDeviceToHost);
			std::cout<<"77777777"<<std::endl;
			outputFile.write((char *)trace,bcount*bsize*sizeof(unsigned int)*6*(nEdgesPerWarp/32+1));
			std::cout<<"88888888"<<std::endl;
		}
		else{
			pulling_kernel_smem<<<bcount, bsize>>>(d_edges, nEdges, nVertices, d_distances_curr, d_distances_prev, d_is_changed);
		}

		cudaDeviceSynchronize();
		if (isIncore == 0)
			cudaMemcpy(d_distances_prev, d_distances_curr, sizeof(unsigned int)*nVertices, cudaMemcpyDeviceToDevice);

		count_iterations++;

		cudaMemcpy(&h_is_changed, d_is_changed, sizeof(int), cudaMemcpyDeviceToHost);
		if (h_is_changed == 0) {
			break;
		}

	}
	free(trace);
	cudaFree(trace_gpu);
	outputFile.close();
	std::cout << "Took "<<count_iterations << " iterations " << getTime() << "ms.\n";

	cudaMemcpy(distance, d_distances_curr, sizeof(unsigned int)*nVertices, cudaMemcpyDeviceToHost);

	cudaFree(d_edges);
	cudaFree(d_distances_curr);
	cudaFree(d_distances_prev);
	cudaFree(d_is_changed);
}
