#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>
#include <stdio.h>
#include "simulator.h"
#include <stdlib.h>

#define softeningSquared 0.01f		// original plumer softener is 0.025. here the value is square of it.
#define damping 1.0f				// 0.999f
#define G 1
//6.67418478E-11


//EXTERN_DLL_EXPORT
//__host__ __device__ Particle updateSingleParticle(int particleInd, Particle* pdata, float step, int nbodies) {
//	Particle p = pdata[particleInd];
//
//	// update gravity (accumulation): naive big loop
//	float3 acc = { 0.0f, 0.0f, 0.0f };
//	float distSqr, distCube, s;
//
//	Particle r;
//
//	for (int i = 0; i < nbodies; i++)
//	{
//		r = pdata[i];
//
//		r.pos.x -= p.pos.x;
//		r.pos.y -= p.pos.y;
//		r.pos.z -= p.pos.z;
//
//		distSqr = r.pos.x * r.pos.x + r.pos.y * r.pos.y + r.pos.z * r.pos.z;
//		distSqr += softeningSquared;
//
//		float dist = sqrtf(distSqr);
//		distCube = dist * dist * dist + 0.001;
//
//		s = r.weight / distCube;
//
//		acc.x += r.pos.x * s;
//		acc.y += r.pos.y * s;
//		acc.z += r.pos.z * s;
//	}
//
//	// update velocity with above acc
//	p.vel.x += acc.x * step;
//	p.vel.y += acc.y * step;
//	p.vel.z += acc.z * step;
//
//	p.vel.x *= damping;
//	p.vel.y *= damping;
//	p.vel.z *= damping;
//
//	// update position
//	p.pos.x += p.vel.x * step;
//	p.pos.y += p.vel.y * step;
//	p.pos.z += p.vel.z * step;
//
//	return p;
//}

EXTERN_DLL_EXPORT
__host__ __device__ Particle updateSingleParticle(int particleInd, Particle* pdata, float step, int nbodies) {
	Particle p = pdata[particleInd];

	// update gravity (accumulation): naive big loop
	float3 acc = { 0.0f, 0.0f, 0.0f };
	float distSqr, distCube, s;

	Particle r;
	double dx, dy, dz;

	for (int i = 0; i < nbodies; i++)
	{
		if (i == particleInd)
			continue;
		r = pdata[i];

		dx = p.pos.x - r.pos.x;
		dy = p.pos.y - r.pos.y;
		dz = p.pos.z - r.pos.z;

		distSqr = (dx * dx + dy * dy + dz * dz + softeningSquared);

		float dist = sqrtf(distSqr);
		float magi = (G * r.weight) / (dist * dist * dist);
		acc.x -= magi * dx;
		acc.y -= magi * dy;
		acc.z -= magi * dz;
	}

	// update velocity with above acc
	p.vel.x += acc.x * step;
	p.vel.y += acc.y * step;
	p.vel.z += acc.z * step;

	/*p.vel.x *= damping;
	p.vel.y *= damping;
	p.vel.z *= damping;*/

	// update position
	p.pos.x += p.vel.x * step;
	p.pos.y += p.vel.y * step;
	p.pos.z += p.vel.z * step;

	return p;
}


__global__ void galaxyKernel(Particle* pdata, float step, int nbodies)
{
	// index for vertex (pos)
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x >= nbodies) {
		return;
	}
	
	Particle p = updateSingleParticle(x, pdata, step, nbodies);
	
	// thread synch
	__syncthreads();

	// update global memory with update value (position, velocity)
	pdata[x] = p;
}

//__global__ void galaxyKernel(float4* pdata, float step, int nbodies)
//{
//	// index for vertex (pos)
//	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
//	unsigned int y = blockDim.x * gridDim.x + x;
//
//	if (x >= nbodies || y >= nbodies) {
//		return;
//	}
//	auto el = pdata[1];
//
//	float4 p = pdata[x];
//	float4 v = pdata[y];
//
//	// update gravity (accumulation): naive big loop
//	float3 acc = { 0.0f, 0.0f, 0.0f };
//	float distSqr, distCube, s;
//
//	unsigned int ni = 0;
//
//	float4 r;
//
//	for (int i = 0; i < nbodies; i++)
//	{
//		r = pdata[i];
//
//		r.x -= p.x;
//		r.y -= p.y;
//		r.z -= p.z;
//
//		distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
//		distSqr += softeningSquared;
//
//		float dist = sqrtf(distSqr);
//		distCube = dist * dist * dist;
//
//		s = r.w / distCube;
//
//		acc.x += r.x * s;
//		acc.y += r.y * s;
//		acc.z += r.z * s;
//	}
//
//	// update velocity with above acc
//	v.x += acc.x * step;
//	v.y += acc.y * step;
//	v.z += acc.z * step;
//
//	// update position
//	p.x += v.x * step;
//	p.y += v.y * step;
//	p.z += v.z * step;
//
//	// thread synch
//	__syncthreads();
//
//	// update global memory with update value (position, velocity)
//	pdata[x] = p;
//	pdata[y] = v;
//}

int updateSimulationCuda(SimulationData* data, float timeStep) {
	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}


	Particle* deviceParticleData;

	cudaStatus = cudaMalloc((void**)&deviceParticleData, data->nbodies * sizeof(Particle));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaMemcpy(deviceParticleData, data->particleData, data->nbodies * sizeof(Particle), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcopyHostToDevice failed!");
		goto Error;
	}
	
	galaxyKernel <<<256, 256 >>> (deviceParticleData, timeStep, data->nbodies);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "galaxyKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(data->particleData, deviceParticleData, data->nbodies * sizeof(Particle), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(deviceParticleData);
	return cudaStatus != cudaSuccess? 1: 0;
}

int updateSimulationC(SimulationData* data, float timeStep) {
	Particle* particles = (Particle*)malloc(data->nbodies * sizeof(Particle));
	if (!particles)
		return 1;
	for (int i = 0; i < data->nbodies; i++) {
		particles[i] = updateSingleParticle(i, data->particleData, timeStep, data->nbodies);
	}
	free(data->particleData);
	data->particleData = particles;
	return 0;
}