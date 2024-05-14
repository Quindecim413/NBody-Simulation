#pragma once
#include <vector_types.h>
#define EXTERN_DLL_EXPORT extern "C" __declspec(dllexport)

struct Particle {
	float3 pos;
	float3 vel;
	float weight;
} ;

struct SimulationData {
	Particle* particleData;
	int nbodies;
};

EXTERN_DLL_EXPORT
int updateSimulationCuda(SimulationData* data, float timeStep);
EXTERN_DLL_EXPORT 
int updateSimulationC(SimulationData* data, float timeStep);