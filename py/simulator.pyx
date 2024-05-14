from libc.stdlib cimport malloc, free
from numpy import empty_like
cimport numpy as np
import numpy as np

cdef extern from "vector_types.h":
    cdef struct float3:
        float x, y, z

cdef extern from "libSimulation/simulator.h":
    cdef struct Particle:
        float3 pos
        float3 vel
        float weight
    
    cdef struct SimulationData:
        Particle* particleData
        int nbodies
    
    int updateSimulationCuda(SimulationData* data, float step)
    int updateSimulationC(SimulationData* data, float timeStep)


cdef class Simulation:
    cdef SimulationData* data

    def __cinit__(self, np.float32_t[:, :] particlesPositions not None, np.float32_t[:, :] particlesVelocities, np.float32_t[:] particlesWeights):
        assert particlesPositions.shape[0] == particlesVelocities.shape[0] == particlesWeights.shape[0]

        cdef int nbodies = particlesPositions.shape[0]

        self.data = <SimulationData*>malloc(sizeof(SimulationData))
        self.data.particleData = <Particle*> malloc(sizeof(Particle) * nbodies)
        self.data.nbodies = nbodies

        cdef:
            int i
            Particle p
            np.float32_t[:] pos
            np.float32_t[:] vel
            np.float weight

        for i in range(nbodies):
            pos = particlesPositions[i]
            vel = particlesVelocities[i]
            weight = particlesWeights[i]

            p = Particle()
            p.pos.x = pos[0]
            p.pos.y = pos[1]
            p.pos.z = pos[2]
            p.vel.x = vel[0]
            p.vel.y = vel[1]
            p.vel.z = vel[2]
            p.weight = weight
            
            self.data.particleData[i] = p


    def __dealloc__(self):
        free(self.data.particleData)
        free(self.data)

    property positions:
        def __get__(self):
            cdef:
                float[:, :] dataPos = np.empty((self.data.nbodies, 3), dtype='float32')
                int i
            for i in range(self.data.nbodies):
                dataPos[i, 0] = self.data.particleData[i].pos.x
                dataPos[i, 1] = self.data.particleData[i].pos.y
                dataPos[i, 2] = self.data.particleData[i].pos.z
            return np.array(dataPos)

    property data:
        def __get__(self):
            cdef:
                float[:, :] dataPos = np.empty((self.data.nbodies, 3), dtype='float32')
                float[:, :] dataVel = np.empty((self.data.nbodies, 3), dtype='float32')
                float[:] dataWeight = np.empty((self.data.nbodies), dtype='float32')
                int i
            
            for i in range(self.data.nbodies):
                dataPos[i, 0] = self.data.particleData[i].pos.x
                dataPos[i, 1] = self.data.particleData[i].pos.y
                dataPos[i, 2] = self.data.particleData[i].pos.z
                dataVel[i, 0] = self.data.particleData[i].vel.x
                dataVel[i, 1] = self.data.particleData[i].vel.y
                dataVel[i, 2] = self.data.particleData[i].vel.z
                dataWeight[i] = self.data.particleData[i].weight

            return np.array(dataPos), np.array(dataVel), np.array(dataWeight)

    def update(self, timestep=0.001, type='C'):
        if type == 'C':
            if updateSimulationC(self.data, timestep):
                raise Exception("Failed to update with C")
        elif type == 'CUDA':
            if updateSimulationCuda(self.data, timestep):
                raise Exception('Failed to update with CUDA')
        else:
            raise Exception(f'Undefined type "{type}". Use "C" or "CUDA" instead.')
            