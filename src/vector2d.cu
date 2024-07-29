#include <cuda_runtime.h>
#include <cmath>

class Vector2D {
public:
    float x;
    float y;

    __host__ __device__ Vector2D() : x(0.0f), y(0.0f) {}
    __host__ __device__ Vector2D(float x, float y) : x(x), y(y) {}

    __host__ __device__ Vector2D add(const Vector2D& v) const {
        return Vector2D(x + v.x, y + v.y);
    }

    __host__ __device__ Vector2D subtract(const Vector2D& v) const {
        return Vector2D(x - v.x, y - v.y);
    }

    __host__ __device__ Vector2D multiply(float scalar) const {
        return Vector2D(x * scalar, y * scalar);
    }

    __host__ __device__ float magnitude() const {
        return sqrtf(x * x + y * y);
    }

    __host__ __device__ Vector2D normalize() const {
        float mag = magnitude();
        if (mag > 0) {
            return Vector2D(x / mag, y / mag);
        }
        return *this;
    }

    __host__ __device__ Vector2D rotate(float angle) const {
        float cos_angle = cosf(angle);
        float sin_angle = sinf(angle);
        return Vector2D(
            x * cos_angle - y * sin_angle,
            x * sin_angle + y * cos_angle
        );
    }
};

__global__ void vectorOperations(Vector2D* vectors, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Vector2D v = vectors[idx];
        
        v = v.add(Vector2D(1.0f, 1.0f));
        v = v.multiply(2.0f);
        v = v.normalize();
        v = v.rotate(0.5f);

        vectors[idx] = v;
    }
}

void launchVectorOperations(Vector2D* h_vectors, int n) {
    Vector2D* d_vectors;
    cudaMalloc(&d_vectors, n * sizeof(Vector2D));
    cudaMemcpy(d_vectors, h_vectors, n * sizeof(Vector2D), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorOperations<<<blocksPerGrid, threadsPerBlock>>>(d_vectors, n);

    cudaMemcpy(h_vectors, d_vectors, n * sizeof(Vector2D), cudaMemcpyDeviceToHost);
    cudaFree(d_vectors);
}