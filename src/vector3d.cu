#include <cuda_runtime.h>
#include <cmath>

class Vector3D {
public:
    float x;
    float y;
    float z;

    __host__ __device__ Vector3D() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ Vector3D(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vector3D add(const Vector3D& v) const {
        return Vector3D(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__ Vector3D subtract(const Vector3D& v) const {
        return Vector3D(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__ Vector3D multiply(float scalar) const {
        return Vector3D(x * scalar, y * scalar, z * scalar);
    }

    __host__ __device__ float magnitude() const {
        return sqrtf(x * x + y * y + z * z);
    }

    __host__ __device__ Vector3D normalize() const {
        float mag = magnitude();
        if (mag > 0) {
            return Vector3D(x / mag, y / mag, z / mag);
        }
        return *this;
    }

    __host__ __device__ float dot(const Vector3D& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    __host__ __device__ Vector3D cross(const Vector3D& v) const {
        return Vector3D(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }

    __host__ __device__ Vector3D rotate(float angle, const Vector3D& axis) const {
        float cos_angle = cosf(angle);
        float sin_angle = sinf(angle);
        Vector3D normalized_axis = axis.normalize();

        Vector3D rotated;
        rotated.x = (cos_angle + (1 - cos_angle) * normalized_axis.x * normalized_axis.x) * x +
                    ((1 - cos_angle) * normalized_axis.x * normalized_axis.y - sin_angle * normalized_axis.z) * y +
                    ((1 - cos_angle) * normalized_axis.x * normalized_axis.z + sin_angle * normalized_axis.y) * z;

        rotated.y = ((1 - cos_angle) * normalized_axis.y * normalized_axis.x + sin_angle * normalized_axis.z) * x +
                    (cos_angle + (1 - cos_angle) * normalized_axis.y * normalized_axis.y) * y +
                    ((1 - cos_angle) * normalized_axis.y * normalized_axis.z - sin_angle * normalized_axis.x) * z;

        rotated.z = ((1 - cos_angle) * normalized_axis.z * normalized_axis.x - sin_angle * normalized_axis.y) * x +
                    ((1 - cos_angle) * normalized_axis.z * normalized_axis.y + sin_angle * normalized_axis.x) * y +
                    (cos_angle + (1 - cos_angle) * normalized_axis.z * normalized_axis.z) * z;

        return rotated;
    }
};

__global__ void vectorOperations(Vector3D* vectors, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        Vector3D v = vectors[idx];
        
        v = v.add(Vector3D(1.0f, 1.0f, 1.0f));
        v = v.multiply(2.0f);
        v = v.normalize();
        v = v.rotate(0.5f, Vector3D(0, 0, 1));

        vectors[idx] = v;
    }
}

void launchVectorOperations(Vector3D* h_vectors, int n) {
    Vector3D* d_vectors;
    cudaMalloc(&d_vectors, n * sizeof(Vector3D));
    cudaMemcpy(d_vectors, h_vectors, n * sizeof(Vector3D), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorOperations<<<blocksPerGrid, threadsPerBlock>>>(d_vectors, n);

    cudaMemcpy(h_vectors, d_vectors, n * sizeof(Vector3D), cudaMemcpyDeviceToHost);
    cudaFree(d_vectors);
}