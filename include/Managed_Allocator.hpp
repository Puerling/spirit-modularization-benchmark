#pragma once

#ifdef SPIRIT_USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#include <sstream>
#include <iomanip>

static void CudaHandleError( cudaError_t err, const char * file, int line, const char * function )
{
    if( err != cudaSuccess )
    {
        std::ostringstream oss;
        oss << file << ":" << line << " in function" << std::quoted(function) << ":\n " << cudaGetErrorString(err);
        throw std::runtime_error( oss.str() );
    }
}

#define CU_HANDLE_ERROR( err ) ( CudaHandleError( err, __FILE__, __LINE__, __func__ ) )

#define CU_CHECK_ERROR() ( CudaHandleError( cudaGetLastError(), __FILE__, __LINE__, __func__ ) )

#define CU_CHECK_AND_SYNC()                                                                                            \
    CU_CHECK_ERROR();                                                                                                  \
    CU_HANDLE_ERROR( cudaDeviceSynchronize() )

template<class T>
class managed_allocator
{
public:
    typedef T value_type;

    managed_allocator() = default;

    // constexpr managed_allocator( const managed_allocator & a ) noexcept() {}
    template<class U>
    constexpr managed_allocator( const managed_allocator<U> & a ) noexcept {}

    [[nodiscard]] T * allocate( size_t n ) noexcept
    {
        T * result;

        CU_HANDLE_ERROR( cudaMallocManaged( &result, n * sizeof( value_type ) ) );

        return result;
    }

    void deallocate( T * ptr, size_t ) noexcept
    {
        CU_HANDLE_ERROR( cudaFree( ptr ) );
    }

};

template<class T, class U>
bool operator==(const managed_allocator <T>&, const managed_allocator <U>&) { return true; }

template<class T, class U>
bool operator!=(const managed_allocator <T>&, const managed_allocator <U>&) { return false; }

#endif
