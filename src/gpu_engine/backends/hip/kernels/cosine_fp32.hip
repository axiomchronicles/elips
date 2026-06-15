// HIP distance kernel stubs — compile with hipcc
// Mirrors CUDA kernels exactly using HIP portability macros

#ifdef ELIPS_HIP_ENABLED
#error "HIP kernels must be compiled with hipcc. See hip/CMakeLists.txt."
#endif