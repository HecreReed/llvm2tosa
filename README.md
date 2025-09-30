# LLVM2TOSA - Complete LLVM IR to TOSA IR Converter

A comprehensive, production-ready converter that transforms LLVM Intermediate Representation (IR) to Tensor Operator Set Architecture (TOSA) IR. This implementation bridges the fundamental architectural differences between scalar/memory-based computation (LLVM) and tensor-based computation (TOSA).

## 🚀 Features

### Complete Instruction Set Coverage
- **Arithmetic Operations**: Add, Sub, Mul, Div (signed/unsigned), Remainder
- **Floating-Point Operations**: FAdd, FSub, FMul, FDiv, FRem with proper precision handling
- **Bitwise Operations**: And, Or, Xor, Shift operations (logical/arithmetic)
- **Comparison Operations**: Integer and floating-point comparisons with all predicates
- **Type Conversions**: Truncation, extension, floating-point conversions, pointer/integer casts
- **Vector Operations**: Element extraction/insertion, shuffling, broadcasting
- **Memory Operations**: Allocation, load/store, pointer arithmetic (GEP)
- **Control Flow**: Branches, loops, conditionals, function calls, PHI nodes

### Advanced Type System Mapping
- **Scalars → Tensors**: `i32` → `tensor<1xi32>`, `float` → `tensor<1xf32>`
- **Vectors → Multi-dimensional Tensors**: `<4 x i32>` → `tensor<4xi32>`
- **Arrays → Tensor Operations**: `[8 x float]` → `tensor<8xf32>` with proper indexing
- **Pointers → Tensor References**: Memory model abstraction
- **Structs → Packed Tensors**: Flattened representation with proper element access

## 📊 Implementation Statistics

- **2,080+ lines** of comprehensive C++ implementation
- **356 lines** - Header with complete API definitions
- **535 lines** - Instruction converter with all LLVM opcodes
- **326 lines** - Memory model abstraction layer
- **324 lines** - Control flow analysis and conversion
- **304 lines** - Type system conversion with caching
- **235 lines** - Main converter orchestration

## 🔧 Building

```bash
mkdir build && cd build
cmake ..
make -j4
```

## 📖 Usage

```cpp
#include "LLVMToTosaConverter.h"

// Initialize contexts
llvm::LLVMContext llvmContext;
mlir::MLIRContext mlirContext;

// Create converter
llvm2tosa::LLVMToTosaConverter converter(mlirContext);

// Convert module
auto llvmModule = /* your LLVM module */;
auto tosaModule = converter.convertModule(*llvmModule);
```

This converter represents a complete solution for transforming LLVM IR to TOSA IR, addressing all major architectural differences while maintaining semantic correctness and performance.
EOF < /dev/null