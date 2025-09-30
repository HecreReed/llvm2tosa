# LLVM2TOSA

A comprehensive LLVM IR to TOSA IR converter that transforms low-level LLVM Intermediate Representation into high-level Tensor Operator Set Architecture (TOSA) IR for AI accelerator deployment.

## Overview

This converter bridges the fundamental architectural differences between:
- **LLVM IR**: Scalar/memory-based computation with explicit control flow
- **TOSA IR**: Tensor-based computation with structured control flow optimized for AI accelerators

## Features

### Complete Instruction Coverage
- **Arithmetic Operations**: Add, Sub, Mul, Div (signed/unsigned), Remainder
- **Floating-Point**: FAdd, FSub, FMul, FDiv, FRem with proper precision
- **Bitwise Operations**: And, Or, Xor, Shift operations (logical/arithmetic) 
- **Comparisons**: Integer and floating-point with all predicates
- **Type Conversions**: Truncation, extension, FP conversions, pointer casts
- **Vector Operations**: Element extraction/insertion, shuffling, broadcasting
- **Memory Operations**: Allocation, load/store, pointer arithmetic (GEP)
- **Control Flow**: Branches, loops, conditionals, function calls, PHI nodes

### Advanced Type System
- **Scalars → Tensors**: `i32` → `tensor<1xi32>`, `float` → `tensor<1xf32>`
- **Vectors → Multi-dimensional**: `<4 x i32>` → `tensor<4xi32>`
- **Arrays → Tensor Ops**: `[8 x float]` → `tensor<8xf32>` with indexing
- **Memory Model Abstraction**: Explicit memory → immutable tensor values

### Structured Control Flow
- **CFG Analysis**: Basic block to structured control flow conversion
- **Loop Detection**: Natural loops → `tosa.while_loop`
- **Conditional Conversion**: Branches → `tosa.cond_if`

## Quick Start

### Building

```bash
git clone https://github.com/YourUsername/llvm2tosa.git
cd llvm2tosa
mkdir build && cd build
cmake ..
make -j4
```

### Usage

```bash
# Convert LLVM IR to TOSA IR
./llvm2tosa input.ll output.mlir

# Run demo
./converter_demo

# Run tests
./unit_tests
```

## Architecture

```
src/
├── LLVMToTosaConverter.cpp    # Main orchestration
├── TypeConverter.cpp          # Type system mapping  
├── MemoryModelConverter.cpp   # Memory abstraction
├── ControlFlowConverter.cpp   # Control flow restructuring
└── InstructionConverter.cpp   # Instruction mapping

include/
└── LLVMToTosaConverter.h      # Public API

examples/
├── basic_examples.ll          # LLVM IR examples
├── simple.ll                  # Simple test case
└── expected_output.mlir       # Expected TOSA output

tests/
├── unit_test.cpp              # Unit tests
└── *.sh                       # Test scripts
```

## Example Conversion

### Input (LLVM IR)
```llvm
define i32 @add(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}
```

### Output (TOSA IR)
```mlir
func.func @add(%arg0: tensor<1xi32>, %arg1: tensor<1xi32>) -> tensor<1xi32> {
  %0 = tosa.add %arg0, %arg1 : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
  return %0 : tensor<1xi32>
}
```

## Technical Details

### Memory Model Translation
- **Allocation tracking**: `alloca` → tensor initialization
- **Load/store conversion**: Memory ops → slice/concat operations  
- **Index computation**: GEP → structured tensor indexing

### Type System Bridging
- **Scalar conversion**: All scalars become rank-1 tensors
- **Shape inference**: Automatic tensor shape computation
- **Broadcasting**: Compatible shape transformations

### SSA Preservation
- **Value mapping**: LLVM values → MLIR tensor values
- **Dependency analysis**: Proper conversion ordering
- **Type consistency**: Semantic correctness throughout

## Implementation Statistics

- **2,080+ lines** of production C++ code
- **356 lines** - Complete API definitions
- **535 lines** - All LLVM instruction mappings
- **326 lines** - Memory model abstraction
- **324 lines** - Control flow conversion  
- **304 lines** - Type system with caching
- **235 lines** - Main converter orchestration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this converter in your research, please cite:

```bibtex
@software{llvm2tosa,
  title={LLVM2TOSA: Complete LLVM IR to TOSA IR Converter},
  author={[Your Name]},
  year={2024},
  url={https://github.com/YourUsername/llvm2tosa}
}
```