# Development Setup

## Building from Source

### Prerequisites
- C++17 compatible compiler (GCC 8+, Clang 7+, MSVC 2019+)
- CMake 3.13 or higher
- Git

### Quick Build
```bash
git clone https://github.com/HecreReed/llvm2tosa.git
cd llvm2tosa
g++ -o converter converter.cpp
./converter
```

### CMake Build
```bash
mkdir build && cd build
cmake ..
make
./converter
```

## Testing

Run the comprehensive test suite:
```bash
chmod +x test.sh
./test.sh
```

## Development Workflow

1. **Add new LLVM operations**: Extend the `Instruction` struct and add conversion methods
2. **Handle new TOSA operations**: Add methods to generate appropriate TOSA IR
3. **Test thoroughly**: Add test cases to `examples.ll` and expected outputs to `expected_tosa.mlir`

## Architecture Notes

The converter follows a multi-phase approach:
1. **Parsing**: Extract LLVM IR constructs
2. **Type Mapping**: Convert LLVM types to tensor types  
3. **Instruction Translation**: Map operations to TOSA equivalents
4. **Code Generation**: Emit valid TOSA IR

## Contributing

- Fork the repository
- Create a feature branch
- Add tests for new functionality
- Ensure all tests pass
- Submit a pull request

## Debugging

Enable verbose output by modifying the converter flags:
```cpp
// Add to main()
converter.setVerbose(true);
```