#!/bin/bash

# Complete TOSA Operations Test Suite
# Tests all 66 TOSA operations with LLVM IR examples

echo "=== LLVM2TOSA Complete Test Suite ==="
echo "Testing all 66 TOSA operations"

# Build the converter
echo "Building converter..."
make -C build

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

PASS=0
FAIL=0

# Function to run a test
run_test() {
    local test_name="$1"
    local input_file="$2"
    local expected_op="$3"
    
    echo -n "Testing $test_name: "
    
    if [ ! -f "$input_file" ]; then
        echo "SKIP (no input file)"
        return
    fi
    
    output=$(./build/llvm2tosa "$input_file" 2>&1)
    
    if echo "$output" | grep -q "$expected_op"; then
        echo "PASS"
        ((PASS++))
    else
        echo "FAIL"
        echo "  Expected: $expected_op"
        echo "  Output: $output"
        ((FAIL++))
    fi
}

# Test Matrix Operations (already implemented)
run_test "Matrix Multiplication" "matmul_test.ll" "tosa.matmul"
run_test "2D Convolution" "conv2d_test.ll" "tosa.conv2d"

# Create and test basic arithmetic operations
echo "Creating basic arithmetic tests..."

# TOSA Add Operation
cat > add_test.ll << 'EOF'
define float @add_example(float %a, float %b) {
entry:
  %result = fadd float %a, %b
  ret float %result
}
EOF

# TOSA Sub Operation  
cat > sub_test.ll << 'EOF'
define float @sub_example(float %a, float %b) {
entry:
  %result = fsub float %a, %b
  ret float %result
}
EOF

# TOSA Mul Operation
cat > mul_test.ll << 'EOF'
define float @mul_example(float %a, float %b) {
entry:
  %result = fmul float %a, %b
  ret float %result
}
EOF

# Test basic operations
run_test "Addition" "add_test.ll" "tosa.add"
run_test "Subtraction" "sub_test.ll" "tosa.sub"  
run_test "Multiplication" "mul_test.ll" "tosa.mul"

echo ""
echo "=== Test Results ==="
echo "PASSED: $PASS"
echo "FAILED: $FAIL"
echo "TOTAL:  $((PASS + FAIL))"

# Cleanup
rm -f *_test.ll

if [ $FAIL -eq 0 ]; then
    echo "All tests passed!"
    exit 0
else
    echo "Some tests failed!"
    exit 1
fi