; Comprehensive LLVM IR test cases for converter

; Test 1: Basic arithmetic operations
define i32 @simple_add(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b
  ret i32 %sum
}

; Test 2: Complex arithmetic
define i32 @complex_arithmetic(i32 %x, i32 %y) {
entry:
  %add = add i32 %x, %y
  %sub = sub i32 %add, %x
  %mul = mul i32 %sub, %y
  %div = sdiv i32 %mul, %x
  ret i32 %div
}

; Test 3: Floating-point operations
define float @float_math(float %a, float %b) {
entry:
  %add = fadd float %a, %b
  %mul = fmul float %add, %a
  %div = fdiv float %mul, %b
  ret float %div
}

; Test 4: Bitwise operations
define i32 @bitwise_ops(i32 %a, i32 %b) {
entry:
  %and_result = and i32 %a, %b
  %or_result = or i32 %and_result, %a
  %xor_result = xor i32 %or_result, %b
  %shl_result = shl i32 %xor_result, 2
  %shr_result = ashr i32 %shl_result, 1
  ret i32 %shr_result
}

; Test 5: Memory operations
define i32 @memory_test() {
entry:
  %ptr = alloca i32, align 4
  store i32 42, i32* %ptr, align 4
  %loaded = load i32, i32* %ptr, align 4
  %incremented = add i32 %loaded, 1
  store i32 %incremented, i32* %ptr, align 4
  %final = load i32, i32* %ptr, align 4
  ret i32 %final
}

; Test 6: Comparison operations
define i1 @comparisons(i32 %x, i32 %y) {
entry:
  %eq = icmp eq i32 %x, %y
  %gt = icmp sgt i32 %x, %y
  %result = and i1 %eq, %gt
  ret i1 %result
}

; Test 7: Select operation
define i32 @conditional_select(i32 %x, i32 %y) {
entry:
  %cmp = icmp sgt i32 %x, %y
  %result = select i1 %cmp, i32 %x, i32 %y
  ret i32 %result
}

; Test 8: Vector operations
define <4 x i32> @vector_arithmetic(<4 x i32> %a, <4 x i32> %b) {
entry:
  %sum = add <4 x i32> %a, %b
  %diff = sub <4 x i32> %sum, %a
  ret <4 x i32> %diff
}

; Test 9: Array operations
define i32 @array_access() {
entry:
  %arr = alloca [4 x i32], align 16
  %ptr0 = getelementptr inbounds [4 x i32], [4 x i32]* %arr, i64 0, i64 0
  store i32 10, i32* %ptr0, align 4
  %ptr1 = getelementptr inbounds [4 x i32], [4 x i32]* %arr, i64 0, i64 1
  store i32 20, i32* %ptr1, align 4
  %val0 = load i32, i32* %ptr0, align 4
  %val1 = load i32, i32* %ptr1, align 4
  %sum = add i32 %val0, %val1
  ret i32 %sum
}

; Test 10: Type conversions
define i32 @type_conversions(i8 %small, i64 %large, float %fp) {
entry:
  %ext = sext i8 %small to i32
  %trunc = trunc i64 %large to i32
  %fp_to_int = fptosi float %fp to i32
  %sum = add i32 %ext, %trunc
  %result = add i32 %sum, %fp_to_int
  ret i32 %result
}

; Test 11: Function calls
declare i32 @external_func(i32)

define i32 @function_call(i32 %x) {
entry:
  %doubled = mul i32 %x, 2
  %result = call i32 @external_func(i32 %doubled)
  ret i32 %result
}

; Test 12: Loop-like control flow
define i32 @loop_like(i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop_body, label %end

loop_body:
  %dec = sub i32 %n, 1
  %cmp2 = icmp sgt i32 %dec, 0
  br i1 %cmp2, label %continue, label %end

continue:
  br label %end

end:
  %result = phi i32 [0, %entry], [%dec, %loop_body], [%n, %continue]
  ret i32 %result
}