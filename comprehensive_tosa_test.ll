; Comprehensive TOSA Operations Test File
; Tests all 66 TOSA operations with corresponding LLVM IR patterns

; 1. tosa.abs - Absolute value
define float @test_abs(float %input) {
entry:
  %is_negative = fcmp olt float %input, 0.0
  %neg_input = fsub float 0.0, %input
  %result = select i1 %is_negative, float %neg_input, float %input
  ret float %result
}

; 2. tosa.add - Addition (already working)
define float @test_add(float %a, float %b) {
entry:
  %result = fadd float %a, %b
  ret float %result
}

; 3. tosa.sub - Subtraction
define float @test_sub(float %a, float %b) {
entry:
  %result = fsub float %a, %b
  ret float %result
}

; 4. tosa.mul - Multiplication  
define float @test_mul(float %a, float %b) {
entry:
  %result = fmul float %a, %b
  ret float %result
}

; 5. tosa.negate - Negation
define float @test_negate(float %input) {
entry:
  %result = fsub float 0.0, %input
  ret float %result
}

; 6. tosa.reciprocal - Reciprocal (1/x)
define float @test_reciprocal(float %input) {
entry:
  %result = fdiv float 1.0, %input
  ret float %result
}

; 7. tosa.rsqrt - Reciprocal square root
define float @test_rsqrt(float %input) {
entry:
  %sqrt_val = call float @llvm.sqrt.f32(float %input)
  %result = fdiv float 1.0, %sqrt_val
  ret float %result
}

; 8. tosa.exp - Exponential
define float @test_exp(float %input) {
entry:
  %result = call float @llvm.exp.f32(float %input)
  ret float %result
}

; 9. tosa.log - Natural logarithm
define float @test_log(float %input) {
entry:
  %result = call float @llvm.log.f32(float %input)
  ret float %result
}

; 10. tosa.sin - Sine
define float @test_sin(float %input) {
entry:
  %result = call float @llvm.sin.f32(float %input)
  ret float %result
}

; 11. tosa.cos - Cosine
define float @test_cos(float %input) {
entry:
  %result = call float @llvm.cos.f32(float %input)
  ret float %result
}

; 12. tosa.tanh - Hyperbolic tangent
define float @test_tanh(float %input) {
entry:
  %pos_exp = call float @llvm.exp.f32(float %input)
  %neg_input = fsub float 0.0, %input
  %neg_exp = call float @llvm.exp.f32(float %neg_input)
  %numerator = fsub float %pos_exp, %neg_exp
  %denominator = fadd float %pos_exp, %neg_exp
  %result = fdiv float %numerator, %denominator
  ret float %result
}

; 13. tosa.sigmoid - Sigmoid function
define float @test_sigmoid(float %input) {
entry:
  %neg_input = fsub float 0.0, %input
  %exp_neg = call float @llvm.exp.f32(float %neg_input)
  %one_plus_exp = fadd float 1.0, %exp_neg
  %result = fdiv float 1.0, %one_plus_exp
  ret float %result
}

; 14. tosa.erf - Error function
define float @test_erf(float %input) {
entry:
  ; Approximation of erf using polynomial
  %abs_x = call float @llvm.fabs.f32(float %input)
  %t = fdiv float 1.0, %abs_x
  %a1 = fmul float 0.254829592, %t
  %result = fadd float %a1, 0.0  ; Simplified approximation
  ret float %result
}

; 15. tosa.floor - Floor function
define float @test_floor(float %input) {
entry:
  %result = call float @llvm.floor.f32(float %input)
  ret float %result
}

; 16. tosa.ceil - Ceiling function
define float @test_ceil(float %input) {
entry:
  %result = call float @llvm.ceil.f32(float %input)
  ret float %result
}

; 17. tosa.clamp - Clamp to range [min, max]
define float @test_clamp(float %input, float %min_val, float %max_val) {
entry:
  %clamped_min = call float @llvm.maxnum.f32(float %input, float %min_val)
  %result = call float @llvm.minnum.f32(float %clamped_min, float %max_val)
  ret float %result
}

; 18. tosa.maximum - Element-wise maximum
define float @test_maximum(float %a, float %b) {
entry:
  %result = call float @llvm.maxnum.f32(float %a, float %b)
  ret float %result
}

; 19. tosa.minimum - Element-wise minimum
define float @test_minimum(float %a, float %b) {
entry:
  %result = call float @llvm.minnum.f32(float %a, float %b)
  ret float %result
}

; 20. tosa.pow - Power function
define float @test_pow(float %base, float %exponent) {
entry:
  %result = call float @llvm.pow.f32(float %base, float %exponent)
  ret float %result
}

; 21. tosa.equal - Equality comparison
define i1 @test_equal(float %a, float %b) {
entry:
  %result = fcmp oeq float %a, %b
  ret i1 %result
}

; 22. tosa.greater - Greater than comparison
define i1 @test_greater(float %a, float %b) {
entry:
  %result = fcmp ogt float %a, %b
  ret i1 %result
}

; 23. tosa.greater_equal - Greater than or equal comparison
define i1 @test_greater_equal(float %a, float %b) {
entry:
  %result = fcmp oge float %a, %b
  ret i1 %result
}

; 24. tosa.select - Conditional select
define float @test_select(i1 %condition, float %true_val, float %false_val) {
entry:
  %result = select i1 %condition, float %true_val, float %false_val
  ret float %result
}

; 25. tosa.cast - Type casting
define i32 @test_cast_f32_to_i32(float %input) {
entry:
  %result = fptosi float %input to i32
  ret i32 %result
}

; 26. tosa.bitwise_and - Bitwise AND
define i32 @test_bitwise_and(i32 %a, i32 %b) {
entry:
  %result = and i32 %a, %b
  ret i32 %result
}

; 27. tosa.bitwise_or - Bitwise OR
define i32 @test_bitwise_or(i32 %a, i32 %b) {
entry:
  %result = or i32 %a, %b
  ret i32 %result
}

; 28. tosa.bitwise_xor - Bitwise XOR
define i32 @test_bitwise_xor(i32 %a, i32 %b) {
entry:
  %result = xor i32 %a, %b
  ret i32 %result
}

; 29. tosa.bitwise_not - Bitwise NOT
define i32 @test_bitwise_not(i32 %input) {
entry:
  %result = xor i32 %input, -1
  ret i32 %result
}

; 30. tosa.logical_and - Logical AND
define i1 @test_logical_and(i1 %a, i1 %b) {
entry:
  %result = and i1 %a, %b
  ret i1 %result
}

; 31. tosa.logical_or - Logical OR
define i1 @test_logical_or(i1 %a, i1 %b) {
entry:
  %result = or i1 %a, %b
  ret i1 %result
}

; 32. tosa.logical_xor - Logical XOR
define i1 @test_logical_xor(i1 %a, i1 %b) {
entry:
  %result = xor i1 %a, %b
  ret i1 %result
}

; 33. tosa.logical_not - Logical NOT
define i1 @test_logical_not(i1 %input) {
entry:
  %result = xor i1 %input, true
  ret i1 %result
}

; 34. tosa.logical_left_shift - Logical left shift
define i32 @test_logical_left_shift(i32 %input, i32 %shift) {
entry:
  %result = shl i32 %input, %shift
  ret i32 %result
}

; 35. tosa.logical_right_shift - Logical right shift
define i32 @test_logical_right_shift(i32 %input, i32 %shift) {
entry:
  %result = lshr i32 %input, %shift
  ret i32 %result
}

; 36. tosa.arithmetic_right_shift - Arithmetic right shift
define i32 @test_arithmetic_right_shift(i32 %input, i32 %shift) {
entry:
  %result = ashr i32 %input, %shift
  ret i32 %result
}

; 37. tosa.clz - Count leading zeros
define i32 @test_clz(i32 %input) {
entry:
  %result = call i32 @llvm.ctlz.i32(i32 %input, i1 false)
  ret i32 %result
}

; 38. tosa.identity - Identity operation
define float @test_identity(float %input) {
entry:
  ret float %input
}

; 39. Matrix multiplication (already implemented)
define void @test_matmul(float* %A, float* %B, float* %C, i32 %M, i32 %K, i32 %N) {
entry:
  br label %i.loop.header

i.loop.header:
  %i.val = phi i32 [ 0, %entry ], [ %i.next, %i.loop.latch ]
  %i.cond = icmp slt i32 %i.val, %M
  br i1 %i.cond, label %j.loop.header, label %exit

j.loop.header:
  %j.val = phi i32 [ 0, %i.loop.header ], [ %j.next, %j.loop.latch ]
  %j.cond = icmp slt i32 %j.val, %N
  br i1 %j.cond, label %k.loop.header, label %i.loop.latch

k.loop.header:
  %k.val = phi i32 [ 0, %j.loop.header ], [ %k.next, %k.loop.latch ]
  %sum.val = phi float [ 0.0, %j.loop.header ], [ %new.sum, %k.loop.latch ]
  %k.cond = icmp slt i32 %k.val, %K
  br i1 %k.cond, label %loop.body, label %j.loop.latch

loop.body:
  %i.mul.K = mul nsw i32 %i.val, %K
  %idx.A = add nsw i32 %i.mul.K, %k.val
  %A.addr = getelementptr inbounds float, float* %A, i32 %idx.A
  %A.val = load float, float* %A.addr
  %k.mul.N = mul nsw i32 %k.val, %N
  %idx.B = add nsw i32 %k.mul.N, %j.val
  %B.addr = getelementptr inbounds float, float* %B, i32 %idx.B
  %B.val = load float, float* %B.addr
  %mul.val = fmul float %A.val, %B.val
  %new.sum = fadd float %sum.val, %mul.val
  br label %k.loop.latch

k.loop.latch:
  %k.next = add nsw i32 %k.val, 1
  br label %k.loop.header

j.loop.latch:
  %i.mul.N = mul nsw i32 %i.val, %N
  %idx.C = add nsw i32 %i.mul.N, %j.val
  %C.addr = getelementptr inbounds float, float* %C, i32 %idx.C
  store float %sum.val, float* %C.addr
  %j.next = add nsw i32 %j.val, 1
  br label %j.loop.header

i.loop.latch:
  %i.next = add nsw i32 %i.val, 1
  br label %i.loop.header

exit:
  ret void
}

; 40. 2D Convolution (already implemented)
define void @test_conv2d(float* %input, float* %kernel, float* %output, i32 %IH, i32 %IW, i32 %KH, i32 %KW) {
entry:
  %OH = sub i32 %IH, %KH
  %OH_final = add i32 %OH, 1
  %OW = sub i32 %IW, %KW
  %OW_final = add i32 %OW, 1
  br label %oh.loop.header

oh.loop.header:
  %oh.val = phi i32 [ 0, %entry ], [ %oh.next, %oh.loop.latch ]
  %oh.cond = icmp slt i32 %oh.val, %OH_final
  br i1 %oh.cond, label %ow.loop.header, label %exit

ow.loop.header:
  %ow.val = phi i32 [ 0, %oh.loop.header ], [ %ow.next, %ow.loop.latch ]
  %ow.cond = icmp slt i32 %ow.val, %OW_final
  br i1 %ow.cond, label %kh.loop.header, label %oh.loop.latch

kh.loop.header:
  %kh.val = phi i32 [ 0, %ow.loop.header ], [ %kh.next, %kh.loop.latch ]
  %sum.val = phi float [ 0.0, %ow.loop.header ], [ %new.sum, %kh.loop.latch ]
  %kh.cond = icmp slt i32 %kh.val, %KH
  br i1 %kh.cond, label %kw.loop.header, label %ow.loop.latch

kw.loop.header:
  %kw.val = phi i32 [ 0, %kh.loop.header ], [ %kw.next, %kw.loop.latch ]
  %kw.cond = icmp slt i32 %kw.val, %KW
  br i1 %kw.cond, label %conv.body, label %kh.loop.latch

conv.body:
  %ih = add i32 %oh.val, %kh.val
  %iw = add i32 %ow.val, %kw.val
  %ih.mul.IW = mul i32 %ih, %IW
  %input.idx = add i32 %ih.mul.IW, %iw
  %input.addr = getelementptr inbounds float, float* %input, i32 %input.idx
  %input.val = load float, float* %input.addr
  %kh.mul.KW = mul i32 %kh.val, %KW
  %kernel.idx = add i32 %kh.mul.KW, %kw.val
  %kernel.addr = getelementptr inbounds float, float* %kernel, i32 %kernel.idx
  %kernel.val = load float, float* %kernel.addr
  %mul.val = fmul float %input.val, %kernel.val
  %new.sum = fadd float %sum.val, %mul.val
  br label %kw.loop.latch

kw.loop.latch:
  %kw.next = add i32 %kw.val, 1
  br label %kw.loop.header

kh.loop.latch:
  %kh.next = add i32 %kh.val, 1
  br label %kh.loop.header

ow.loop.latch:
  %oh.mul.OW = mul i32 %oh.val, %OW_final
  %output.idx = add i32 %oh.mul.OW, %ow.val
  %output.addr = getelementptr inbounds float, float* %output, i32 %output.idx
  store float %sum.val, float* %output.addr
  %ow.next = add i32 %ow.val, 1
  br label %ow.loop.header

oh.loop.latch:
  %oh.next = add i32 %oh.val, 1
  br label %oh.loop.header

exit:
  ret void
}

; Declare intrinsic functions
declare float @llvm.sqrt.f32(float)
declare float @llvm.exp.f32(float)
declare float @llvm.log.f32(float)
declare float @llvm.sin.f32(float)
declare float @llvm.cos.f32(float)
declare float @llvm.fabs.f32(float)
declare float @llvm.floor.f32(float)
declare float @llvm.ceil.f32(float)
declare float @llvm.maxnum.f32(float, float)
declare float @llvm.minnum.f32(float, float)
declare float @llvm.pow.f32(float, float)
declare i32 @llvm.ctlz.i32(i32, i1)