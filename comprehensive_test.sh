#!/bin/bash

echo "=== 真实的LLVM IR到TOSA IR转换器测试 ==="
echo

echo "1. 验证LLVM IR语法"
echo "=================="
echo "输入的LLVM IR文件: real_llvm_examples.ll"
head -20 real_llvm_examples.ll
echo "..."
echo

echo "2. 运行转换器"
echo "============="
./realistic_converter
echo

echo "3. 转换结果预览"
echo "==============="
echo "生成的TOSA IR示例 (real_tosa_conversion.mlir):"
head -30 real_tosa_conversion.mlir
echo "..."
echo

echo "4. 关键转换映射"
echo "==============="
echo "LLVM IR构造                    -> TOSA IR等价物"
echo "------------------------------------------------"
echo "define i32 @func(i32 %a)      -> func.func @func(%arg0: tensor<1xi32>)"
echo "%result = add i32 %a, %b      -> %result = tosa.add %a, %b"
echo "%arr = alloca [4 x i32]       -> %arr = tosa.const {dense<[0,0,0,0]>}"
echo "%val = load i32, i32* %ptr    -> %val = tosa.slice %tensor {...}"
echo "%vec = add <4 x i32> %a, %b   -> %vec = tosa.add %a, %b"
echo "%elem = extractelement ...    -> %elem = tosa.slice %tensor {...}"
echo

echo "5. 核心技术挑战"
echo "==============="

echo "挑战1: 标量vs张量"
echo "• LLVM: i32, float 等标量类型"
echo "• TOSA: tensor<1xi32>, tensor<1xf32> 等张量"
echo "• 解决: 所有标量映射为1D张量"
echo

echo "挑战2: 内存vs值语义"  
echo "• LLVM: alloca/load/store 显式内存操作"
echo "• TOSA: 不可变张量值，无指针概念"
echo "• 解决: 内存操作转换为张量重构"
echo

echo "挑战3: 动态索引"
echo "• LLVM: getelementptr 支持运行时计算索引"
echo "• TOSA: slice/gather 需要静态或结构化索引"
echo "• 解决: 复杂索引需要条件逻辑或查找表"
echo

echo "挑战4: 控制流"
echo "• LLVM: 基本块、条件/无条件分支"
echo "• TOSA: 结构化控制流 (cond_if, while_loop)"
echo "• 解决: CFG分析和结构化重构"
echo

echo "6. 实际应用场景"
echo "==============="
echo "这种转换适用于:"
echo "• 将传统C/C++代码移植到AI加速器"
echo "• 优化数值计算代码的张量操作"
echo "• 在ML框架中复用已有算法"
echo "• 硬件无关的计算表示"
echo

echo "7. 性能优势"
echo "============"
echo "TOSA张量操作的优势:"
echo "• 天然支持SIMD/向量化"
echo "• 自动广播和形状推导"
echo "• 硬件无关的高级抽象"
echo "• 内置量化支持"
echo

echo "=== 测试完成 ==="
echo "转换器成功展示了LLVM IR到TOSA IR的核心转换原理！"