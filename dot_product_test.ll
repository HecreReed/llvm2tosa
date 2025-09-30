; ModuleID = 'dot_product.c'
; target triple = "x86_64-pc-linux-gnu"

; 定义 dot_product 函数，返回一个 float 标量
define float @dot_product(float* %a, float* %b, i32 %n) {
entry:
  ; 检查 n > 0，否则直接返回 0.0
  %start.cond = icmp sgt i32 %n, 0
  br i1 %start.cond, label %loop.header, label %loop.exit.early

loop.header:
  ; 循环变量 i 的 phi 节点
  %i.val = phi i32 [ 0, %entry ], [ %i.next, %loop.latch ]
  ; 累加器 sum 的 phi 节点
  ; 第一次进入时，sum 的值为 0.0
  ; 后续迭代时，sum 的值为上一次计算出的 %new.sum
  %sum.val = phi float [ 0.0, %entry ], [ %new.sum, %loop.latch ]
  
  ; 循环条件: i < n ?
  %loop.cond = icmp slt i32 %i.val, %n
  br i1 %loop.cond, label %loop.body, label %loop.exit

loop.body:
  ; --- 核心计算: sum = sum + a[i] * b[i] ---
  
  ; 1. 加载 a[i] 和 b[i] 的值
  %a.addr = getelementptr inbounds float, float* %a, i32 %i.val
  %a.val = load float, float* %a.addr
  %b.addr = getelementptr inbounds float, float* %b, i32 %i.val
  %b.val = load float, float* %b.addr
  
  ; 2. 计算 a[i] * b[i]
  %mul.val = fmul float %a.val, %b.val
  
  ; 3. 将乘积与当前的 sum 相加，得到新的 sum
  %new.sum = fadd float %sum.val, %mul.val
  
  br label %loop.latch

loop.latch:
  ; 循环变量 i 递增
  %i.next = add nsw i32 %i.val, 1
  br label %loop.header

loop.exit:
  ; 循环正常结束后，返回最终的累加值 %sum.val
  ret float %sum.val

loop.exit.early:
  ; 如果 n <= 0，直接返回 0.0
  ret float 0.0
}