; ModuleID = 'dot_product.c'
; target triple = "x86_64-pc-linux-gnu"

; dot_product function returns a float scalar
define float @dot_product(float* %a, float* %b, i32 %n) {
entry:
  ; check n > 0, otherwise return 0.0 directly
  %start.cond = icmp sgt i32 %n, 0
  br i1 %start.cond, label %loop.header, label %loop.exit.early

loop.header:
  ; loop variable i phi node
  %i.val = phi i32 [ 0, %entry ], [ %i.next, %loop.latch ]
  ; accumulator sum phi node
  ; first iteration: sum = 0.0
  ; subsequent iterations: sum = previous %new.sum
  %sum.val = phi float [ 0.0, %entry ], [ %new.sum, %loop.latch ]
  
  ; loop condition: i < n ?
  %loop.cond = icmp slt i32 %i.val, %n
  br i1 %loop.cond, label %loop.body, label %loop.exit

loop.body:
  ; --- core computation: sum = sum + a[i] * b[i] ---
  
  ; 1. load a[i] and b[i] values
  %a.addr = getelementptr inbounds float, float* %a, i32 %i.val
  %a.val = load float, float* %a.addr
  %b.addr = getelementptr inbounds float, float* %b, i32 %i.val
  %b.val = load float, float* %b.addr
  
  ; 2. compute a[i] * b[i]
  %mul.val = fmul float %a.val, %b.val
  
  ; 3. add product to current sum
  %new.sum = fadd float %sum.val, %mul.val
  
  br label %loop.latch

loop.latch:
  ; increment loop variable i
  %i.next = add nsw i32 %i.val, 1
  br label %loop.header

loop.exit:
  ; return final accumulated value %sum.val
  ret float %sum.val

loop.exit.early:
  ; if n <= 0, return 0.0 directly
  ret float 0.0
}