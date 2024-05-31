/*
file: asm_linreg.s
author: LuigiG - LG@THLG.NL
license: MIT

ARM64 NEON SIMD implementation of linear regression

*/
.global _mean
.global _linreg

.p2align 2
// CALC MEAN
// in: x0 = array ptr
//     x1 = num of elements
// out: d0 = mean
_mean:
    eor v4.16b, v4.16b, v4.16b // zero init v4-v7 as accumulator
    eor v5.16b, v5.16b, v5.16b
    eor v6.16b, v6.16b, v6.16b
    eor v7.16b, v7.16b, v7.16b
    eor v0.16b, v0.16b, v0.16b
    and x3, x1, 7   // x3 = x1 % 8
    lsr x2, x1, 3   // x2 = x1 / 8
    cmp xzr, x2
    beq .skip1
.loop1:
    ld1 {v0.2d, v1.2d, v2.2d, v3.2d}, [x0], #64  // ld 8 elms - for more on ld1 see https://eclecticlight.co/2021/08/23/code-in-arm-assembly-lanes-and-loads-in-neon/
    fadd v4.2d, v4.2d, v0.2d  // add/accumulate 
    fadd v5.2d, v5.2d, v1.2d   
    fadd v6.2d, v6.2d, v2.2d
    fadd v7.2d, v7.2d, v3.2d
    subs x2, x2, 1    
    bne .loop1
    fadd v4.2d, v4.2d, v5.2d  // sideways-add the 8 values into d0
    fadd v4.2d, v4.2d, v6.2d
    fadd v4.2d, v4.2d, v7.2d    
    faddp d0, v4.2d
.skip1:
    cmp xzr, x3   // add the remaining values (if any) 1 at a time, 
    beq .skip2                       
.loop2:                              
    ldr d1, [x0], #8                  
    fadd d0, d0, d1                  
    subs x3, x3, 1                   
    bne .loop2                       
.skip2:                              
    ucvtf d1, x1  // finally, divide by number of values to get the mean
    fdiv  d0, d0, d1
.end:
    ret

.p2align 2
// CALC LINEAR REGRESSION
// in: x0 = ptr array x
//     x1 = ptr array y
//     x2 = num elms
//     x3 = ptr result slope
//     x4 = ptr result intercept
_linreg:
    stp x29, x30, [sp, #-16]!
    stp x19, x20, [sp, #-16]!
    stp x21, x22, [sp, #-16]!
    stp x23, x24, [sp, #-16]!
    stp q8, q12,  [sp, #-32]!

    mov x19, x0   // x19 = ptr array x
    mov x20, x1   // x20 = ptr array y
    mov x21, x2   // x21 = num elms
    mov x22, x3   // x22 = ptr result 1
    mov x23, x4   // x23 = ptr result 2
    
    mov x0, x19   // calc mean_x
    mov x1, x21
    bl _mean   
    fmov d30, d0 // d30 = mean_x
    
    mov x0, x20  // calc mean_y
    mov x1, x21
    bl _mean   
    fmov d31, d0 // d31 = mean_y

    eor v0.16b, v0.16b, v0.16b // zero the accumulators
    eor v1.16b, v1.16b, v1.16b
    eor v2.16b, v2.16b, v2.16b
    eor v3.16b, v3.16b, v3.16b
    
    eor v4.16b, v4.16b, v4.16b
    eor v5.16b, v5.16b, v5.16b
    eor v6.16b, v6.16b, v6.16b
    eor v7.16b, v7.16b, v7.16b
    
    dup v8.2d, v30.d[0]   // broadcast x_mean
    dup v12.2d, v31.d[0]  // broadcast y_mean
    
    mov x0, x19           // init loop counters & pointers
    mov x1, x20
    and x3, x21, 7        // x3 = num.elm % 8
    lsr x2, x21, 3        // x2 = num.elm / 8
    cmp xzr, x2
    beq .lrskip1
 .lr1:
    ld1 {v16.2d, v17.2d, v18.2d, v19.2d}, [x0], #64    // load 8 x elements
    ld1 {v20.2d, v21.2d, v22.2d, v23.2d}, [x1], #64    // load 8 y elements
 
    fsub v16.2d, v16.2d, v8.2d  // x - mean_x
    fsub v17.2d, v17.2d, v8.2d  // x - mean_x
    fsub v18.2d, v18.2d, v8.2d  // x - mean_x
    fsub v19.2d, v19.2d, v8.2d  // x - mean_x
 
    fsub v20.2d, v20.2d, v12.2d  // y - mean_y
    fsub v21.2d, v21.2d, v12.2d  // y - mean_y
    fsub v22.2d, v22.2d, v12.2d  // y - mean_y
    fsub v23.2d, v23.2d, v12.2d  // y - mean_y
 
    fmul v20.2d, v20.2d, v16.2d  // (x - mean_x) * (y - mean_y)
    fmul v21.2d, v21.2d, v17.2d  // (x - mean_x) * (y - mean_y)
    fmul v22.2d, v22.2d, v18.2d  // (x - mean_x) * (y - mean_y)
    fmul v23.2d, v23.2d, v19.2d  // (x - mean_x) * (y - mean_y)
 
    fadd v0.2d, v0.2d, v20.2d    // accumulate numerator
    fadd v1.2d, v1.2d, v21.2d    // accumulate numerator
    fadd v2.2d, v2.2d, v22.2d    // accumulate numerator
    fadd v3.2d, v3.2d, v23.2d    // accumulate numerator
 
    fmul v16.2d, v16.2d, v16.2d  // (x - mean_x)²
    fmul v17.2d, v17.2d, v17.2d  // (x - mean_x)²
    fmul v18.2d, v18.2d, v18.2d  // (x - mean_x)²
    fmul v19.2d, v19.2d, v19.2d  // (x - mean_x)²
 
    fadd v4.2d, v4.2d, v16.2d   // accumulate denominator
    fadd v5.2d, v5.2d, v17.2d   // accumulate denominator
    fadd v6.2d, v6.2d, v18.2d   // accumulate denominator
    fadd v7.2d, v7.2d, v19.2d   // accumulate denominator
 
    subs x2, x2, 1    
    bne .lr1

    fadd v0.2d, v0.2d, v1.2d   // reduce numerator vectors
    fadd v0.2d, v0.2d, v2.2d
    fadd v0.2d, v0.2d, v3.2d    
    faddp d0, v0.2d
    
    fadd v4.2d, v4.2d, v5.2d   // reduce denominator vectors
    fadd v4.2d, v4.2d, v6.2d
    fadd v4.2d, v4.2d, v7.2d    
    faddp d1, v4.2d
.lrskip1:
    cmp xzr, x3                // accumulate the remaining values (if any) 1 at a time, 
    beq .lrskip2                       
.lr2:                              
    ldr d2, [x0], #8 // load x = x[i]
    ldr d3, [x1], #8 // load y = y[i]
    fsub d4, d2, d30 // x - mean_x
    fsub d5, d3, d31 // y - mean_y
    fmul d5, d5, d4  // (x - mean_x) * (y - mean_y)
    fadd d0, d0, d5  // accumulate numerator
    fmul d4, d4, d4  // (x - mean_x)²
    fadd d1, d1, d4  // accumulate denominator
    subs x3, x3, 1
    bne .lr2
.lrskip2:                              
    fdiv d2, d0, d1  // slope = numerator / denominator = sum[i..n]((y[i] - mean_y) * (x[i] - mean_x)) / sum[i..n]((x[i] - mean_x)²) 
    fmul d3, d2, d30 // intercept = y_mean - (slope * x_mean);
    fsub d3, d31, d3
    str d2, [x22]
    str d3, [x23]

    ldp q8, q12,  [sp], #32
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16   
    ldp x29, x30, [sp], #16
    ret
