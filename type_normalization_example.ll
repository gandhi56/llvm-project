; Example demonstrating type normalization in LoadStoreVectorizer
; This shows how i32 and <2 x i16> loads can be vectorized together

; Without type normalization (default behavior):
; opt -S < example.ll -passes=load-store-vectorizer
; Result: Two separate loads, no vectorization

; With type normalization enabled:
; opt -S < example.ll -passes=load-store-vectorizer -load-store-vectorizer-normalize-types
; Result: Vectorized load with bitcasts

define void @example_mixed_types(ptr %base, ptr %out) {
entry:
  ; Load an i32 from base pointer
  %val0 = load i32, ptr %base, align 4
  
  ; Load a <2 x i16> from base+4 (contiguous with the i32)
  %ptr1 = getelementptr i32, ptr %base, i64 1
  %val1 = load <2 x i16>, ptr %ptr1, align 4
  
  ; Store results
  store i32 %val0, ptr %out, align 4
  %out1 = getelementptr i32, ptr %out, i64 1
  store <2 x i16> %val1, ptr %out1, align 4
  
  ret void
}

; Expected output with type normalization enabled:
; The loads would be combined into a single <4 x i16> vector load,
; then bitcast/extracted to produce the original i32 and <2 x i16> values.
;
; Similarly, the stores would be combined by bitcasting values to i16,
; building a <4 x i16> vector, and storing it in one operation.
;
; Benefits:
; - Fewer memory transactions (especially important on GPUs)
; - Better memory coalescing
; - Improved bandwidth utilization
; 
; Trade-offs:
; - Additional bitcast instructions
; - Only applied when cost model determines it's profitable

