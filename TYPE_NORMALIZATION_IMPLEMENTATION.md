# Type Normalization Implementation for LoadStoreVectorizer

## Summary

Successfully implemented type normalization support in the LoadStoreVectorizer pass to enable vectorization of mixed-type loads and stores (e.g., `i32` and `<2 x i16>`). The implementation follows a lazy normalization approach where bitcasts are only inserted after chains survive all splitting criteria and pass a profitability check.

## Changes Made

### 1. Command-Line Flag
- Added `-load-store-vectorizer-normalize-types` flag (default: false)
- Located in: `LoadStoreVectorizer.cpp:125-129`
- Allows opt-in experimentation with the feature

### 2. Equivalence Class Grouping
- **Modified**: `EqClassKey` to conditionally group by total type size (when flag enabled) vs scalar element size (original behavior)
- **Location**: `LoadStoreVectorizer.cpp:132-143`
- **Impact**: Allows `i32` (32 bits) and `<2 x i16>` (32 bits total) in the same equivalence class when normalization is enabled

### 3. Chain Element Tracking
- **Extended**: `ChainElem` struct with `OriginalType` field
- **Location**: `LoadStoreVectorizer.cpp:157-168`
- **Purpose**: Tracks original load/store types for proper bitcast insertion

### 4. Type Normalization Analysis
- **New Method**: `analyzeAndNormalizeChain(const Chain &C)`
- **Location**: `LoadStoreVectorizer.cpp:719-760`
- **Logic**: 
  - Returns `std::nullopt` if all types match (no normalization needed)
  - Computes GCD of element sizes to find finest granularity type
  - Examples:
    - `{i32, <2 x i16>}` → normalize to `i16`
    - `{<4 x i8>, i32}` → normalize to `i8`

### 5. Profitability Analysis
- **New Method**: `isProfitableToNormalize(const Chain &C, Type *NormalizedElemTy)`
- **Location**: `LoadStoreVectorizer.cpp:762-827`
- **Cost Model**:
  - Estimates scalar access costs (sum of individual loads/stores)
  - Estimates vectorized access cost
  - Estimates bitcast overhead
  - Returns `true` if `(vector_cost + cast_cost) < scalar_cost`
  - Uses TTI for platform-specific cost queries

### 6. Enhanced getChainElemTy
- **Modified**: Added optional `NormalizedElemTy` parameter
- **Location**: `LoadStoreVectorizer.cpp:686-715`
- **Behavior**: Returns normalized type if provided, otherwise uses original logic

### 7. Updated vectorizeChain
- **Modified**: Beginning of function to check for normalization needs
- **Location**: `LoadStoreVectorizer.cpp:1042-1062`
- **Flow**:
  1. Call `analyzeAndNormalizeChain()` to check for mixed types
  2. If normalization needed, call `isProfitableToNormalize()`
  3. If not profitable, skip vectorization
  4. Otherwise, proceed with normalized element type

### 8. Load Chain Bitcasting
- **Modified**: Load value extraction logic
- **Location**: `LoadStoreVectorizer.cpp:1106-1147`
- **Changes**:
  - Uses `E.OriginalType` instead of `getLoadStoreType(I)`
  - Adds bitcasts from normalized type to original type when needed
  - Handles both scalar and vector original types

### 9. Store Chain Bitcasting
- **Modified**: Store value building logic
- **Location**: `LoadStoreVectorizer.cpp:1155-1199`
- **Changes**:
  - Bitcasts store values from original type to normalized type
  - Handles vector store values by bitcasting entire vector first
  - Then extracts and inserts elements into final vector

### 10. Assertion Updates
- **Modified**: Relaxed assertion that checks scalar type sizes
- **Location**: `LoadStoreVectorizer.cpp:1081-1089`
- **Reason**: With normalization, original types may legitimately differ

## Test Cases Created

### 1. General Tests (`type-normalization.ll`)
- Basic i32 + <2 x i16> loads and stores
- <4 x i8> + i32 combinations
- float + i32 combinations
- Chains with aliasing (should split properly)
- Non-contiguous accesses (should not vectorize)
- Three-type combinations (i32, <2 x i16>, i16)
- Misaligned accesses
- Verification without flag (should not vectorize mixed types)

### 2. X86 Tests (`X86/type-normalization-x86.ll`)
- Platform-specific tests for X86
- Verifies behavior with and without flag

### 3. NVPTX/CUDA Tests (`NVPTX/type-normalization-cuda.ll`)
- GPU-specific tests (main target for this pass)
- Tests with global memory (address space 1)
- Tests with shared memory (address space 3)
- Mixed float/int types

## Architecture Highlights

### Lazy Normalization
- Type normalization happens **after** chains are:
  - Grouped into equivalence classes
  - Built via offset analysis
  - Split by may-alias instructions
  - Split by contiguity requirements
  - Split by alignment requirements
- This avoids wasteful casts on chains that won't be vectorized anyway

### Cost Model Integration
- Leverages existing TTI infrastructure
- Platform can return appropriate costs for:
  - Memory operations (vectorized vs scalar)
  - Bitcast operations
- Guards normalization with profitability check

### Backward Compatibility
- Feature is opt-in via command-line flag
- When disabled, behavior is identical to original pass
- Existing tests should continue to pass

## Future Enhancements

1. **Sub-byte types**: Currently handles i8 and larger. Could extend to i1, i4, etc.
2. **Adaptive thresholds**: Could add flag for maximum acceptable cast overhead percentage
3. **Pointer type mixing**: Currently excludes pointer types. Could handle with ptrtoint/inttoptr
4. **Auto-enable for specific targets**: Could default to enabled for GPU targets

## How to Use

```bash
# Enable type normalization
opt -passes=load-store-vectorizer -load-store-vectorizer-normalize-types < input.ll > output.ll

# Without flag (original behavior)
opt -passes=load-store-vectorizer < input.ll > output.ll
```

## Files Modified

1. `llvm/lib/Transforms/Vectorize/LoadStoreVectorizer.cpp` - Main implementation
2. `llvm/test/Transforms/LoadStoreVectorizer/type-normalization.ll` - General tests
3. `llvm/test/Transforms/LoadStoreVectorizer/X86/type-normalization-x86.ll` - X86 tests
4. `llvm/test/Transforms/LoadStoreVectorizer/NVPTX/type-normalization-cuda.ll` - CUDA tests

