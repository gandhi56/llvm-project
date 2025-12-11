# Fix: Split Functions Now Normalization-Aware

## Summary

Fixed an architectural inconsistency where `splitChainByContiguity` and `splitChainByAlignment` were using the original type selection heuristic instead of the normalized type. This caused them to make splitting decisions based on different assumptions than `vectorizeChain`, potentially leading to suboptimal vectorization.

## Problem Identified

### Before Fix
```
splitChainByContiguity: 
  Calls getChainElemTy(C) → Returns i32 (original heuristic)
  Makes contiguity decisions assuming 32-bit elements

splitChainByAlignment:
  Calls getChainElemTy(C) → Returns i32 (original heuristic)
  Calculates alignment assuming 32-bit elements

vectorizeChain:
  Calls analyzeAndNormalizeChain(C) → Determines i16 (normalized)
  Performs vectorization with 16-bit elements
  
RESULT: Mismatch! Different element type assumptions across pipeline stages.
```

### After Fix
```
splitChainByContiguity:
  Calls analyzeAndNormalizeChain(C) → Determines i16
  Calls getChainElemTy(C, i16) → Returns i16
  Makes contiguity decisions with 16-bit elements

splitChainByAlignment:
  Calls analyzeAndNormalizeChain(C) → Determines i16
  Calls getChainElemTy(C, i16) → Returns i16
  Calculates alignment with 16-bit elements

vectorizeChain:
  Calls analyzeAndNormalizeChain(C) → Determines i16
  Performs vectorization with 16-bit elements
  
RESULT: Consistent! All stages use the same element type.
```

## Changes Made

### 1. Updated splitChainByContiguity (Lines 636-657)

**Before**:
```cpp
std::vector<Chain> Vectorizer::splitChainByContiguity(Chain &C) {
  if (C.empty())
    return {};
  sortChainInOffsetOrder(C);
  
  LLVM_DEBUG({
    dbgs() << "LSV: splitChainByContiguity considering chain:\n";
    dumpChain(C);
  });
  
  std::vector<Chain> Ret;
  Ret.push_back({C.front()});
  
  unsigned ChainElemTyBits = DL.getTypeSizeInBits(getChainElemTy(C));
  // ... rest of function
}
```

**After**:
```cpp
std::vector<Chain> Vectorizer::splitChainByContiguity(Chain &C) {
  if (C.empty())
    return {};
  sortChainInOffsetOrder(C);
  
  // Determine if we need type normalization for this chain
  std::optional<Type *> MaybeNormalizedTy = analyzeAndNormalizeChain(C);
  Type *NormalizedElemTy = MaybeNormalizedTy.value_or(nullptr);
  
  LLVM_DEBUG({
    dbgs() << "LSV: splitChainByContiguity considering chain:\n";
    dumpChain(C);
    if (NormalizedElemTy)
      dbgs() << "LSV:   Using normalized type: " << *NormalizedElemTy << "\n";
  });
  
  std::vector<Chain> Ret;
  Ret.push_back({C.front()});
  
  // Use normalized type if available, otherwise use heuristic
  unsigned ChainElemTyBits = DL.getTypeSizeInBits(getChainElemTy(C, NormalizedElemTy));
  // ... rest of function unchanged
}
```

**Key changes**:
- Added normalization analysis at the start
- Added debug output when normalized type is used
- Pass normalized type to `getChainElemTy()`

### 2. Updated splitChainByAlignment (Lines 841-865, 912-914)

**Before (at function start)**:
```cpp
std::vector<Chain> Vectorizer::splitChainByAlignment(Chain &C) {
  if (C.empty())
    return {};
  sortChainInOffsetOrder(C);
  
  LLVM_DEBUG({
    dbgs() << "LSV: splitChainByAlignment considering chain:\n";
    dumpChain(C);
  });
  // ... rest
}
```

**After (at function start)**:
```cpp
std::vector<Chain> Vectorizer::splitChainByAlignment(Chain &C) {
  if (C.empty())
    return {};
  sortChainInOffsetOrder(C);
  
  // Determine if we need type normalization for this chain
  std::optional<Type *> MaybeNormalizedTy = analyzeAndNormalizeChain(C);
  Type *NormalizedElemTy = MaybeNormalizedTy.value_or(nullptr);
  
  LLVM_DEBUG({
    dbgs() << "LSV: splitChainByAlignment considering chain:\n";
    dumpChain(C);
    if (NormalizedElemTy)
      dbgs() << "LSV:   Using normalized type: " << *NormalizedElemTy << "\n";
  });
  // ... rest
}
```

**Before (line 912)**:
```cpp
Type *VecElemTy = getChainElemTy(C);
```

**After (line 914)**:
```cpp
// Use normalized type if available, otherwise use heuristic
Type *VecElemTy = getChainElemTy(C, NormalizedElemTy);
```

**Key changes**:
- Added normalization analysis at function start
- Added debug output when normalized type is used
- Pass normalized type to `getChainElemTy()` in the inner loop

### 3. Created Test Case

**File**: `llvm/test/Transforms/LoadStoreVectorizer/X86/consistent-normalization.ll`

Tests verify:
- All split stages log "Using normalized type: i16" for {i32, <2 x i16>}
- All split stages log "Using normalized type: i8" for {<4 x i8>, i32}
- Uniform type chains don't trigger normalization
- Both loads and stores maintain consistency

## Performance Considerations

### Trade-off Analysis

**Cost**: `analyzeAndNormalizeChain()` is now called up to 3 times per chain:
1. Once in `splitChainByContiguity`
2. Once in `splitChainByAlignment`
3. Once in `vectorizeChain`

**Mitigation**:
- `analyzeAndNormalizeChain()` is cheap: single pass over the chain
- Only called for non-trivial chains (size ≥ 2)
- When normalization disabled, function returns immediately
- When no mixed types, function returns after quick check

**Benefit**:
- Correctness: All stages now use consistent element types
- Better splitting decisions: Chains aren't incorrectly split or kept together
- More vectorization opportunities: Proper element size means better contiguity/alignment checks

### Alternative Considered

**Option A**: Thread normalized type through all function signatures
- **Pro**: Only analyze once
- **Con**: Invasive changes, complex refactoring

**Option B** (Implemented): Each function analyzes independently
- **Pro**: Simple, self-contained, no API changes
- **Con**: Slight redundant computation (but negligible)

## Debug Output

When running with `-debug-only=load-store-vectorizer`, you'll now see:

```
LSV: splitChainByContiguity considering chain:
  load i32, ptr %base
  load <2 x i16>, ptr %ptr1
LSV:   Using normalized type: i16

LSV: splitChainByAlignment considering chain:
  load i32, ptr %base
  load <2 x i16>, ptr %ptr1
LSV:   Using normalized type: i16
```

This helps verify that all stages are using the same normalized type.

## Testing

### Existing Tests
All existing tests should pass unchanged because:
1. When normalization is disabled (default), behavior is identical
2. When enabled, the fix makes decisions more consistent (improvements only)

### New Test
`consistent-normalization.ll` specifically tests:
- Debug output consistency across all split stages
- Various type combinations (i32+<2xi16>, <4xi8>+i32, etc.)
- Uniform types (should not normalize)

## Verification

Run with debug output to verify consistency:
```bash
opt -S -passes=load-store-vectorizer -load-store-vectorizer-normalize-types \
    -debug-only=load-store-vectorizer < test.ll 2>&1 | grep "normalized type"
```

Should show the same normalized type being used in all split stages for mixed-type chains.

## Files Modified

1. `llvm/lib/Transforms/Vectorize/LoadStoreVectorizer.cpp`:
   - `splitChainByContiguity()`: Added normalization analysis (~8 lines)
   - `splitChainByAlignment()`: Added normalization analysis (~8 lines)
   
2. `llvm/test/Transforms/LoadStoreVectorizer/X86/consistent-normalization.ll`:
   - New test file (~140 lines)

## Impact

- **Lines changed**: ~30 lines
- **Lines added**: ~150 lines (mostly tests)
- **Risk**: Very low - changes are guarded by feature flag
- **Performance**: Negligible - small duplicate analysis overhead
- **Correctness**: High improvement - consistent type usage across pipeline

