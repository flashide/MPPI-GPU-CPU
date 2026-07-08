// XOROSHIRO128+ pseudo-random number generator for OpenCL.
// Replaces RNG in the GPU implementation.
//
// XOROSHIRO128+ is chosen for:
//   - low register footprint (2 × uint64 state = 16 bytes per thread)
//   - excellent statistical quality (passes BigCrush)
//   - no lookup tables (good for Mali-G610 Valhall which has limited
//     shared/constant cache but generous register file)
//
// Reference: Blackman & Vigna, "Scrambled Linear Pseudorandom Number
// Generators", ACM TOMS 47(4), 2021.

#ifndef MPPI_RNG_CL
#define MPPI_RNG_CL

// ---- state type -----------------------------------------------------------
typedef struct {
  ulong s[2];
} rng_state_t;

// ---- helpers --------------------------------------------------------------

inline ulong rng_rotl(ulong x, int k)
{
  return (x << k) | (x >> (64 - k));
}

// ---- core generator -------------------------------------------------------

// Return a uniform uint64 and advance state.
inline ulong rng_next(rng_state_t * st)
{
  const ulong s0 = st->s[0];
  ulong s1 = st->s[1];
  const ulong result = s0 + s1;
  s1 ^= s0;
  st->s[0] = rng_rotl(s0, 24) ^ s1 ^ (s1 << 16);
  st->s[1] = rng_rotl(s1, 37);
  return result;
}

// ---- seeding kernel -------------------------------------------------------

// Initialise RNG state from a 64-bit seed and the global thread id.
// Splittable: each thread gets a deterministic independent stream.
__kernel void init_rng(__global rng_state_t * states,
                       ulong seed,
                       int n)
{
  int i = get_global_id(0);
  if (i >= n) return;

  // SplitMix64 seeding to decorrelate nearby seeds
  ulong z = seed + i;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
  z = z ^ (z >> 31);
  states[i].s[0] = z;

  z = z + 0x9e3779b97f4a7c15UL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
  z = z ^ (z >> 31);
  states[i].s[1] = z;
}

// ---- standard normal (float32) --------------------------------------------

// Box-Muller transform using the Marsaglia polar method.
// Returns a single-precision standard normal variate N(0, 1).
//
// Average 2.27 × rng_next() calls per output (rejection sampling).
// On Mali-G610, native_sqrt / native_log provide hardware-accelerated
// approximations with sufficient precision for MPPI noise.
inline float rng_normal(rng_state_t * st)
{
  float u1, u2, s;
  do {
    // Convert uint64 → uniform [−1, 1) float32
    u1 = (float)(rng_next(st) >> 11) * 0x1.0p-53f * 2.0f - 1.0f;
    u2 = (float)(rng_next(st) >> 11) * 0x1.0p-53f * 2.0f - 1.0f;
    s = u1 * u1 + u2 * u2;
  } while (s >= 1.0f || s == 0.0f);

  // Mali-G610: native_sqrt / native_log are IEEE-754 conformant for
  // normal-range inputs (G610 Valhall arch).
  return u1 * native_sqrt(-2.0f * native_log(s) / s);
}

#endif  // MPPI_RNG_CL
