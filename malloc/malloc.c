
/* Malloc implementation for multiple threads without lock contention.
   Copyright (C) 1996-2024 Free Software Foundation, Inc.
   Copyright The GNU Toolchain Authors.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

   /*
     This is a version (aka ptmalloc2) of malloc/free/realloc written by
     Doug Lea and adapted to multiple threads/arenas by Wolfram Gloger.

     There have been substantial changes made after the integration into
     glibc in all parts of the code.  Do not look for much commonality
     with the ptmalloc2 version.

   * Version ptmalloc2-20011215
     based on:
     VERSION 2.7.0 Sun Mar 11 14:14:06 2001  Doug Lea  (dl at gee)

   * Quickstart

     In order to compile this implementation, a Makefile is provided with
     the ptmalloc2 distribution, which has pre-defined targets for some
     popular systems (e.g. "make posix" for Posix threads).  All that is
     typically required with regard to compiler flags is the selection of
     the thread package via defining one out of USE_PTHREADS, USE_THR or
     USE_SPROC.  Check the thread-m.h file for what effects this has.
     Many/most systems will additionally require USE_TSD_DATA_HACK to be
     defined, so this is the default for "make posix".

   * Why use this malloc?

     This is not the fastest, most space-conserving, most portable, or
     most tunable malloc ever written. However it is among the fastest
     while also being among the most space-conserving, portable and tunable.
     Consistent balance across these factors results in a good general-purpose
     allocator for malloc-intensive programs.

     The main properties of the algorithms are:
     * For large (>= 512 bytes) requests, it is a pure best-fit allocator,
       with ties normally decided via FIFO (i.e. least recently used).
     * For small (<= 64 bytes by default) requests, it is a caching
       allocator, that maintains pools of quickly recycled chunks.
     * In between, and for combinations of large and small requests, it does
       the best it can trying to meet both goals at once.
     * For very large requests (>= 128KB by default), it relies on system
       memory mapping facilities, if supported.

     For a longer but slightly out of date high-level description, see
        http://gee.cs.oswego.edu/dl/html/malloc.html

     You may already by default be using a C library containing a malloc
     that is  based on some version of this malloc (for example in
     linux). You might still want to use the one in this file in order to
     customize settings or to avoid overheads associated with library
     versions.

   * Contents, described in more detail in "description of public routines" below.

     Standard (ANSI/SVID/...)  functions:
       malloc(size_t n);
       calloc(size_t n_elements, size_t element_size);
       free(void* p);
       realloc(void* p, size_t n);
       memalign(size_t alignment, size_t n);
       valloc(size_t n);
       mallinfo()
       mallopt(int parameter_number, int parameter_value)

     Additional functions:
       independent_calloc(size_t n_elements, size_t size, void* chunks[]);
       independent_comalloc(size_t n_elements, size_t sizes[], void* chunks[]);
       pvalloc(size_t n);
       malloc_trim(size_t pad);
       malloc_usable_size(void* p);
       malloc_stats();

   * Vital statistics:

     Supported pointer representation:       4 or 8 bytes
     Supported size_t  representation:       4 or 8 bytes
          Note that size_t is allowed to be 4 bytes even if pointers are 8.
          You can adjust this by defining INTERNAL_SIZE_T

     Alignment:                              2 * sizeof(size_t) (default)
          (i.e., 8 byte alignment with 4byte size_t). This suffices for
          nearly all current machines and C compilers. However, you can
          define MALLOC_ALIGNMENT to be wider than this if necessary.

     Minimum overhead per allocated chunk:   4 or 8 bytes
          Each malloced chunk has a hidden word of overhead holding size
          and status information.

     Minimum allocated size: 4-byte ptrs:  16 bytes    (including 4 overhead)
                 8-byte ptrs:  24/32 bytes (including, 4/8 overhead)

          When a chunk is freed, 12 (for 4byte ptrs) or 20 (for 8 byte
          ptrs but 4 byte size) or 24 (for 8/8) additional bytes are
          needed; 4 (8) for a trailing size field and 8 (16) bytes for
          free list pointers. Thus, the minimum allocatable size is
          16/24/32 bytes.

          Even a request for zero bytes (i.e., malloc(0)) returns a
          pointer to something of the minimum allocatable size.

          The maximum overhead wastage (i.e., number of extra bytes
          allocated than were requested in malloc) is less than or equal
          to the minimum size, except for requests >= mmap_threshold that
          are serviced via mmap(), where the worst case wastage is 2 *
          sizeof(size_t) bytes plus the remainder from a system page (the
          minimal mmap unit); typically 4096 or 8192 bytes.

     Maximum allocated size:  4-byte size_t: 2^32 minus about two pages
                  8-byte size_t: 2^64 minus about two pages

          It is assumed that (possibly signed) size_t values suffice to
          represent chunk sizes. `Possibly signed' is due to the fact
          that `size_t' may be defined on a system as either a signed or
          an unsigned type. The ISO C standard says that it must be
          unsigned, but a few systems are known not to adhere to this.
          Additionally, even when size_t is unsigned, sbrk (which is by
          default used to obtain memory from system) accepts signed
          arguments, and may not be able to handle size_t-wide arguments
          with negative sign bit.  Generally, values that would
          appear as negative after accounting for overhead and alignment
          are supported only via mmap(), which does not have this
          limitation.

          Requests for sizes outside the allowed range will perform an optional
          failure action and then return null. (Requests may also
          also fail because a system is out of memory.)

     Thread-safety: thread-safe

     Compliance: I believe it is compliant with the 1997 Single Unix Specification
          Also SVID/XPG, ANSI C, and probably others as well.

   * Synopsis of compile-time options:

       People have reported using previous versions of this malloc on all
       versions of Unix, sometimes by tweaking some of the defines
       below. It has been tested most extensively on Solaris and Linux.
       People also report using it in stand-alone embedded systems.

       The implementation is in straight, hand-tuned ANSI C.  It is not
       at all modular. (Sorry!)  It uses a lot of macros.  To be at all
       usable, this code should be compiled using an optimizing compiler
       (for example gcc -O3) that can simplify expressions and control
       paths. (FAQ: some macros import variables as arguments rather than
       declare locals because people reported that some debuggers
       otherwise get confused.)

       OPTION                     DEFAULT VALUE

       Compilation Environment options:

       HAVE_MREMAP                0

       Changing default word sizes:

       INTERNAL_SIZE_T            size_t

       Configuration and functionality options:

       USE_PUBLIC_MALLOC_WRAPPERS NOT defined
       USE_MALLOC_LOCK            NOT defined
       MALLOC_DEBUG               NOT defined
       REALLOC_ZERO_BYTES_FREES   1
       TRIM_FASTBINS              0

       Options for customizing MORECORE:

       MORECORE                   sbrk
       MORECORE_FAILURE           -1
       MORECORE_CONTIGUOUS        1
       MORECORE_CANNOT_TRIM       NOT defined
       MORECORE_CLEARS            1
       MMAP_AS_MORECORE_SIZE      (1024 * 1024)

       Tuning options that are also dynamically changeable via mallopt:

       DEFAULT_MXFAST             64 (for 32bit), 128 (for 64bit)
       DEFAULT_TRIM_THRESHOLD     128 * 1024
       DEFAULT_TOP_PAD            0
       DEFAULT_MMAP_THRESHOLD     128 * 1024
       DEFAULT_MMAP_MAX           65536

       There are several other #defined constants and macros that you
       probably don't want to touch unless you are extending or adapting malloc.
   */

   /*
     void* is the pointer type that malloc should say it returns
   */

#ifndef void
#  define void void
#endif /*void*/

#include <stddef.h> /* for size_t */
#include <stdlib.h> /* for getenv(), abort() */
#include <unistd.h> /* for __libc_enable_secure */

#include <atomic.h>
#include <_itoa.h>
#include <bits/wordsize.h>
#include <sys/sysinfo.h>

#include <ldsodefs.h>
#include <setvmaname.h>

#include <unistd.h>
#include <stdio.h> /* needed for malloc_stats */
#include <errno.h>
#include <assert.h>

#include <shlib-compat.h>

   /* For uintptr_t.  */
#include <stdint.h>

/* For va_arg, va_start, va_end.  */
#include <stdarg.h>

/* For MIN, MAX, powerof2.  */
#include <sys/param.h>

/* For ALIGN_UP et. al.  */
#include <libc-pointer-arith.h>

/* For DIAG_PUSH/POP_NEEDS_COMMENT et al.  */
#include <libc-diag.h>

/* For memory tagging.  */
#include <libc-mtag.h>

#include <malloc/malloc-internal.h>

/* For SINGLE_THREAD_P.  */
#include <sysdep-cancel.h>

#include <libc-internal.h>

/* For tcache double-free check.  */
#include <random-bits.h>
#include <sys/random.h>
#include <not-cancel.h>

/*
  Debugging:

  Because freed chunks may be overwritten with bookkeeping fields, this
  malloc will often die when freed memory is overwritten by user
  programs.  This can be very effective (albeit in an annoying way)
  in helping track down dangling pointers.

  If you compile with -DMALLOC_DEBUG, a number of assertion checks are
  enabled that will catch more memory errors. You probably won't be
  able to make much sense of the actual assertion errors, but they
  should help you locate incorrectly overwritten memory.  The checking
  is fairly extensive, and will slow down execution
  noticeably. Calling malloc_stats or mallinfo with MALLOC_DEBUG set
  will attempt to check every non-mmapped allocated and free chunk in
  the course of computing the summaries. (By nature, mmapped regions
  cannot be checked very much automatically.)

  Setting MALLOC_DEBUG may also be helpful if you are trying to modify
  this code. The assertions in the check routines spell out in more
  detail the assumptions and invariants underlying the algorithms.

  Setting MALLOC_DEBUG does NOT provide an automated mechanism for
  checking that all accesses to malloced memory stay within their
  bounds. However, there are several add-ons and adaptations of this
  or other mallocs available that do this.
*/

#ifndef MALLOC_DEBUG
#  define MALLOC_DEBUG 0
#endif

#if USE_TCACHE
/* We want 64 entries.  This is an arbitrary limit, which tunables can reduce.
 */
#  define TCACHE_MAX_BINS 64
#  define MAX_TCACHE_SIZE tidx2usize (TCACHE_MAX_BINS - 1)

 /* Only used to pre-fill the tunables.  */
#  define tidx2usize(idx)                                                     \
    (((size_t) idx) * MALLOC_ALIGNMENT + MINSIZE - SIZE_SZ)

/* When "x" is from chunksize().  */
#  define csize2tidx(x)                                                       \
    (((x) - MINSIZE + MALLOC_ALIGNMENT - 1) / MALLOC_ALIGNMENT)
/* When "x" is a user-provided size.  */
#  define usize2tidx(x) csize2tidx (request2size (x))

/* With rounding and alignment, the bins are...
   idx 0   bytes 0..24 (64-bit) or 0..12 (32-bit)
   idx 1   bytes 25..40 or 13..20
   idx 2   bytes 41..56 or 21..28
   etc.  */

   /* This is another arbitrary limit, which tunables can change.  Each
      tcache bin will hold at most this number of chunks.  */
#  define TCACHE_FILL_COUNT 7

      /* Maximum chunks in tcache bins for tunables.  This value must fit the range
         of tcache->counts[] entries, else they may overflow.  */
#  define MAX_TCACHE_COUNT UINT16_MAX
#endif

         /* Safe-Linking:
            Use randomness from ASLR (mmap_base) to protect single-linked lists
            of Fast-Bins and TCache.  That is, mask the "next" pointers of the
            lists' chunks, and also perform allocation alignment checks on them.
            This mechanism reduces the risk of pointer hijacking, as was done with
            Safe-Unlinking in the double-linked lists of Small-Bins.
            It assumes a minimum page size of 4096 bytes (12 bits).  Systems with
            larger pages provide less entropy, although the pointer mangling
            still works.  */
#define PROTECT_PTR(pos, ptr)                                                 \
  ((__typeof (ptr)) ((((size_t) pos) >> 12) ^ ((size_t) ptr)))
#define REVEAL_PTR(ptr) PROTECT_PTR (&ptr, ptr)

            /*
              The REALLOC_ZERO_BYTES_FREES macro controls the behavior of realloc (p, 0)
              when p is nonnull.  If the macro is nonzero, the realloc call returns NULL;
              otherwise, the call returns what malloc (0) would.  In either case,
              p is freed.  Glibc uses a nonzero REALLOC_ZERO_BYTES_FREES, which
              implements common historical practice.

              ISO C17 says the realloc call has implementation-defined behavior,
              and it might not even free p.
            */

#ifndef REALLOC_ZERO_BYTES_FREES
#  define REALLOC_ZERO_BYTES_FREES 1
#endif

            /*
              TRIM_FASTBINS controls whether free() of a very small chunk can
              immediately lead to trimming. Setting to true (1) can reduce memory
              footprint, but will almost always slow down programs that use a lot
              of small chunks.

              Define this only if you are willing to give up some speed to more
              aggressively reduce system-level memory footprint when releasing
              memory in programs that use many small chunks.  You can get
              essentially the same effect by setting MXFAST to 0, but this can
              lead to even greater slowdowns in programs using many small chunks.
              TRIM_FASTBINS is an in-between compile-time option, that disables
              only those chunks bordering topmost memory from being placed in
              fastbins.
            */

#ifndef TRIM_FASTBINS
#  define TRIM_FASTBINS 0
#endif

            /* Definition for getting more memory from the OS.  */
#include "morecore.c"

#define MORECORE (*__glibc_morecore)
#define MORECORE_FAILURE 0

/* Memory tagging.  */

/* Some systems support the concept of tagging (sometimes known as
   coloring) memory locations on a fine grained basis.  Each memory
   location is given a color (normally allocated randomly) and
   pointers are also colored.  When the pointer is dereferenced, the
   pointer's color is checked against the memory's color and if they
   differ the access is faulted (sometimes lazily).

   We use this in glibc by maintaining a single color for the malloc
   data structures that are interleaved with the user data and then
   assigning separate colors for each block allocation handed out.  In
   this way simple buffer overruns will be rapidly detected.  When
   memory is freed, the memory is recolored back to the glibc default
   so that simple use-after-free errors can also be detected.

   If memory is reallocated the buffer is recolored even if the
   address remains the same.  This has a performance impact, but
   guarantees that the old pointer cannot mistakenly be reused (code
   that compares old against new will see a mismatch and will then
   need to behave as though realloc moved the data to a new location).

   Internal API for memory tagging support.

   The aim is to keep the code for memory tagging support as close to
   the normal APIs in glibc as possible, so that if tagging is not
   enabled in the library, or is disabled at runtime then standard
   operations can continue to be used.  Support macros are used to do
   this:

   void *tag_new_zero_region (void *ptr, size_t size)

   Allocates a new tag, colors the memory with that tag, zeros the
   memory and returns a pointer that is correctly colored for that
   location.  The non-tagging version will simply call memset with 0.

   void *tag_region (void *ptr, size_t size)

   Color the region of memory pointed to by PTR and size SIZE with
   the color of PTR.  Returns the original pointer.

   void *tag_new_usable (void *ptr)

   Allocate a new random color and use it to color the user region of
   a chunk; this may include data from the subsequent chunk's header
   if tagging is sufficiently fine grained.  Returns PTR suitably
   recolored for accessing the memory there.

   void *tag_at (void *ptr)

   Read the current color of the memory at the address pointed to by
   PTR (ignoring it's current color) and return PTR recolored to that
   color.  PTR must be valid address in all other respects.  When
   tagging is not enabled, it simply returns the original pointer.
*/

#ifdef USE_MTAG
static bool mtag_enabled = false;
static int mtag_mmap_flags = 0;
#else
#  define mtag_enabled false
#  define mtag_mmap_flags 0
#endif

static __always_inline void*
tag_region(void* ptr, size_t size)
{
    if (__glibc_unlikely(mtag_enabled))
        return __libc_mtag_tag_region(ptr, size);
    return ptr;
}

static __always_inline void*
tag_new_zero_region(void* ptr, size_t size)
{
    if (__glibc_unlikely(mtag_enabled))
        return __libc_mtag_tag_zero_region(__libc_mtag_new_tag(ptr), size);
    return memset(ptr, 0, size);
}

/* Defined later.  */
static void* tag_new_usable(void* ptr);

static __always_inline void*
tag_at(void* ptr)
{
    if (__glibc_unlikely(mtag_enabled))
        return __libc_mtag_address_get_tag(ptr);
    return ptr;
}

#include <string.h>

/*
  MORECORE-related declarations. By default, rely on sbrk
*/

/*
  MORECORE is the name of the routine to call to obtain more memory
  from the system.  See below for general guidance on writing
  alternative MORECORE functions, as well as a version for WIN32 and a
  sample version for pre-OSX macos.
*/

#ifndef MORECORE
#  define MORECORE sbrk
#endif

/*
  MORECORE_FAILURE is the value returned upon failure of MORECORE
  as well as mmap. Since it cannot be an otherwise valid memory address,
  and must reflect values of standard sys calls, you probably ought not
  try to redefine it.
*/

#ifndef MORECORE_FAILURE
#  define MORECORE_FAILURE (-1)
#endif

/*
  If MORECORE_CONTIGUOUS is true, take advantage of fact that
  consecutive calls to MORECORE with positive arguments always return
  contiguous increasing addresses.  This is true of unix sbrk.  Even
  if not defined, when regions happen to be contiguous, malloc will
  permit allocations spanning regions obtained from different
  calls. But defining this when applicable enables some stronger
  consistency checks and space efficiencies.
*/

#ifndef MORECORE_CONTIGUOUS
#  define MORECORE_CONTIGUOUS 1
#endif

/*
  Define MORECORE_CANNOT_TRIM if your version of MORECORE
  cannot release space back to the system when given negative
  arguments. This is generally necessary only if you are using
  a hand-crafted MORECORE function that cannot handle negative arguments.
*/

/* #define MORECORE_CANNOT_TRIM */

/*  MORECORE_CLEARS           (default 1)
     The degree to which the routine mapped to MORECORE zeroes out
     memory: never (0), only for newly allocated space (1) or always
     (2).  The distinction between (1) and (2) is necessary because on
     some systems, if the application first decrements and then
     increments the break value, the contents of the reallocated space
     are unspecified.
 */

#ifndef MORECORE_CLEARS
#  define MORECORE_CLEARS 1
#endif

 /*
    MMAP_AS_MORECORE_SIZE is the minimum mmap size argument to use if
    sbrk fails, and mmap is used as a backup.  The value must be a
    multiple of page size.  This backup strategy generally applies only
    when systems have "holes" in address space, so sbrk cannot perform
    contiguous expansion, but there is still space available on system.
    On systems for which this is known to be useful (i.e. most linux
    kernels), this occurs only when programs allocate huge amounts of
    memory.  Between this, and the fact that mmap regions tend to be
    limited, the size should be large, to avoid too many mmap calls and
    thus avoid running out of kernel resources.  */

#ifndef MMAP_AS_MORECORE_SIZE
#  define MMAP_AS_MORECORE_SIZE (1024 * 1024)
#endif

    /*
      Define HAVE_MREMAP to make realloc() use mremap() to re-allocate
      large blocks.
    */

#ifndef HAVE_MREMAP
#  define HAVE_MREMAP 0
#endif

    /*
      This version of malloc supports the standard SVID/XPG mallinfo
      routine that returns a struct containing usage properties and
      statistics. It should work on any SVID/XPG compliant system that has
      a /usr/include/malloc.h defining struct mallinfo. (If you'd like to
      install such a thing yourself, cut out the preliminary declarations
      as described above and below and save them in a malloc.h file. But
      there's no compelling reason to bother to do this.)

      The main declaration needed is the mallinfo struct that is returned
      (by-copy) by mallinfo().  The SVID/XPG malloinfo struct contains a
      bunch of fields that are not even meaningful in this version of
      malloc.  These fields are are instead filled by mallinfo() with
      other numbers that might be of interest.
    */

    /* ---------- description of public routines ------------ */

#if IS_IN(libc)
/*
  malloc(size_t n)
  Returns a pointer to a newly allocated chunk of at least n bytes, or null
  if no space is available. Additionally, on failure, errno is
  set to ENOMEM on ANSI C systems.

  If n is zero, malloc returns a minimum-sized chunk. (The minimum
  size is 16 bytes on most 32bit systems, and 24 or 32 bytes on 64bit
  systems.)  On most systems, size_t is an unsigned type, so calls
  with negative arguments are interpreted as requests for huge amounts
  of space, which will often fail. The maximum supported value of n
  differs across systems, but is in all cases less than the maximum
  representable value of a size_t.
*/
void* __libc_malloc(size_t);
libc_hidden_proto(__libc_malloc)

/*
  free(void* p)
  Releases the chunk of memory pointed to by p, that had been previously
  allocated using malloc or a related routine such as realloc.
  It has no effect if p is null. It can have arbitrary (i.e., bad!)
  effects if p has already been freed.

  Unless disabled (using mallopt), freeing very large spaces will
  when possible, automatically trigger operations that give
  back unused memory to the system, thus reducing program footprint.
*/
void __libc_free(void*);
libc_hidden_proto(__libc_free)

/*
  calloc(size_t n_elements, size_t element_size);
  Returns a pointer to n_elements * element_size bytes, with all locations
  set to zero.
*/
void* __libc_calloc(size_t, size_t);

/*
  realloc(void* p, size_t n)
  Returns a pointer to a chunk of size n that contains the same data
  as does chunk p up to the minimum of (n, p's size) bytes, or null
  if no space is available.

  The returned pointer may or may not be the same as p. The algorithm
  prefers extending p when possible, otherwise it employs the
  equivalent of a malloc-copy-free sequence.

  If p is null, realloc is equivalent to malloc.

  If space is not available, realloc returns null, errno is set (if on
  ANSI) and p is NOT freed.

  if n is for fewer bytes than already held by p, the newly unused
  space is lopped off and freed if possible.  Unless the #define
  REALLOC_ZERO_BYTES_FREES is set, realloc with a size argument of
  zero (re)allocates a minimum-sized chunk.

  Large chunks that were internally obtained via mmap will always be
  grown using malloc-copy-free sequences unless the system supports
  MREMAP (currently only linux).

  The old unix realloc convention of allowing the last-free'd chunk
  to be used as an argument to realloc is not supported.
*/
void* __libc_realloc(void*, size_t);
libc_hidden_proto(__libc_realloc)

/*
  memalign(size_t alignment, size_t n);
  Returns a pointer to a newly allocated chunk of n bytes, aligned
  in accord with the alignment argument.

  The alignment argument should be a power of two. If the argument is
  not a power of two, the nearest greater power is used.
  8-byte alignment is guaranteed by normal malloc calls, so don't
  bother calling memalign with an argument of 8 or less.

  Overreliance on memalign is a sure way to fragment space.
*/
void* __libc_memalign(size_t, size_t);
libc_hidden_proto(__libc_memalign)

/*
  valloc(size_t n);
  Equivalent to memalign(pagesize, n), where pagesize is the page
  size of the system. If the pagesize is unknown, 4096 is used.
*/
void* __libc_valloc(size_t);

/*
  mallinfo()
  Returns (by copy) a struct containing various summary statistics:

  arena:     current total non-mmapped bytes allocated from system
  ordblks:   the number of free chunks
  smblks:    the number of fastbin blocks (i.e., small chunks that
           have been freed but not reused or consolidated)
  hblks:     current number of mmapped regions
  hblkhd:    total bytes held in mmapped regions
  usmblks:   always 0
  fsmblks:   total bytes held in fastbin blocks
  uordblks:  current total allocated space (normal or mmapped)
  fordblks:  total free space
  keepcost:  the maximum number of bytes that could ideally be released
           back to system via malloc_trim. ("ideally" means that
           it ignores page restrictions etc.)

  Because these fields are ints, but internal bookkeeping may
  be kept as longs, the reported values may wrap around zero and
  thus be inaccurate.
*/
struct mallinfo2 __libc_mallinfo2(void);
libc_hidden_proto(__libc_mallinfo2)

struct mallinfo __libc_mallinfo(void);

/*
  pvalloc(size_t n);
  Equivalent to valloc(minimum-page-that-holds(n)), that is,
  round up n to nearest pagesize.
 */
void* __libc_pvalloc(size_t);

/*
  malloc_trim(size_t pad);

  If possible, gives memory back to the system (via negative
  arguments to sbrk) if there is unused memory at the `high' end of
  the malloc pool. You can call this after freeing large blocks of
  memory to potentially reduce the system-level memory requirements
  of a program. However, it cannot guarantee to reduce memory. Under
  some allocation patterns, some large free blocks of memory will be
  locked between two used chunks, so they cannot be given back to
  the system.

  The `pad' argument to malloc_trim represents the amount of free
  trailing space to leave untrimmed. If this argument is zero,
  only the minimum amount of memory to maintain internal data
  structures will be left (one page or less). Non-zero arguments
  can be supplied to maintain enough trailing space to service
  future expected allocations without having to re-obtain memory
  from the system.

  Malloc_trim returns 1 if it actually released any memory, else 0.
  On systems that do not support "negative sbrks", it will always
  return 0.
*/
int __malloc_trim(size_t);

/*
  malloc_usable_size(void* p);

  Returns the number of bytes you can actually use in
  an allocated chunk, which may be more than you requested (although
  often not) due to alignment and minimum size constraints.
  You can use this many bytes without worrying about
  overwriting other allocated objects. This is not a particularly great
  programming practice. malloc_usable_size can be more useful in
  debugging and assertions, for example:

  p = malloc(n);
  assert(malloc_usable_size(p) >= 256);

*/
size_t __malloc_usable_size(void*);

/*
  malloc_stats();
  Prints on stderr the amount of space obtained from the system (both
  via sbrk and mmap), the maximum amount (which may be more than
  current if malloc_trim and/or munmap got called), and the current
  number of bytes allocated via malloc (or realloc, etc) but not yet
  freed. Note that this is the number of bytes allocated, not the
  number requested. It will be larger than the number requested
  because of alignment and bookkeeping overhead. Because it includes
  alignment wastage as being in use, this figure may be greater than
  zero even when no user-level chunks are allocated.

  The reported current and maximum system memory can be inaccurate if
  a program makes other calls to system memory allocation functions
  (normally sbrk) outside of malloc.

  malloc_stats prints only the most commonly interesting statistics.
  More information can be obtained by calling mallinfo.

*/
void __malloc_stats(void);

/*
  posix_memalign(void **memptr, size_t alignment, size_t size);

  POSIX wrapper like memalign(), checking for validity of size.
*/
int __posix_memalign(void**, size_t, size_t);
#endif /* IS_IN (libc) */

/*
  mallopt(int parameter_number, int parameter_value)
  Sets tunable parameters The format is to provide a
  (parameter-number, parameter-value) pair.  mallopt then sets the
  corresponding parameter to the argument value if it can (i.e., so
  long as the value is meaningful), and returns 1 if successful else
  0.  SVID/XPG/ANSI defines four standard param numbers for mallopt,
  normally defined in malloc.h.  Only one of these (M_MXFAST) is used
  in this malloc. The others (M_NLBLKS, M_GRAIN, M_KEEP) don't apply,
  so setting them has no effect. But this malloc also supports four
  other options in mallopt. See below for details.  Briefly, supported
  parameters are as follows (listed defaults are for "typical"
  configurations).

  Symbol            param #   default    allowed param values
  M_MXFAST          1         64         0-80  (0 disables fastbins)
  M_TRIM_THRESHOLD -1         128*1024   any   (-1U disables trimming)
  M_TOP_PAD        -2         0          any
  M_MMAP_THRESHOLD -3         128*1024   any   (or 0 if no MMAP support)
  M_MMAP_MAX       -4         65536      any   (0 disables use of mmap)
*/
int __libc_mallopt(int, int);
#if IS_IN(libc)
libc_hidden_proto(__libc_mallopt)
#endif

/* mallopt tuning options */

/*
  M_MXFAST is the maximum request size used for "fastbins", special bins
  that hold returned chunks without consolidating their spaces. This
  enables future requests for chunks of the same size to be handled
  very quickly, but can increase fragmentation, and thus increase the
  overall memory footprint of a program.

  This malloc manages fastbins very conservatively yet still
  efficiently, so fragmentation is rarely a problem for values less
  than or equal to the default.  The maximum supported value of MXFAST
  is 80. You wouldn't want it any higher than this anyway.  Fastbins
  are designed especially for use with many small structs, objects or
  strings -- the default handles structs/objects/arrays with sizes up
  to 8 4byte fields, or small strings representing words, tokens,
  etc. Using fastbins for larger objects normally worsens
  fragmentation without improving speed.

  M_MXFAST is set in REQUEST size units. It is internally used in
  chunksize units, which adds padding and alignment.  You can reduce
  M_MXFAST to 0 to disable all use of fastbins.  This causes the malloc
  algorithm to be a closer approximation of fifo-best-fit in all cases,
  not just for larger requests, but will generally cause it to be
  slower.
*/

/* M_MXFAST is a standard SVID/XPG tuning option, usually listed in malloc.h */
#ifndef M_MXFAST
#  define M_MXFAST 1
#endif

#ifndef DEFAULT_MXFAST
#  define DEFAULT_MXFAST (64 * SIZE_SZ / 4)
#endif

/*
  M_TRIM_THRESHOLD is the maximum amount of unused top-most memory
  to keep before releasing via malloc_trim in free().

  Automatic trimming is mainly useful in long-lived programs.
  Because trimming via sbrk can be slow on some systems, and can
  sometimes be wasteful (in cases where programs immediately
  afterward allocate more large chunks) the value should be high
  enough so that your overall system performance would improve by
  releasing this much memory.

  The trim threshold and the mmap control parameters (see below)
  can be traded off with one another. Trimming and mmapping are
  two different ways of releasing unused memory back to the
  system. Between these two, it is often possible to keep
  system-level demands of a long-lived program down to a bare
  minimum. For example, in one test suite of sessions measuring
  the XF86 X server on Linux, using a trim threshold of 128K and a
  mmap threshold of 192K led to near-minimal long term resource
  consumption.

  If you are using this malloc in a long-lived program, it should
  pay to experiment with these values.  As a rough guide, you
  might set to a value close to the average size of a process
  (program) running on your system.  Releasing this much memory
  would allow such a process to run in memory.  Generally, it's
  worth it to tune for trimming rather tham memory mapping when a
  program undergoes phases where several large chunks are
  allocated and released in ways that can reuse each other's
  storage, perhaps mixed with phases where there are no such
  chunks at all.  And in well-behaved long-lived programs,
  controlling release of large blocks via trimming versus mapping
  is usually faster.

  However, in most programs, these parameters serve mainly as
  protection against the system-level effects of carrying around
  massive amounts of unneeded memory. Since frequent calls to
  sbrk, mmap, and munmap otherwise degrade performance, the default
  parameters are set to relatively high values that serve only as
  safeguards.

  The trim value It must be greater than page size to have any useful
  effect.  To disable trimming completely, you can set to
  (unsigned long)(-1)

  Trim settings interact with fastbin (MXFAST) settings: Unless
  TRIM_FASTBINS is defined, automatic trimming never takes place upon
  freeing a chunk with size less than or equal to MXFAST. Trimming is
  instead delayed until subsequent freeing of larger chunks. However,
  you can still force an attempted trim by calling malloc_trim.

  Also, trimming is not generally possible in cases where
  the main arena is obtained via mmap.

  Note that the trick some people use of mallocing a huge space and
  then freeing it at program startup, in an attempt to reserve system
  memory, doesn't have the intended effect under automatic trimming,
  since that memory will immediately be returned to the system.
*/

#define M_TRIM_THRESHOLD -1

#ifndef DEFAULT_TRIM_THRESHOLD
#  define DEFAULT_TRIM_THRESHOLD (128 * 1024)
#endif

/*
  M_TOP_PAD is the amount of extra `padding' space to allocate or
  retain whenever sbrk is called. It is used in two ways internally:

  * When sbrk is called to extend the top of the arena to satisfy
  a new malloc request, this much padding is added to the sbrk
  request.

  * When malloc_trim is called automatically from free(),
  it is used as the `pad' argument.

  In both cases, the actual amount of padding is rounded
  so that the end of the arena is always a system page boundary.

  The main reason for using padding is to avoid calling sbrk so
  often. Having even a small pad greatly reduces the likelihood
  that nearly every malloc request during program start-up (or
  after trimming) will invoke sbrk, which needlessly wastes
  time.

  Automatic rounding-up to page-size units is normally sufficient
  to avoid measurable overhead, so the default is 0.  However, in
  systems where sbrk is relatively slow, it can pay to increase
  this value, at the expense of carrying around more memory than
  the program needs.
*/

#define M_TOP_PAD -2

#ifndef DEFAULT_TOP_PAD
#  define DEFAULT_TOP_PAD (0)
#endif

/*
  MMAP_THRESHOLD_MAX and _MIN are the bounds on the dynamically
  adjusted MMAP_THRESHOLD.
*/

#ifndef DEFAULT_MMAP_THRESHOLD_MIN
#  define DEFAULT_MMAP_THRESHOLD_MIN (128 * 1024)
#endif

#ifndef DEFAULT_MMAP_THRESHOLD_MAX
/* For 32-bit platforms we cannot increase the maximum mmap
   threshold much because it is also the minimum value for the
   maximum heap size and its alignment.  Going above 512k (i.e., 1M
   for new heaps) wastes too much address space.  */
#  if __WORDSIZE == 32
#    define DEFAULT_MMAP_THRESHOLD_MAX (512 * 1024)
#  else
#    define DEFAULT_MMAP_THRESHOLD_MAX (4 * 1024 * 1024 * sizeof (long))
#  endif
#endif

   /*
     M_MMAP_THRESHOLD is the request size threshold for using mmap()
     to service a request. Requests of at least this size that cannot
     be allocated using already-existing space will be serviced via mmap.
     (If enough normal freed space already exists it is used instead.)

     Using mmap segregates relatively large chunks of memory so that
     they can be individually obtained and released from the host
     system. A request serviced through mmap is never reused by any
     other request (at least not directly; the system may just so
     happen to remap successive requests to the same locations).

     Segregating space in this way has the benefits that:

      1. Mmapped space can ALWAYS be individually released back
         to the system, which helps keep the system level memory
         demands of a long-lived program low.
      2. Mapped memory can never become `locked' between
         other chunks, as can happen with normally allocated chunks, which
         means that even trimming via malloc_trim would not release them.
      3. On some systems with "holes" in address spaces, mmap can obtain
         memory that sbrk cannot.

     However, it has the disadvantages that:

      1. The space cannot be reclaimed, consolidated, and then
         used to service later requests, as happens with normal chunks.
      2. It can lead to more wastage because of mmap page alignment
         requirements
      3. It causes malloc performance to be more dependent on host
         system memory management support routines which may vary in
         implementation quality and may impose arbitrary
         limitations. Generally, servicing a request via normal
         malloc steps is faster than going through a system's mmap.

     The advantages of mmap nearly always outweigh disadvantages for
     "large" chunks, but the value of "large" varies across systems.  The
     default is an empirically derived value that works well in most
     systems.


     Update in 2006:
     The above was written in 2001. Since then the world has changed a lot.
     Memory got bigger. Applications got bigger. The virtual address space
     layout in 32 bit linux changed.

     In the new situation, brk() and mmap space is shared and there are no
     artificial limits on brk size imposed by the kernel. What is more,
     applications have started using transient allocations larger than the
     128Kb as was imagined in 2001.

     The price for mmap is also high now; each time glibc mmaps from the
     kernel, the kernel is forced to zero out the memory it gives to the
     application. Zeroing memory is expensive and eats a lot of cache and
     memory bandwidth. This has nothing to do with the efficiency of the
     virtual memory system, by doing mmap the kernel just has no choice but
     to zero.

     In 2001, the kernel had a maximum size for brk() which was about 800
     megabytes on 32 bit x86, at that point brk() would hit the first
     mmaped shared libraries and couldn't expand anymore. With current 2.6
     kernels, the VA space layout is different and brk() and mmap
     both can span the entire heap at will.

     Rather than using a static threshold for the brk/mmap tradeoff,
     we are now using a simple dynamic one. The goal is still to avoid
     fragmentation. The old goals we kept are
     1) try to get the long lived large allocations to use mmap()
     2) really large allocations should always use mmap()
     and we're adding now:
     3) transient allocations should use brk() to avoid forcing the kernel
        having to zero memory over and over again

     The implementation works with a sliding threshold, which is by default
     limited to go between 128Kb and 32Mb (64Mb for 64 bitmachines) and starts
     out at 128Kb as per the 2001 default.

     This allows us to satisfy requirement 1) under the assumption that long
     lived allocations are made early in the process' lifespan, before it has
     started doing dynamic allocations of the same size (which will
     increase the threshold).

     The upperbound on the threshold satisfies requirement 2)

     The threshold goes up in value when the application frees memory that was
     allocated with the mmap allocator. The idea is that once the application
     starts freeing memory of a certain size, it's highly probable that this is
     a size the application uses for transient allocations. This estimator
     is there to satisfy the new third requirement.

   */

#define M_MMAP_THRESHOLD -3

#ifndef DEFAULT_MMAP_THRESHOLD
#  define DEFAULT_MMAP_THRESHOLD DEFAULT_MMAP_THRESHOLD_MIN
#endif

   /*
     M_MMAP_MAX is the maximum number of requests to simultaneously
     service using mmap. This parameter exists because
     some systems have a limited number of internal tables for
     use by mmap, and using more than a few of them may degrade
     performance.

     The default is set to a value that serves only as a safeguard.
     Setting to 0 disables use of mmap for servicing large requests.
   */

#define M_MMAP_MAX -4

#ifndef DEFAULT_MMAP_MAX
#  define DEFAULT_MMAP_MAX (65536)
#endif

#include <malloc.h>

#ifndef RETURN_ADDRESS
#  define RETURN_ADDRESS(X_) (NULL)
#endif

   /* Forward declarations.  */
    struct malloc_chunk;
typedef struct malloc_chunk* mchunkptr;

/* Internal routines.  */

static void* _int_malloc(mstate, size_t);
static void _int_free(mstate, mchunkptr, int);
static void _int_free_merge_chunk(mstate, mchunkptr, INTERNAL_SIZE_T);
static INTERNAL_SIZE_T _int_free_create_chunk(mstate, mchunkptr,
    INTERNAL_SIZE_T, mchunkptr,
    INTERNAL_SIZE_T);
static void _int_free_maybe_consolidate(mstate, INTERNAL_SIZE_T);
static void* _int_realloc(mstate, mchunkptr, INTERNAL_SIZE_T,
    INTERNAL_SIZE_T);
static void* _int_memalign(mstate, size_t, size_t);
#if IS_IN(libc)
static void* _mid_memalign(size_t, size_t, void*);
#endif

static void malloc_printerr(const char* str) __attribute__((noreturn));

static void munmap_chunk(mchunkptr p);
#if HAVE_MREMAP
static mchunkptr mremap_chunk(mchunkptr p, size_t new_size);
#endif

static size_t musable(void* mem);

/* ------------------ MMAP support ------------------  */

#include <fcntl.h>
#include <sys/mman.h>

#if !defined(MAP_ANONYMOUS) && defined(MAP_ANON)
#  define MAP_ANONYMOUS MAP_ANON
#endif

#define MMAP(addr, size, prot, flags)                                         \
  __mmap ((addr), (size), (prot), (flags) | MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)

/*
  -----------------------  Chunk representations -----------------------
*/

/*
  This struct declaration is misleading (but accurate and necessary).
  It declares a "view" into memory allowing access to necessary
  fields at known offsets from a given base. See explanation below.
*/

struct malloc_chunk
{
    // 上一个块的大小
    INTERNAL_SIZE_T mchunk_prev_size; /* Size of previous chunk (if free).  */
    // 当前块的大小，包括开销，32位系统中，最后3没有被使用，64位系统中，最后4位没有被使用，最后3位分别代表：
    // NON_MAIN_ARENA：记录当前 chunk 是否不属于主线程。
    // IS_MAPPED：记录当前chunk是否是由mmap分配的。
    // PREV_INUSE：如果前面一个chunk处于分配状态，那么此位为1。一般来说，堆中第一个被分配的内存块的
    // size 字段的 P 位都会被设置为 1，以便于防止访问前面的非法内存。当一个chunk
    // 的size 的P位为0时，我们能通过 prev_size 字段来获取上一个 chunk
    // 的大小以及地址。这也方便进行空闲chunk之间的合并。

    INTERNAL_SIZE_T mchunk_size; /* Size in bytes, including overhead. */
    // 下一个
    struct malloc_chunk* fd; /* double links -- used only if free. */
    // 前一个
    struct malloc_chunk* bk;

    /* Only used for large blocks: pointer to next larger size.  */
    // 仅用于大块：指向下一个更大尺寸的指针
    struct malloc_chunk* fd_nextsize; /* double links -- used only if free. */
    struct malloc_chunk* bk_nextsize;
};

/*
   malloc_chunk details:

    (The following includes lightly edited explanations by Colin Plumb.)

    Chunks of memory are maintained using a `boundary tag' method as
    described in e.g., Knuth or Standish.  (See the paper by Paul
    Wilson ftp://ftp.cs.utexas.edu/pub/garbage/allocsrv.ps for a
    survey of such techniques.)  Sizes of free chunks are stored both
    in the front of each chunk and at the end.  This makes
    consolidating fragmented chunks into bigger chunks very fast.  The
    size fields also hold bits representing whether chunks are free or
    in use.

    An allocated chunk looks like this:


    chunk-> +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |             Size of previous chunk, if unallocated (P clear)  |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |             Size of chunk, in bytes                     |A|M|P|
      mem-> +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |             User data starts here...                          .
        .                                                               .
        .             (malloc_usable_size() bytes)                      .
        .                                                               |
nextchunk-> +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |             (size of chunk, but used for application data)    |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |             Size of next chunk, in bytes                |A|0|1|
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

    Where "chunk" is the front of the chunk for the purpose of most of
    the malloc code, but "mem" is the pointer that is returned to the
    user.  "Nextchunk" is the beginning of the next contiguous chunk.

    Chunks always begin on even word boundaries, so the mem portion
    (which is returned to the user) is also on an even word boundary, and
    thus at least double-word aligned.

    Free chunks are stored in circular doubly-linked lists, and look like this:

    chunk-> +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |             Size of previous chunk, if unallocated (P clear)  |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    `head:' |             Size of chunk, in bytes                     |A|0|P|
      mem-> +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |             Forward pointer to next chunk in list             |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |             Back pointer to previous chunk in list            |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |             Unused space (may be 0 bytes long)                .
        .                                                               .
        .                                                               |
nextchunk-> +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
    `foot:' |             Size of chunk, in bytes                           |
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
        |             Size of next chunk, in bytes                |A|0|0|
        +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

    The P (PREV_INUSE) bit, stored in the unused low-order bit of the
    chunk size (which is always a multiple of two words), is an in-use
    bit for the *previous* chunk.  If that bit is *clear*, then the
    word before the current chunk size contains the previous chunk
    size, and can be used to find the front of the previous chunk.
    The very first chunk allocated always has this bit set,
    preventing access to non-existent (or non-owned) memory. If
    prev_inuse is set for any given chunk, then you CANNOT determine
    the size of the previous chunk, and might even get a memory
    addressing fault when trying to do so.

    The A (NON_MAIN_ARENA) bit is cleared for chunks on the initial,
    main arena, described by the main_arena variable.  When additional
    threads are spawned, each thread receives its own arena (up to a
    configurable limit, after which arenas are reused for multiple
    threads), and the chunks in these arenas have the A bit set.  To
    find the arena for a chunk on such a non-main arena, heap_for_ptr
    performs a bit mask operation and indirection through the ar_ptr
    member of the per-heap header heap_info (see arena.c).

    Note that the `foot' of the current chunk is actually represented
    as the prev_size of the NEXT chunk. This makes it easier to
    deal with alignments etc but can be very confusing when trying
    to extend or adapt this code.

    The three exceptions to all this are:

     1. The special chunk `top' doesn't bother using the
    trailing size field since there is no next contiguous chunk
    that would have to index off it. After initialization, `top'
    is forced to always exist.  If it would become less than
    MINSIZE bytes long, it is replenished.

     2. Chunks allocated via mmap, which have the second-lowest-order
    bit M (IS_MMAPPED) set in their size fields.  Because they are
    allocated one-by-one, each must contain its own trailing size
    field.  If the M bit is set, the other bits are ignored
    (because mmapped chunks are neither in an arena, nor adjacent
    to a freed chunk).  The M bit is also used for chunks which
    originally came from a dumped heap via malloc_set_state in
    hooks.c.

     3. Chunks in fastbins are treated as allocated chunks from the
    point of view of the chunk allocator.  They are consolidated
    with their neighbors only in bulk, in malloc_consolidate.
*/

/*
  ---------- Size and alignment checks and conversions ----------
*/

/* Conversion from malloc headers to user pointers, and back.  When
   using memory tagging the user data and the malloc data structure
   headers have distinct tags.  Converting fully from one to the other
   involves extracting the tag at the other address and creating a
   suitable pointer using it.  That can be quite expensive.  There are
   cases when the pointers are not dereferenced (for example only used
   for alignment check) so the tags are not relevant, and there are
   cases when user data is not tagged distinctly from malloc headers
   (user data is untagged because tagging is done late in malloc and
   early in free).  User memory tagging across internal interfaces:

      sysmalloc: Returns untagged memory.
      _int_malloc: Returns untagged memory.
      _int_free: Takes untagged memory.
      _int_memalign: Returns untagged memory.
      _int_memalign: Returns untagged memory.
      _mid_memalign: Returns tagged memory.
      _int_realloc: Takes and returns tagged memory.
*/

/* The chunk header is two SIZE_SZ elements, but this is used widely, so
   we define it here for clarity later.
   块头是两个SIZE_SZ元素，但这被广泛使用，因此为了稍后清楚起见，我们在这里定义它。
   */
#define CHUNK_HDR_SZ (2 * SIZE_SZ)

   /* Convert a chunk address to a user mem pointer without correcting
      the tag.
      chunk的起始地址转换到用户内存mem地址。chunk起始地址在低地址，所以通过加上2*SIZE_SZ的方式，转换到高地址的mem地址指针
      */
#define chunk2mem(p) ((void *) ((char *) (p) + CHUNK_HDR_SZ))

      /* Convert a chunk address to a user mem pointer and extract the right tag.
       用户内存mem地址转换到chunk的起始地址。用户内存mem地址在高地址，所以通过减去2*SIZE_SZ的方式，转到低地址的chunk的起始地址
      */
#define chunk2mem_tag(p) ((void *) tag_at ((char *) (p) + CHUNK_HDR_SZ))

      /* Convert a user mem pointer to a chunk address and extract the right tag.
       用户内存mem地址转换到chunk的起始地址并提取正确的标签
      */
#define mem2chunk(mem) ((mchunkptr) tag_at (((char *) (mem) - CHUNK_HDR_SZ)))

      /* The smallest possible chunk */
      // 最小可能得块
#define MIN_CHUNK_SIZE (offsetof (struct malloc_chunk, fd_nextsize))

/* The smallest size we can malloc is an aligned minimal chunk */
// 内存对齐的最小可能块的大小
#define MINSIZE                                                               \
  (unsigned long) (((MIN_CHUNK_SIZE + MALLOC_ALIGN_MASK) & ~MALLOC_ALIGN_MASK))

/* Check if m has acceptable alignment
   检测m是否是正确的对齐
*/
#define aligned_OK(m) (((unsigned long) (m) & MALLOC_ALIGN_MASK) == 0)

#define misaligned_chunk(p)                                                   \
  ((uintptr_t) (MALLOC_ALIGNMENT == CHUNK_HDR_SZ ? (p) : chunk2mem (p))       \
   & MALLOC_ALIGN_MASK)

/* pad request bytes into a usable size -- internal version */
/* Note: This must be a macro that evaluates to a compile time constant
   if passed a literal constant.  */
   // 通过对齐后，实际chunk的大小。如果内存大小小于MINSIZE，则使用MINSIZE空间；否则通过MALLOC_ALIGN_MASK进行字节对齐。
   // request2size主要逻辑是最小的chunk大小为MINISIZE（64位系统32字节/32位系统16字节），
   // SIZE_SZ是一个字段的大小（64位8字节/32位4字节），MALLOC_ALIGN_MASK为对齐掩码（64位
   // 16-1=15；32位 8-1=7）
#define request2size(req)                                                     \
  (((req) + SIZE_SZ + MALLOC_ALIGN_MASK < MINSIZE)                            \
       ? MINSIZE                                                              \
       : ((req) + SIZE_SZ + MALLOC_ALIGN_MASK) & ~MALLOC_ALIGN_MASK)

/* Check if REQ overflows when padded and aligned and if the resulting
   value is less than PTRDIFF_T.  Returns the requested size or
   MINSIZE in case the value is less than MINSIZE, or 0 if any of the
   previous checks fail.  */
static inline size_t
checked_request2size(size_t req) __nonnull(1)
{
    if (__glibc_unlikely(req > PTRDIFF_MAX))
        return 0;

    /* When using tagged memory, we cannot share the end of the user
       block with the header for the next chunk, so ensure that we
       allocate blocks that are rounded up to the granule size.  Take
       care not to overflow from close to MAX_SIZE_T to a small
       number.  Ideally, this would be part of request2size(), but that
       must be a macro that produces a compile time constant if passed
       a constant literal.  */
    if (__glibc_unlikely(mtag_enabled))
    {
        /* Ensure this is not evaluated if !mtag_enabled, see gcc PR 99551.  */
        asm("");

        req = (req + (__MTAG_GRANULE_SIZE - 1))
            & ~(size_t)(__MTAG_GRANULE_SIZE - 1);
    }
    // 计算bytes数据需要分配的内存大小，
    return request2size(req);
}

/*
   --------------- Physical chunk operations ---------------
 */

 /* size field is or'ed with PREV_INUSE when previous adjacent chunk in use */
 // 当使用前一个相邻块时，大小字段与PREV_INUSE进行或运算
#define PREV_INUSE 0x1

/* extract inuse bit of previous chunk */
// 提取前一个相邻块的inuse位
#define prev_inuse(p) ((p)->mchunk_size & PREV_INUSE)

/* size field is or'ed with IS_MMAPPED if the chunk was obtained with mmap() */
// 如果块是通过mmap()获取时，大小字段与IS_MMAPPED进行或运算
#define IS_MMAPPED 0x2

/* check for mmap()'ed chunk */
// 是否是通过mmap()获取的块
#define chunk_is_mmapped(p) ((p)->mchunk_size & IS_MMAPPED)

/* size field is or'ed with NON_MAIN_ARENA if the chunk was obtained
   from a non-main arena.  This is only set immediately before handing
   the chunk to the user, if necessary.  */
   // 如果块是从非主arena获取时，size字段与NON_MAIN_ARENA进行或运算
#define NON_MAIN_ARENA 0x4

/* Check for chunk from main arena.  */
// 是否是从主arena获取的块
#define chunk_main_arena(p) (((p)->mchunk_size & NON_MAIN_ARENA) == 0)

/* Mark a chunk as not being on the main arena.  */
// 将块标记为不在主arena中
#define set_non_main_arena(p) ((p)->mchunk_size |= NON_MAIN_ARENA)

/*
   Bits to mask off when extracting size
   提取大小的掩码位
   Note: IS_MMAPPED is intentionally not masked off from size field in
   macros for which mmapped chunks should never be seen. This should
   cause helpful core dumps to occur if it is tried by accident by
   people extending or adapting this malloc.
 */
#define SIZE_BITS (PREV_INUSE | IS_MMAPPED | NON_MAIN_ARENA)

 /* Get size, ignoring use bits */
 // 获取大小，忽略使用位
#define chunksize(p) (chunksize_nomask (p) & ~(SIZE_BITS))

/* Like chunksize, but do not mask SIZE_BITS.  */
// 获取大小，不考虑掩码位
#define chunksize_nomask(p) ((p)->mchunk_size)

/* Ptr to next physical malloc_chunk. */
// 获取下一个malloc_chunk指针
#define next_chunk(p) ((mchunkptr) (((char *) (p)) + chunksize (p)))

/* Size of the chunk below P.  Only valid if !prev_inuse (P).  */
// 得到前一个块的大小，只有在前一个块未使用时才有效
#define prev_size(p) ((p)->mchunk_prev_size)

/* Set the size of the chunk below P.  Only valid if !prev_inuse (P).  */
// 设置前一个块的大小，只有在前一个块未使用时才有效
#define set_prev_size(p, sz) ((p)->mchunk_prev_size = (sz))

/* Ptr to previous physical malloc_chunk.  Only valid if !prev_inuse (P).  */
// 获取前一个块的指针，只有在前一个块未使用时才有效
#define prev_chunk(p) ((mchunkptr) (((char *) (p)) - prev_size (p)))

/* Treat space at ptr + offset as a chunk */
// 将ptr + offset处的地址作为块
#define chunk_at_offset(p, s) ((mchunkptr) (((char *) (p)) + (s)))

/* extract p's inuse bit */
// 提取当前块的inuse值，实际是查看下一个块的chunksize
#define inuse(p)                                                              \
  ((((mchunkptr) (((char *) (p)) + chunksize (p)))->mchunk_size) & PREV_INUSE)

/* set/clear chunk as being inuse without otherwise disturbing */
// 设置或者清除当前块的inuse值，实际是写入下一个块的chunksize中
#define set_inuse(p)                                                          \
  ((mchunkptr) (((char *) (p)) + chunksize (p)))->mchunk_size |= PREV_INUSE

#define clear_inuse(p)                                                        \
  ((mchunkptr) (((char *) (p)) + chunksize (p)))->mchunk_size &= ~(PREV_INUSE)

/* check/set/clear inuse bits in known places */
// 检查/设置/清除已知地址的inuse值
#define inuse_bit_at_offset(p, s)                                             \
  (((mchunkptr) (((char *) (p)) + (s)))->mchunk_size & PREV_INUSE)

#define set_inuse_bit_at_offset(p, s)                                         \
  (((mchunkptr) (((char *) (p)) + (s)))->mchunk_size |= PREV_INUSE)

#define clear_inuse_bit_at_offset(p, s)                                       \
  (((mchunkptr) (((char *) (p)) + (s)))->mchunk_size &= ~(PREV_INUSE))

/* Set size at head, without disturbing its use bit */
// 在头部设置大小，不影响其使用位
#define set_head_size(p, s)                                                   \
  ((p)->mchunk_size = (((p)->mchunk_size & SIZE_BITS) | (s)))

/* Set size/use field */
// 设置大小/使用位
#define set_head(p, s) ((p)->mchunk_size = (s))

/* Set size at footer (only when chunk is not in use) */
// 在下一个块上设置当前块的大小
#define set_foot(p, s)                                                        \
  (((mchunkptr) ((char *) (p) + (s)))->mchunk_prev_size = (s))

#pragma GCC poison mchunk_size
#pragma GCC poison mchunk_prev_size

/* This is the size of the real usable data in the chunk.  Not valid for
   dumped heap chunks.  */
#define memsize(p)                                                            \
  (__MTAG_GRANULE_SIZE > SIZE_SZ && __glibc_unlikely (mtag_enabled)           \
       ? chunksize (p) - CHUNK_HDR_SZ                                         \
       : chunksize (p) - CHUNK_HDR_SZ + (chunk_is_mmapped (p) ? 0 : SIZE_SZ))

   /* If memory tagging is enabled the layout changes to accommodate the granule
      size, this is wasteful for small allocations so not done by default.
      Both the chunk header and user data has to be granule aligned.  */
_Static_assert (__MTAG_GRANULE_SIZE <= CHUNK_HDR_SZ,
    "memory tagging is not supported with large granule.");

static __always_inline void*
tag_new_usable(void* ptr)
{
    if (__glibc_unlikely(mtag_enabled) && ptr)
    {
        mchunkptr cp = mem2chunk(ptr);
        ptr = __libc_mtag_tag_region(__libc_mtag_new_tag(ptr), memsize(cp));
    }
    return ptr;
}

/*
   -------------------- Internal data structures --------------------

   All internal state is held in an instance of malloc_state defined
   below. There are no other static variables, except in two optional
   cases:
 * If USE_MALLOC_LOCK is defined, the mALLOC_MUTEx declared above.
 * If mmap doesn't support MAP_ANONYMOUS, a dummy file descriptor
     for mmap.

   Beware of lots of tricks that minimize the total bookkeeping space
   requirements. The result is a little over 1K bytes (for 4byte
   pointers and size_t.)
 */

 /*
    Bins

     An array of bin headers for free chunks. Each bin is doubly
     linked.  The bins are approximately proportionally (log) spaced.
     There are a lot of these bins (128). This may look excessive, but
     works very well in practice.  Most bins hold sizes that are
     unusual as malloc request sizes, but are more usual for fragments
     and consolidated sets of chunks, which is what these bins hold, so
     they can be found quickly.  All procedures maintain the invariant
     that no consolidated chunk physically borders another one, so each
     chunk in a list is known to be preceded and followed by either
     inuse chunks or the ends of memory.

     Chunks in bins are kept in size order, with ties going to the
     approximately least recently used chunk. Ordering isn't needed
     for the small bins, which all contain the same-sized chunks, but
     facilitates best-fit allocation for larger chunks. These lists
     are just sequential. Keeping them in order almost never requires
     enough traversal to warrant using fancier ordered data
     structures.

     Chunks of the same size are linked with the most
     recently freed at the front, and allocations are taken from the
     back.  This results in LRU (FIFO) allocation order, which tends
     to give each chunk an equal opportunity to be consolidated with
     adjacent freed chunks, resulting in larger free chunks and less
     fragmentation.

     To simplify use in double-linked lists, each bin header acts
     as a malloc_chunk. This avoids special-casing for headers.
     But to conserve space and improve locality, we allocate
     only the fd/bk pointers of bins, and then use repositioning tricks
     to treat these as the fields of a malloc_chunk*.
  */

typedef struct malloc_chunk* mbinptr;

/* addressing -- note that bin_at(0) does not exist */
// 通过bin_at方法，找到bin对应位置
#define bin_at(m, i)                                                          \
  (mbinptr) (((char *) &((m)->bins[((i) - 1) * 2]))                           \
	     - offsetof (struct malloc_chunk, fd))

/* analog of ++bin */
// 通过bin上的一个chunk，找到物理地址的下一个chunk地址
#define next_bin(b) ((mbinptr) ((char *) (b) + (sizeof (mchunkptr) << 1)))

/* Reminders about list directionality within bins */
// 获取空闲bins上的chunk双向链表指针，找到前后的chunk
#define first(b) ((b)->fd)
#define last(b) ((b)->bk)

/*
   Indexing

    Bins for sizes < 512 bytes contain chunks of all the same size, spaced
    8 bytes apart. Larger bins are approximately logarithmically spaced:

    64 bins of size       8
    32 bins of size      64
    16 bins of size     512
     8 bins of size    4096
     4 bins of size   32768
     2 bins of size  262144
     1 bin  of size what's left

    There is actually a little bit of slop in the numbers in bin_index
    for the sake of speed. This makes no difference elsewhere.

    The bins top out around 1MB because we expect to service large
    requests via mmap.

    Bin 0 does not exist.  Bin 1 is the unordered list; if that would be
    a valid chunk size the small bins are bumped up one.
 */

#define NBINS 128			// 默认128个bins
#define NSMALLBINS 64			// small bin的个数
#define SMALLBIN_WIDTH MALLOC_ALIGNMENT // small bin的宽度
#define SMALLBIN_CORRECTION (MALLOC_ALIGNMENT > CHUNK_HDR_SZ)
#define MIN_LARGE_SIZE ((NSMALLBINS - SMALLBIN_CORRECTION) * SMALLBIN_WIDTH)
 // sz的大小是否属于small bin
#define in_smallbin_range(sz)                                                 \
  ((unsigned long) (sz) < (unsigned long) MIN_LARGE_SIZE)

// 转换为smallbin的索引
#define smallbin_index(sz)                                                    \
  ((SMALLBIN_WIDTH == 16 ? (((unsigned) (sz)) >> 4)                           \
			 : (((unsigned) (sz)) >> 3))                          \
   + SMALLBIN_CORRECTION)

// 32位系统转换为largebin的索引
#define largebin_index_32(sz)                                                 \
  (((((unsigned long) (sz)) >> 6) <= 38) ? 56 + (((unsigned long) (sz)) >> 6) \
   : ((((unsigned long) (sz)) >> 9) <= 20)                                    \
       ? 91 + (((unsigned long) (sz)) >> 9)                                   \
   :\   
((((unsigned long)(sz)) >> 12) <= 10)                                     \
? 110 + (((unsigned long)(sz)) >> 12)                                 \
    : ((((unsigned long)(sz)) >> 15) <= 4)                                    \
    ? 119 + (((unsigned long)(sz)) >> 15)                                 \
    : ((((unsigned long)(sz)) >> 18) <= 2)                                    \
    ? 124 + (((unsigned long)(sz)) >> 18)                                 \
    : 126)

#define largebin_index_32_big(sz)                                             \
  (((((unsigned long) (sz)) >> 6) <= 45) ?  49 + (((unsigned long) (sz)) >> 6) :\   // 步长为2^6，为64
    ((((unsigned long)(sz)) >> 9) <= 20) ? 91 + (((unsigned long)(sz)) >> 9) : \   // 步长为2^9，为512   
    ((((unsigned long)(sz)) >> 12) <= 10) ? 110 + (((unsigned long)(sz)) >> 12) : \ // 步长为2^12，为4096
    ((((unsigned long)(sz)) >> 15) <= 4) ? 119 + (((unsigned long)(sz)) >> 15) : \  // 步长为2^15，为32768
    ((((unsigned long)(sz)) >> 18) <= 2) ? 124 + (((unsigned long)(sz)) >> 18) : \  // 步长为2^18，为262144
    126)

    // XXX It remains to be seen whether it is good to keep the widths of
    // XXX the buckets the same or whether it should be scaled by a factor
    // XXX of two as well.
    // 64位系统转换为largebin的索引
#define largebin_index_64(sz)                                                 \
  (((((unsigned long) (sz)) >> 6) <= 48) ?  48 + (((unsigned long) (sz)) >> 6) :\   // 步长为2^6，为64
    ((((unsigned long)(sz)) >> 9) <= 20) ? 91 + (((unsigned long)(sz)) >> 9) : \   // 步长为2^9，为512
    ((((unsigned long)(sz)) >> 12) <= 10) ? 110 + (((unsigned long)(sz)) >> 12) : \ // 步长为2^12，为4096
    ((((unsigned long)(sz)) >> 15) <= 4) ? 119 + (((unsigned long)(sz)) >> 15) : \  // 步长为2^15，为32768
    ((((unsigned long)(sz)) >> 18) <= 2) ? 124 + (((unsigned long)(sz)) >> 18) : \  // 步长为2^18，为262144
    126)

    // 大小转换为索引的largebin总接口
#define largebin_index(sz)                                                    \
  (SIZE_SZ == 8		    ? largebin_index_64 (sz)                          \
   : MALLOC_ALIGNMENT == 16 ? largebin_index_32_big (sz)                      \
			    : largebin_index_32 (sz))

   // 将sz大小转换成对应的数组下标。（判断属于large bin还是small bin）。
   // 例：第一个largebin的起始大小为1024，那么1024>>6=16，所以其在bins数组中的下标为48+16=64
#define bin_index(sz)                                                         \
  ((in_smallbin_range (sz)) ? smallbin_index (sz) : largebin_index (sz))

/* Take a chunk off a bin list.  */
static void
unlink_chunk(mstate av, mchunkptr p)
{
    if (chunksize(p) != prev_size(next_chunk(p)))
        malloc_printerr("corrupted size vs. prev_size");

    mchunkptr fd = p->fd;
    mchunkptr bk = p->bk;

    if (__builtin_expect(fd->bk != p || bk->fd != p, 0))
        malloc_printerr("corrupted double-linked list");

    fd->bk = bk;
    bk->fd = fd;
    if (!in_smallbin_range(chunksize_nomask(p)) && p->fd_nextsize != NULL)
    {
        if (p->fd_nextsize->bk_nextsize != p
            || p->bk_nextsize->fd_nextsize != p)
            malloc_printerr("corrupted double-linked list (not small)");

        if (fd->fd_nextsize == NULL)
        {
            if (p->fd_nextsize == p)
                fd->fd_nextsize = fd->bk_nextsize = fd;
            else
            {
                fd->fd_nextsize = p->fd_nextsize;
                fd->bk_nextsize = p->bk_nextsize;
                p->fd_nextsize->bk_nextsize = fd;
                p->bk_nextsize->fd_nextsize = fd;
            }
        }
        else
        {
            p->fd_nextsize->bk_nextsize = p->bk_nextsize;
            p->bk_nextsize->fd_nextsize = p->fd_nextsize;
        }
    }
}

/*
   Unsorted chunks

    All remainders from chunk splits, as well as all returned chunks,
    are first placed in the "unsorted" bin. They are then placed
    in regular bins after malloc gives them ONE chance to be used before
    binning. So, basically, the unsorted_chunks list acts as a queue,
    with chunks being placed on it in free (and malloc_consolidate),
    and taken off (to be either used or placed in bins) in malloc.

    The NON_MAIN_ARENA flag is never set for unsorted chunks, so it
    does not have to be taken into account in size comparisons.
 */

 /* The otherwise unindexable 1-bin is used to hold unsorted chunks. */
#define unsorted_chunks(M) (bin_at (M, 1))

   /*
      Top

       The top-most available chunk (i.e., the one bordering the end of
       available memory) is treated specially. It is never included in
       any bin, is used only if no other chunk is available, and is
       released back to the system if it is very large (see
       M_TRIM_THRESHOLD).  Because top initially
       points to its own bin with initial zero size, thus forcing
       extension on the first malloc request, we avoid having any special
       code in malloc to check whether it even exists yet. But we still
       need to do so when getting memory from system, so we make
       initial_top treat the bin as a legal but unusable chunk during the
       interval between initialization and the first call to
       sysmalloc. (This is somewhat delicate, since it relies on
       the 2 preceding words to be zero during this interval as well.)
    */

    /* Conveniently, the unsorted bin can be used as dummy top on first call */
#define initial_top(M) (unsorted_chunks (M))

   /*
      Binmap

       To help compensate for the large number of bins, a one-level index
       structure is used for bin-by-bin searching.  `binmap' is a
       bitvector recording whether bins are definitely empty so they can
       be skipped over during during traversals.  The bits are NOT always
       cleared as soon as bins are empty, but instead only
       when they are noticed to be empty during traversal in malloc.
    */

    /* Conservatively use 32 bits per map word, even if on 64bit system */
 //即使是64位系统，也使用32位的方式映射；无符号int 4个字节，每个字节8bit，一共 32位
#define BINMAPSHIFT 5
//值为32
#define BITSPERMAP (1U << BINMAPSHIFT)
// binmap一共128bits，binmap按int分成4个block，每个block有32个bit
#define BINMAPSIZE (NBINS / BITSPERMAP)

//一共128个bin，i从0开始到127，通过(i) >> BINMAPSHIFT，可以得出在第几个block中
#define idx2block(i) ((i) >> BINMAPSHIFT) //计算出该 bin 在 binmap 对应的 bit 属于哪个 block
//取第i位为1，其他位都为0的掩码
#define idx2bit(i) ((1U << ((i) & ((1U << BINMAPSHIFT) - 1))))
// 设置第i个bin在binmap中对应的bit位为 1
#define mark_bin(m, i) ((m)->binmap[idx2block (i)] |= idx2bit (i))
// 设置第i个bin在binmap中对应的bit位为0
#define unmark_bin(m, i) ((m)->binmap[idx2block (i)] &= ~(idx2bit (i)))
//获取第i个bin在binmap中对应的bit
#define get_binmap(m, i) ((m)->binmap[idx2block (i)] & idx2bit (i))

   /*
      Fastbins

       An array of lists holding recently freed small chunks.  Fastbins
       are not doubly linked.  It is faster to single-link them, and
       since chunks are never removed from the middles of these lists,
       double linking is not necessary. Also, unlike regular bins, they
       are not even processed in FIFO order (they use faster LIFO) since
       ordering doesn't much matter in the transient contexts in which
       fastbins are normally used.

       Chunks in fastbins keep their inuse bit set, so they cannot
       be consolidated with other free chunks. malloc_consolidate
       releases all chunks in fastbins and consolidates them with
       other free chunks.
    */

typedef struct malloc_chunk* mfastbinptr;
#define fastbin(ar_ptr, idx) ((ar_ptr)->fastbinsY[idx])

/* offset 2 to use otherwise unindexable first 2 bins */
#define fastbin_index(sz)                                                     \
  ((((unsigned int) (sz)) >> (SIZE_SZ == 8 ? 4 : 3)) - 2)

   /* The maximum fastbin request size we support */
#define MAX_FAST_SIZE (80 * SIZE_SZ / 4)

#define NFASTBINS (fastbin_index (request2size (MAX_FAST_SIZE)) + 1)

   /*
      FASTBIN_CONSOLIDATION_THRESHOLD is the size of a chunk in free()
      that triggers automatic consolidation of possibly-surrounding
      fastbin chunks. This is a heuristic, so the exact value should not
      matter too much. It is defined at half the default trim threshold as a
      compromise heuristic to only attempt consolidation if it is likely
      to lead to trimming. However, it is not dynamically tunable, since
      consolidation reduces fragmentation surrounding large chunks even
      if trimming is not used.
    */

#define FASTBIN_CONSOLIDATION_THRESHOLD (65536UL)

    /*
       NONCONTIGUOUS_BIT indicates that MORECORE does not return contiguous
       regions.  Otherwise, contiguity is exploited in merging together,
       when possible, results from consecutive MORECORE calls.

       The initial value comes from MORECORE_CONTIGUOUS, but is
       changed dynamically if mmap is ever used as an sbrk substitute.

       NONCONTIGUOUS_BIT指示MORECORE不返回连续区域。 否则，在可能的情况下，会利用连续的MORECORE调用的结果来合并在一起。
       初始值来自MORECORE_CONTIGUOUS，但如果mmap曾经用作sbrk替代品，则会动态更改。
     */

#define NONCONTIGUOUS_BIT (2U)

#define contiguous(M) (((M)->flags & NONCONTIGUOUS_BIT) == 0)
#define noncontiguous(M) (((M)->flags & NONCONTIGUOUS_BIT) != 0)
#define set_noncontiguous(M) ((M)->flags |= NONCONTIGUOUS_BIT)
#define set_contiguous(M) ((M)->flags &= ~NONCONTIGUOUS_BIT)

     /* Maximum size of memory handled in fastbins.  */
static uint8_t global_max_fast;

/*
   Set value of max_fast.
   Use impossibly small value if 0.
   Precondition: there are no existing fastbin chunks in the main arena.
   Since do_check_malloc_state () checks this, we call malloc_consolidate ()
   before changing max_fast.  Note other arenas will leak their fast bin
   entries if max_fast is reduced.
 */

#define set_max_fast(s)                                                       \
  global_max_fast = (((size_t) (s) <= MALLOC_ALIGN_MASK - SIZE_SZ)            \
			 ? MIN_CHUNK_SIZE / 2                                 \
			 : ((s + SIZE_SZ) & ~MALLOC_ALIGN_MASK))

static inline INTERNAL_SIZE_T
get_max_fast(void)
{
    /* Tell the GCC optimizers that global_max_fast is never larger
   than MAX_FAST_SIZE.  This avoids out-of-bounds array accesses in
   _int_malloc after constant propagation of the size parameter.
   (The code never executes because malloc preserves the
   global_max_fast invariant, but the optimizers may not recognize
   this.)  */
    if (global_max_fast > MAX_FAST_SIZE)
        __builtin_unreachable();
    return global_max_fast;
}

/*
   ----------- Internal state representation and initialization -----------
 */

 /*
    have_fastchunks indicates that there are probably some fastbin chunks.
    It is set true on entering a chunk into any fastbin, and cleared early in
    malloc_consolidate.  The value is approximate since it may be set when
    there are no fastbin chunks, or it may be clear even if there are fastbin
    chunks available.  Given it's sole purpose is to reduce number of
    redundant calls to malloc_consolidate, it does not affect correctness. As
    a result we can safely use relaxed atomic accesses.
  */

struct malloc_state
{
    /* Serialize access.  */
    // 同步访问互斥锁
    __libc_lock_define(, mutex);

    /* Flags (formerly in max_fast).  */
    // 用于标记当前主分配区的状态
    int flags;

    /* Set if the fastbin chunks contain recently inserted free blocks.  */
    /* Note this is a bool but not all targets support atomics on booleans. */
    // 用于标记是否有fastchunk，如果fastbin chunks包含最近插入的空闲块就置位
    int have_fastchunks;

    /* Fastbins */
    // fast
    // bins是bins的高速缓冲区，大约有10个定长队列。当用户释放一块不大于max_fast(默认值为64)的chunk的时候，会默认放在fastbins中
    mfastbinptr fastbinsY[NFASTBINS];

    /* Base of the topmost chunk -- not otherwise kept in a bin */
    // 并不是所有的chunk都会被放到bins上。
    // top
    // chunk相当于分配区的顶部空闲内存，当bins上都不能满足内存分配要求的时候，就会来top
    // chunk上分配。
    mchunkptr top;

    /* The remainder from the most recent split of a small request */
    // 最近一次小请求分割的剩余部分
    mchunkptr last_remainder;

    /* Normal bins packed as described above
     * 常规 bins chunk的链表数组
     * 1. unsorted
     * bin：是bins的一个缓冲区。当用户释放的内存大于max_fast或者fast
     * bins合并后的chunk都会进入unsorted bin上
     * 2. small bins和large bins。small bins和large
     * bins是真正用来放置chunk双向链表的。每个bin之间相差8个字节，并且通过上面的这个列表，
     * 可以快速定位到合适大小的空闲chunk。
     * 3. 下标1是unsorted bin，2到63是small bin，64到126是large
     * bin，共126个bin
     */
    mchunkptr bins[NBINS * 2 - 2];

    /* Bitmap of bins */
    // 表示bin数组当中某一个下标的bin是否为空，用来在分配的时候加速
    unsigned int binmap[BINMAPSIZE];

    /* Linked list 分配区全局链表
   通过next来链接分配区，其中主分配区放链表头部，新加入的分配区放main_arena.next
    */
    struct malloc_state* next;

    /* Linked list for free arenas.  Access to this field is serialized
   by free_list_lock in arena.c.  */
   // 分配区空闲链表
    struct malloc_state* next_free;

    /* Number of threads attached to this arena.  0 if the arena is on
   the free list.  Access to this field is serialized by
   free_list_lock in arena.c.  */
   // freelist的状态，0-空闲 1-正在使用中，关联的线程数
    INTERNAL_SIZE_T attached_threads;

    /* Memory allocated from the system in this arena.  */
    // 该分配区从系统中分配的内存
    INTERNAL_SIZE_T system_mem;
    INTERNAL_SIZE_T max_system_mem;
};

struct malloc_par
{
    /* Tunable parameters */
    unsigned long trim_threshold;
    INTERNAL_SIZE_T top_pad;
    INTERNAL_SIZE_T mmap_threshold;
    INTERNAL_SIZE_T arena_test;
    INTERNAL_SIZE_T arena_max;

    /* Transparent Large Page support.  */
    INTERNAL_SIZE_T thp_pagesize;
    /* A value different than 0 means to align mmap allocation to hp_pagesize
   add hp_flags on flags.
   值不等于0意味着将mmap分配与hp_pagesize对齐，并在标志上添加hp_flags。*/
    INTERNAL_SIZE_T hp_pagesize;
    int hp_flags;

    /* Memory map support */
    int n_mmaps;
    int n_mmaps_max;
    int max_n_mmaps;
    /* the mmap_threshold is dynamic, until the user sets
   it manually, at which point we need to disable any
   dynamic behavior. */
    int no_dyn_threshold;

    /* Statistics */
    INTERNAL_SIZE_T mmapped_mem;
    INTERNAL_SIZE_T max_mmapped_mem;

    /* First address handed out by MORECORE/sbrk.  */
    char* sbrk_base;

#if USE_TCACHE
    /* Maximum number of buckets to use.  */
    size_t tcache_bins;
    size_t tcache_max_bytes;
    /* Maximum number of chunks in each bucket.  */
    size_t tcache_count;
    /* Maximum number of chunks to remove from the unsorted list, which
   aren't used to prefill the cache.  */
    size_t tcache_unsorted_limit;
#endif
};

/* There are several instances of this struct ("arenas") in this
   malloc.  If you are adapting this malloc in a way that does NOT use
   a static or mmapped malloc_state, you MUST explicitly zero-fill it
   before using. This malloc relies on the property that malloc_state
   is initialized to all zeroes (as is true of C statics).  */

static struct malloc_state main_arena = { .mutex = _LIBC_LOCK_INITIALIZER,
                      .next = &main_arena,
                      .attached_threads = 1 };

/* There is only one instance of the malloc parameters.  */

static struct malloc_par mp_ = {
  .top_pad = DEFAULT_TOP_PAD,
  .n_mmaps_max = DEFAULT_MMAP_MAX,
  .mmap_threshold = DEFAULT_MMAP_THRESHOLD,
  .trim_threshold = DEFAULT_TRIM_THRESHOLD,
#define NARENAS_FROM_NCORES(n) ((n) * (sizeof (long) == 4 ? 2 : 8))
     .arena_test = NARENAS_FROM_NCORES(1)
#if USE_TCACHE
     ,
     .tcache_count = TCACHE_FILL_COUNT,
     .tcache_bins = TCACHE_MAX_BINS,
     .tcache_max_bytes = tidx2usize(TCACHE_MAX_BINS - 1),
     .tcache_unsorted_limit = 0 /* No limit.  */
#endif
};

/*
   Initialize a malloc_state struct.
   初始化malloc_state结构体
   This is called from ptmalloc_init () or from _int_new_arena ()
   when creating a new arena.
 */

static void
malloc_init_state(mstate av)
{
    int i;
    mbinptr bin;

    /* Establish circular links for normal bins */
    // 将bin数组中的每个bin的fd和bk都指向自己
    for (i = 1; i < NBINS; ++i)
    {
        bin = bin_at(av, i);
        bin->fd = bin->bk = bin;
    }

#if MORECORE_CONTIGUOUS
    if (av != &main_arena)
#endif
        set_noncontiguous(av);
    if (av == &main_arena)
        set_max_fast(DEFAULT_MXFAST);
    /* 默认fastchunk 是false的，没有被初始化的 */
    atomic_store_relaxed(&av->have_fastchunks, false);
    // 初始化Top chunk，默认指向了unsorted bin上的第一个chunk
    av->top = initial_top(av);
}

/*
   Other internal utilities operating on mstates
 */

static void* sysmalloc(INTERNAL_SIZE_T, mstate);
static int systrim(size_t, mstate);
static void malloc_consolidate(mstate);

/* -------------- Early definitions for debugging hooks ---------------- */

/* This function is called from the arena shutdown hook, to free the
   thread cache (if it exists).  */
static void tcache_thread_shutdown(void);

/* ------------------ Testing support ----------------------------------*/

static int perturb_byte;

static void
alloc_perturb(char* p, size_t n)
{
    if (__glibc_unlikely(perturb_byte))
        memset(p, perturb_byte ^ 0xff, n);
}

static void
free_perturb(char* p, size_t n)
{
    if (__glibc_unlikely(perturb_byte))
        memset(p, perturb_byte, n);
}

#include <stap-probe.h>

/* ----------- Routines dealing with transparent huge pages ----------- */

static inline void
madvise_thp(void* p, INTERNAL_SIZE_T size)
{
#ifdef MADV_HUGEPAGE
    /* Do not consider areas smaller than a huge page or if the tunable is
   not active.  */
    if (mp_.thp_pagesize == 0 || size < mp_.thp_pagesize)
        return;

    /* Linux requires the input address to be page-aligned, and unaligned
   inputs happens only for initial data segment.  */
    if (__glibc_unlikely(!PTR_IS_ALIGNED(p, GLRO(dl_pagesize))))
    {
        void* q = PTR_ALIGN_DOWN(p, GLRO(dl_pagesize));
        size += PTR_DIFF(p, q);
        p = q;
    }

    __madvise(p, size, MADV_HUGEPAGE);
#endif
}

/* ------------------- Support for multiple arenas -------------------- */
#include "arena.c"

   /*
      Debugging support

      These routines make a number of assertions about the states
      of data structures that should be true at all times. If any
      are not true, it's very likely that a user program has somehow
      trashed memory. (It's also possible that there is a coding error
      in malloc. In which case, please report it!)
    */

#if !MALLOC_DEBUG

#  define check_chunk(A, P)
#  define check_free_chunk(A, P)
#  define check_inuse_chunk(A, P)
#  define check_remalloced_chunk(A, P, N)
#  define check_malloced_chunk(A, P, N)
#  define check_malloc_state(A)

#else

#  define check_chunk(A, P) do_check_chunk (A, P)
#  define check_free_chunk(A, P) do_check_free_chunk (A, P)
#  define check_inuse_chunk(A, P) do_check_inuse_chunk (A, P)
#  define check_remalloced_chunk(A, P, N) do_check_remalloced_chunk (A, P, N)
#  define check_malloced_chunk(A, P, N) do_check_malloced_chunk (A, P, N)
#  define check_malloc_state(A) do_check_malloc_state (A)

    /*
       Properties of all chunks
     */

static void
do_check_chunk(mstate av, mchunkptr p)
{
    unsigned long sz = chunksize(p);
    /* min and max possible addresses assuming contiguous allocation */
    char* max_address = (char*)(av->top) + chunksize(av->top);
    char* min_address = max_address - av->system_mem;

    if (!chunk_is_mmapped(p))
    {
        /* Has legal address ... */
        if (p != av->top)
        {
            if (contiguous(av))
            {
                assert(((char*)p) >= min_address);
                assert(((char*)p + sz) <= ((char*)(av->top)));
            }
        }
        else
        {
            /* top size is always at least MINSIZE */
            assert((unsigned long)(sz) >= MINSIZE);
            /* top predecessor always marked inuse */
            assert(prev_inuse(p));
        }
    }
    else
    {
        /* address is outside main heap  */
        if (contiguous(av) && av->top != initial_top(av))
        {
            assert(((char*)p) < min_address
                || ((char*)p) >= max_address);
        }
        /* chunk is page-aligned */
        assert(((prev_size(p) + sz) & (GLRO(dl_pagesize) - 1)) == 0);
        /* mem is aligned */
        assert(aligned_OK(chunk2mem(p)));
    }
}

/*
   Properties of free chunks
 */

static void
do_check_free_chunk(mstate av, mchunkptr p)
{
    INTERNAL_SIZE_T sz
        = chunksize_nomask(p) & ~(PREV_INUSE | NON_MAIN_ARENA);
    mchunkptr next = chunk_at_offset(p, sz);

    do_check_chunk(av, p);

    /* Chunk must claim to be free ... */
    assert(!inuse(p));
    assert(!chunk_is_mmapped(p));

    /* Unless a special marker, must have OK fields */
    if ((unsigned long)(sz) >= MINSIZE)
    {
        assert((sz & MALLOC_ALIGN_MASK) == 0);
        assert(aligned_OK(chunk2mem(p)));
        /* ... matching footer field */
        assert(prev_size(next_chunk(p)) == sz);
        /* ... and is fully consolidated */
        assert(prev_inuse(p));
        assert(next == av->top || inuse(next));

        /* ... and has minimally sane links */
        assert(p->fd->bk == p);
        assert(p->bk->fd == p);
    }
    else /* markers are always of size SIZE_SZ */
        assert(sz == SIZE_SZ);
}

/*
   Properties of inuse chunks
 */

static void
do_check_inuse_chunk(mstate av, mchunkptr p)
{
    mchunkptr next;

    do_check_chunk(av, p);

    if (chunk_is_mmapped(p))
        return; /* mmapped chunks have no next/prev */

    /* Check whether it claims to be in use ... */
    assert(inuse(p));

    next = next_chunk(p);

    /* ... and is surrounded by OK chunks.
   Since more things can be checked with free chunks than inuse ones,
   if an inuse chunk borders them and debug is on, it's worth doing them.
     */
    if (!prev_inuse(p))
    {
        /* Note that we cannot even look at prev unless it is not inuse */
        mchunkptr prv = prev_chunk(p);
        assert(next_chunk(prv) == p);
        do_check_free_chunk(av, prv);
    }

    if (next == av->top)
    {
        assert(prev_inuse(next));
        assert(chunksize(next) >= MINSIZE);
    }
    else if (!inuse(next))
        do_check_free_chunk(av, next);
}

/*
   Properties of chunks recycled from fastbins
 */

static void
do_check_remalloced_chunk(mstate av, mchunkptr p, INTERNAL_SIZE_T s)
{
    INTERNAL_SIZE_T sz
        = chunksize_nomask(p) & ~(PREV_INUSE | NON_MAIN_ARENA);

    if (!chunk_is_mmapped(p))
    {
        assert(av == arena_for_chunk(p));
        if (chunk_main_arena(p))
            assert(av == &main_arena);
        else
            assert(av != &main_arena);
    }

    do_check_inuse_chunk(av, p);

    /* Legal size ... */
    assert((sz & MALLOC_ALIGN_MASK) == 0);
    assert((unsigned long)(sz) >= MINSIZE);
    /* ... and alignment */
    assert(aligned_OK(chunk2mem(p)));
    /* chunk is less than MINSIZE more than request */
    assert((long)(sz)-(long)(s) >= 0);
    assert((long)(sz)-(long)(s + MINSIZE) < 0);
}

/*
   Properties of nonrecycled chunks at the point they are malloced
 */

static void
do_check_malloced_chunk(mstate av, mchunkptr p, INTERNAL_SIZE_T s)
{
    /* same as recycled case ... */
    do_check_remalloced_chunk(av, p, s);

    /*
   ... plus,  must obey implementation invariant that prev_inuse is
   always true of any allocated chunk; i.e., that each allocated
   chunk borders either a previously allocated and still in-use
   chunk, or the base of its memory arena. This is ensured
   by making all allocations from the `lowest' part of any found
   chunk.  This does not necessarily hold however for chunks
   recycled via fastbins.
     */

    assert(prev_inuse(p));
}

/*
   Properties of malloc_state.

   This may be useful for debugging malloc, as well as detecting user
   programmer errors that somehow write into malloc_state.

   If you are extending or experimenting with this malloc, you can
   probably figure out how to hack this routine to print out or
   display chunk addresses, sizes, bins, and other instrumentation.
 */

static void
do_check_malloc_state(mstate av)
{
    int i;
    mchunkptr p;
    mchunkptr q;
    mbinptr b;
    unsigned int idx;
    INTERNAL_SIZE_T size;
    unsigned long total = 0;
    int max_fast_bin;

    /* internal size_t must be no wider than pointer type */
    assert(sizeof(INTERNAL_SIZE_T) <= sizeof(char*));

    /* alignment is a power of 2 */
    assert((MALLOC_ALIGNMENT & (MALLOC_ALIGNMENT - 1)) == 0);

    /* Check the arena is initialized. */
    assert(av->top != 0);

    /* No memory has been allocated yet, so doing more tests is not possible.
     */
    if (av->top == initial_top(av))
        return;

    /* pagesize is a power of 2 */
    assert(powerof2(GLRO(dl_pagesize)));

    /* A contiguous main_arena is consistent with sbrk_base.  */
    if (av == &main_arena && contiguous(av))
        assert((char*)mp_.sbrk_base + av->system_mem
            == (char*)av->top + chunksize(av->top));

    /* properties of fastbins */

    /* max_fast is in allowed range */
    assert((get_max_fast() & ~1) <= request2size(MAX_FAST_SIZE));

    max_fast_bin = fastbin_index(get_max_fast());

    for (i = 0; i < NFASTBINS; ++i)
    {
        p = fastbin(av, i);

        /* The following test can only be performed for the main arena.
           While mallopt calls malloc_consolidate to get rid of all fast
           bins (especially those larger than the new maximum) this does
           only happen for the main arena.  Trying to do this for any
           other arena would mean those arenas have to be locked and
           malloc_consolidate be called for them.  This is excessive.  And
           even if this is acceptable to somebody it still cannot solve
           the problem completely since if the arena is locked a
           concurrent malloc call might create a new arena which then
           could use the newly invalid fast bins.  */

           /* all bins past max_fast are empty */
        if (av == &main_arena && i > max_fast_bin)
            assert(p == 0);

        while (p != 0)
        {
            if (__glibc_unlikely(misaligned_chunk(p)))
                malloc_printerr("do_check_malloc_state(): "
                    "unaligned fastbin chunk detected");
            /* each chunk claims to be inuse */
            do_check_inuse_chunk(av, p);
            total += chunksize(p);
            /* chunk belongs in this bin */
            assert(fastbin_index(chunksize(p)) == i);
            p = REVEAL_PTR(p->fd);
        }
    }

    /* check normal bins */
    for (i = 1; i < NBINS; ++i)
    {
        b = bin_at(av, i);

        /* binmap is accurate (except for bin 1 == unsorted_chunks) */
        if (i >= 2)
        {
            unsigned int binbit = get_binmap(av, i);
            int empty = last(b) == b;
            if (!binbit)
                assert(empty);
            else if (!empty)
                assert(binbit);
        }

        for (p = last(b); p != b; p = p->bk)
        {
            /* each chunk claims to be free */
            do_check_free_chunk(av, p);
            size = chunksize(p);
            total += size;
            if (i >= 2)
            {
                /* chunk belongs in bin */
                idx = bin_index(size);
                assert(idx == i);
                /* lists are sorted */
                assert(p->bk == b
                    || (unsigned long)chunksize(p->bk)
                    >= (unsigned long)chunksize(p));

                if (!in_smallbin_range(size))
                {
                    if (p->fd_nextsize != NULL)
                    {
                        if (p->fd_nextsize == p)
                            assert(p->bk_nextsize == p);
                        else
                        {
                            if (p->fd_nextsize == first(b))
                                assert(chunksize(p)
                                    < chunksize(p->fd_nextsize));
                            else
                                assert(chunksize(p)
                                    > chunksize(p->fd_nextsize));

                            if (p == first(b))
                                assert(chunksize(p)
    > chunksize(p->bk_nextsize));
                            else
                                assert(chunksize(p)
                                    < chunksize(p->bk_nextsize));
                        }
                    }
                    else
                        assert(p->bk_nextsize == NULL);
                }
            }
            else if (!in_smallbin_range(size))
                assert(p->fd_nextsize == NULL && p->bk_nextsize == NULL);
            /* chunk is followed by a legal chain of inuse chunks */
            for (q = next_chunk(p);
                (q != av->top && inuse(q)
                    && (unsigned long)(chunksize(q)) >= MINSIZE);
                q = next_chunk(q))
                do_check_inuse_chunk(av, q);
        }
    }

    /* top chunk is OK */
    check_chunk(av, av->top);
}
#endif

/* ----------------- Support for debugging hooks -------------------- */
#if IS_IN(libc)
#  include "hooks.c"
#endif

   /* ----------- Routines dealing with system allocation -------------- */

   /*
      sysmalloc handles malloc cases requiring more memory from the system.
      On entry, it is assumed that av->top does not have enough
      space to service request for nb bytes, thus requiring that av->top
      be extended or replaced.
    */

static void*
sysmalloc_mmap(INTERNAL_SIZE_T nb, size_t pagesize, int extra_flags,
    mstate av)
{
    long int size;

    /*
      Round up size to nearest page.  For mmapped chunks, the overhead is one
      SIZE_SZ unit larger than for normal chunks, because there is no
      following chunk whose prev_size field could be used.

      See the front_misalign handling below, for glibc there is no need for
      further alignments unless we have have high alignment.
     */
    if (MALLOC_ALIGNMENT == CHUNK_HDR_SZ)
        size = ALIGN_UP(nb + SIZE_SZ, pagesize);
    else
        size = ALIGN_UP(nb + SIZE_SZ + MALLOC_ALIGN_MASK, pagesize);

    /* Don't try if size wraps around 0.  */
    if ((unsigned long)(size) <= (unsigned long)(nb))
        return MAP_FAILED;

    char* mm = (char*)MMAP(
        0, size, mtag_mmap_flags | PROT_READ | PROT_WRITE, extra_flags);
    if (mm == MAP_FAILED)
        return mm;

#ifdef MAP_HUGETLB
    if (!(extra_flags & MAP_HUGETLB))
        madvise_thp(mm, size);
#endif

    __set_vma_name(mm, size, " glibc: malloc");

    /*
      The offset to the start of the mmapped region is stored in the prev_size
      field of the chunk.  This allows us to adjust returned start address to
      meet alignment requirements here and in memalign(), and still be able to
      compute proper address argument for later munmap in free() and
      realloc().
     */

    INTERNAL_SIZE_T front_misalign; /* unusable bytes at front of new space */

    if (MALLOC_ALIGNMENT == CHUNK_HDR_SZ)
    {
        /* For glibc, chunk2mem increases the address by CHUNK_HDR_SZ and
           MALLOC_ALIGN_MASK is CHUNK_HDR_SZ-1.  Each mmap'ed area is page
           aligned and therefore definitely MALLOC_ALIGN_MASK-aligned.  */
        assert(((INTERNAL_SIZE_T)chunk2mem(mm) & MALLOC_ALIGN_MASK) == 0);
        front_misalign = 0;
    }
    else
        front_misalign = (INTERNAL_SIZE_T)chunk2mem(mm) & MALLOC_ALIGN_MASK;

    mchunkptr p; /* the allocated/returned chunk */

    if (front_misalign > 0)
    {
        ptrdiff_t correction = MALLOC_ALIGNMENT - front_misalign;
        p = (mchunkptr)(mm + correction);
        set_prev_size(p, correction);
        set_head(p, (size - correction) | IS_MMAPPED);
    }
    else
    {
        p = (mchunkptr)mm;
        set_prev_size(p, 0);
        set_head(p, size | IS_MMAPPED);
    }

    /* update statistics */
    int new = atomic_fetch_add_relaxed(&mp_.n_mmaps, 1) + 1;
    atomic_max(&mp_.max_n_mmaps, new);

    unsigned long sum;
    sum = atomic_fetch_add_relaxed(&mp_.mmapped_mem, size) + size;
    atomic_max(&mp_.max_mmapped_mem, sum);

    check_chunk(av, p);

    return chunk2mem(p);
}

/*
   Allocate memory using mmap() based on S and NB requested size, aligning
   to PAGESIZE if required.  The EXTRA_FLAGS is used on mmap() call.  If the
   call succeeds S is updated with the allocated size.  This is used as a
   fallback if MORECORE fails.
 */
static void*
sysmalloc_mmap_fallback(long int* s, INTERNAL_SIZE_T nb,
    INTERNAL_SIZE_T old_size, size_t minsize,
    size_t pagesize, int extra_flags, mstate av)
{
    long int size = *s;

    /* Cannot merge with old top, so add its size back in */
    if (contiguous(av))
        size = ALIGN_UP(size + old_size, pagesize);

    /* If we are relying on mmap as backup, then use larger units */
    if ((unsigned long)(size) < minsize)
        size = minsize;

    /* Don't try if size wraps around 0 */
    if ((unsigned long)(size) <= (unsigned long)(nb))
        return MORECORE_FAILURE;

    char* mbrk = (char*)(MMAP(
        0, size, mtag_mmap_flags | PROT_READ | PROT_WRITE, extra_flags));
    if (mbrk == MAP_FAILED)
        return MAP_FAILED;

#ifdef MAP_HUGETLB
    if (!(extra_flags & MAP_HUGETLB))
        madvise_thp(mbrk, size);
#endif

    __set_vma_name(mbrk, size, " glibc: malloc");

    /* Record that we no longer have a contiguous sbrk region.  After the
   first time mmap is used as backup, we do not ever rely on contiguous
   space since this could incorrectly bridge regions.  */
    set_noncontiguous(av);

    *s = size;
    return mbrk;
}

/*
  调用系统分配函数：sysmalloc
  说明：进入sysmalloc则表示 top chunk的空间不足了，需要进行扩容av->top
  nb：请求的内存大小
  mstate：内存分配状态机(分配区)
*/
static void*
sysmalloc(INTERNAL_SIZE_T nb, mstate av)
{
    mchunkptr old_top;	       /* 老的Top chunk的地址指针 incoming value of av->top */
    INTERNAL_SIZE_T old_size; /* 老的Top chunk的大小 its size */
    char* old_end;	       /* 老的Top chunk的尾部地址 its end address */

    long size; /* 第一次分配的内存 arg to first MORECORE or mmap call */
    char* brk; /* 通过brk分配后返回的对象 return value from MORECORE */

    long correction; /* 用于记录第二次分配的值 arg to 2nd MORECORE call */
    char* snd_brk;   /* 第二次处理后返回的值（也是第一次的尾部）2nd return val */

    INTERNAL_SIZE_T front_misalign; /* 不可用的新空间的头部字节 unusable bytes at front of new space */
    INTERNAL_SIZE_T end_misalign; /* 新内存块尾部对齐的字节 partial page left at end of new space */
    char* aligned_brk;		   /* 对齐后的brk值 aligned offset into brk */

    mchunkptr p;		   /* 返回的结果p the allocated/returned chunk */
    mchunkptr remainder;	   /* Top chunk切割后剩余的remainder chunk remainder from allocation */
    unsigned long remainder_size; /* Top chunk切割后剩余的remainder chunk 的size  its size */

    size_t pagesize = GLRO(dl_pagesize);
    bool tried_mmap = false;

    /*
     If have mmap, and the request size meets the mmap threshold, and
     the system supports mmap, and there are few enough currently
     allocated mmapped regions, try to directly map this request
     rather than expanding top.

     1. av==NULL，则直接采用MMAP的方式分配内存
     2. nb分配的内存为大对象（超过128k），并且符合MMAP的阀值以及系统支持MMAP，则采用MMAP分配
     mp_.n_mmaps_max = 65536
     mp_.mmap_threshold = 128*1024
     需要goto try_mmap，才会进入MMAP分配逻辑，该场景则直接返回MMAP分配的内存chunk，不调整Top chunk
   */

    if (av == NULL
        || ((unsigned long)(nb) >= (unsigned long)(mp_.mmap_threshold)
            && (mp_.n_mmaps < mp_.n_mmaps_max)))
    {
        char* mm;
        // 需要与hp_pagesize对齐，并在标志上添加hp_flags
        if (mp_.hp_pagesize > 0 && nb >= mp_.hp_pagesize)
        {
            /* There is no need to issue the THP madvise call if Huge Pages
          are used directly.  */
          // 直接调用mmap，使用mp_.hp_pagesize和对应的标志
            mm = sysmalloc_mmap(nb, mp_.hp_pagesize, mp_.hp_flags, av);
            if (mm != MAP_FAILED)
                return mm;
        }
        mm = sysmalloc_mmap(nb, pagesize, 0, av);
        if (mm != MAP_FAILED)
            return mm;
        tried_mmap = true;
    }

    /* There are no usable arenas and mmap also failed.  */
    // 如果没有可用的分配区直接失败
    if (av == NULL)
        return 0;

    /* Record incoming configuration of top */
    // 记录top传入的值，前后边界和大小
    old_top = av->top;
    old_size = chunksize(old_top);
    old_end = (char*)(chunk_at_offset(old_top, old_size));

    // brk=第一次分配返回值 ；snd_brk=第二次分配返回值
    brk = snd_brk = (char*)(MORECORE_FAILURE);

    /*
    If not the first time through, we require old_size to be
    at least MINSIZE and to have prev_inuse set.
    如果不是第一次，我们要求old_size至少为MINSIZE并设置prev_inuse 。
    */

    assert((old_top == initial_top(av) && old_size == 0)
        || ((unsigned long)(old_size) >= MINSIZE && prev_inuse(old_top)
            && ((unsigned long)old_end & (pagesize - 1)) == 0));

    /* Precondition: not enough current space to satisfy nb request */
    assert((unsigned long)(old_size) < (unsigned long)(nb + MINSIZE));
    // 不是主分区，通过heap_info方式获取堆信息结构，并且通过MMAP分配内存
    if (av != &main_arena)
    {
        heap_info* old_heap, * heap;
        size_t old_heap_size;

        /* First try to extend the current heap. */
      // 通过heap_for_ptr，获取当前的heap的数据结构
        old_heap = heap_for_ptr(old_top);
        old_heap_size = old_heap->size;

        /**
         * 如果剩余的空间不足，首先通过grow_heap尝试heap区的扩容
         * grow_heap函数进行扩容，有分配页大小的限制，分配页的大小要小于HEAP_MAX_SIZE
         */
        if ((long)(MINSIZE + nb - old_size) > 0
            && grow_heap(old_heap, MINSIZE + nb - old_size) == 0)
        {
            // 扩容成功
            av->system_mem += old_heap->size - old_heap_size; // 变更系统内存记录，增加部分为新的heap的size减去老的heap的值
            // 设置p->mchunk_size（size+使用中）
            set_head(old_top, (((char*)old_heap + old_heap->size)
                - (char*)old_top)
                | PREV_INUSE);
        }
        // 创建一个新的堆,并将av->top指向到新的heap
        // new_heap通过MMAP方式分配一块内存， 32位系统每次映射1M，64位系统每次映射64M
        else if ((heap
            = new_heap(nb + (MINSIZE + sizeof(*heap)), mp_.top_pad)))
        {
            /* Use a newly allocated heap.  */
            // 设置堆的分配区
            heap->ar_ptr = av;
            // 设置堆的prev指针，和原来的堆组成一个链表
            heap->prev = old_heap;
            // 增加分配区从系统分配的内存
            av->system_mem += heap->size;
            /* Set up the new top.  */
            // 设置堆可以使用的内存开始地址
            top(av) = chunk_at_offset(heap, sizeof(*heap));
            // 设置p->mchunk_size
            set_head(top(av), (heap->size - sizeof(*heap)) | PREV_INUSE);

            /* Setup fencepost and free the old top chunk with a multiple of
          MALLOC_ALIGNMENT in size. */
          /* The fencepost takes at least MINSIZE bytes, because it might
        become the top chunk again later.  Note that a footer is set
        up, too, although the chunk is marked in use. */
        // 计算老的堆大小
            old_size = (old_size - MINSIZE) & ~MALLOC_ALIGN_MASK;
            // 设置老的块的p->mchunk_size
            set_head(chunk_at_offset(old_top, old_size + CHUNK_HDR_SZ),
                0 | PREV_INUSE);
            // 如果原来top的大小大于MINSIZE，
            if (old_size >= MINSIZE)
            {
                set_head(chunk_at_offset(old_top, old_size),
                    CHUNK_HDR_SZ | PREV_INUSE);
                set_foot(chunk_at_offset(old_top, old_size), CHUNK_HDR_SZ);
                set_head(old_top, old_size | PREV_INUSE | NON_MAIN_ARENA);
                // 释放old_top
                _int_free(av, old_top, 1);
            }
            else
            {
                // 设置p->mchunk_size （size+PREV使用中）
                set_head(old_top, (old_size + CHUNK_HDR_SZ) | PREV_INUSE);
                // 设置后一个内存块的mchunk_prev_size
                set_foot(old_top, (old_size + CHUNK_HDR_SZ));
            }
        }
        // 如果创建新的堆失败，直接调用mmap分配一块内存
        else if (!tried_mmap)
        {
            /* We can at least try to use to mmap memory.  If new_heap fails
            it is unlikely that trying to allocate huge pages will
            succeed.*/
            char* mm = sysmalloc_mmap(nb, pagesize, 0, av);
            if (mm != MAP_FAILED)
                return mm;
        }
    }
    // 主分配区，则直接扩容top_chunk(brk和mmap的方式都有)
    else /* av == main_arena */
    {
        /* Request enough space for nb + pad + overhead */
        // 计算最终需要的空间，mp_.top_pad初始化或扩展堆的时候需要多申请的内存大小
        size = nb + mp_.top_pad + MINSIZE;
        /*
          If contiguous, we can subtract out existing space that we hope to
          combine with new space. We add it back later only if
          we don't actually get contiguous space.
          如果是连续的，老的空间+一部分新的空间可以得到需要的空间。仅当我们实际上没有获得连续空间时，我们才会稍后将其添加回来。
          contiguous:用于判断是否为连续brk分配，主分配区先减去已经存在的top空间,再向操作系统申请
          如果是连续brk分配，可以减去原有的old_size如果是非连续brk分配，后续还会降old_size加回去
          由于old_size小于nb，top_pad又是128K，所以减去之后也有足够的空间存储
        */
        // 如果是连续空间，
        if (contiguous(av))
            size -= old_size;

        /*
           Round to a multiple of page size or huge page size.
           If MORECORE is not contiguous, this ensures that we only call it
           with whole-page arguments.  And if MORECORE is contiguous and
           this is not first time through, this preserves page-alignment of
           previous calls. Otherwise, we correct to page-align below.
         */

#ifdef MADV_HUGEPAGE
         /* Defined in brk.c.  */
        extern void* __curbrk;
        if (__glibc_unlikely(mp_.thp_pagesize != 0))
        {
            uintptr_t top
                = ALIGN_UP((uintptr_t)__curbrk + size, mp_.thp_pagesize);
            size = top - (uintptr_t)__curbrk;
        }
        else
#endif
            // 按照页面进行对齐
            size = ALIGN_UP(size, GLRO(dl_pagesize));

        /*
           Don't try to call MORECORE if argument is so big as to appear
           negative. Note that since mmap takes size_t arg, it may succeed
           below even if we cannot call MORECORE.
         */
         //size大于0，然后通过系统调用(sbrk)分配size大小的内存
        if (size > 0)
        {
            brk = (char*)(MORECORE(size));
            if (brk != (char*)(MORECORE_FAILURE))
                madvise_thp(brk, size);
            LIBC_PROBE(memory_sbrk_more, 2, brk, size);
        }
        // 如果调用brk分配失败，使用MMap分配
        if (brk == (char*)(MORECORE_FAILURE))
        {
            /*
              If have mmap, try using it as a backup when MORECORE fails or
              cannot be used. This is worth doing on systems that have
              "holes" in address space, so sbrk cannot extend to give
              contiguous space, but space is available elsewhere.  Note that
              we ignore mmap max count and threshold limits, since the space
              will not be used as a segregated mmap region.
            */

            char* mbrk = MAP_FAILED;
            // 
            if (mp_.hp_pagesize > 0)
                mbrk = sysmalloc_mmap_fallback(
                    &size, nb, old_size, mp_.hp_pagesize, mp_.hp_pagesize,
                    mp_.hp_flags, av);
            // 如果分配失败，只分配对应的大小
            if (mbrk == MAP_FAILED)
                mbrk = sysmalloc_mmap_fallback(&size, nb, old_size,
                    MMAP_AS_MORECORE_SIZE, pagesize,
                    0, av);
            // 分配成功
            if (mbrk != MAP_FAILED)
            {
                /* We do not need, and cannot use, another sbrk call to find
                * end */
                // brk设置为内存的头部
                brk = mbrk;
                // snd_brk设置为内存的尾部
                snd_brk = brk + size;
            }
        }
        // 分配成功
        if (brk != (char*)(MORECORE_FAILURE))
        {
            if (mp_.sbrk_base == 0)
                mp_.sbrk_base = brk;
            av->system_mem += size;

            /*
            If MORECORE extends previous space, we can likewise extend top
            size.
            */
            // 判断是否是通过brk分配的用brk分配的，并且地址是连续的，则直接更新Top chunk即可（就是Top chunk的扩容）
            if (brk == old_end && snd_brk == (char*)(MORECORE_FAILURE))
                set_head(old_top, (size + old_size) | PREV_INUSE);

            else if (contiguous(av) && old_size && brk < old_end)
                /* Oops!  Someone else killed our space..  Can't touch anything.
             */
                malloc_printerr("break adjusted to free malloc space");

            /*
               Otherwise, make adjustments:

             * If the first time through or noncontiguous, we need to call
             sbrk just to find out where the end of memory lies.

             * We need to ensure that all returned chunks from malloc will
             meet MALLOC_ALIGNMENT

             * If there was an intervening foreign sbrk, we need to adjust
             sbrk request size to account for fact that we will not be able to
            combine new space with existing space in old_top.

             * Almost all systems internally allocate whole pages at a time,
             in which case we might as well use the whole last page of
             request. So we allocate enough more memory to hit a page boundary
             now, which in turn causes future contiguous calls to page-align.
             */
             //新分配的内存地址大于原来的top chunk的结束地址，说明地址是不连续的，使用MMAP分配的，则无法直接扩容Top chunk
            else
            {
                front_misalign = 0;
                end_misalign = 0;
                correction = 0;
                aligned_brk = brk;

                /* handle contiguous cases */
                // 连续标记，说明是brk方式分配的
                if (contiguous(av))
                {
                    /* Count foreign sbrk as system_mem.  */
                    // Topchunk不连续，需要记录外部不连续内存的大小
                    if (old_size)
                        av->system_mem += brk - old_end;

                    /* Guarantee alignment of first new chunk made from this
                     * space */
                    // 分配的mm内存地址需要进行内存对齐，front_misalign表示对齐的地址前有多少个可以对齐的字节
                    front_misalign = (INTERNAL_SIZE_T)chunk2mem(brk)
                        & MALLOC_ALIGN_MASK;
                    if (front_misalign > 0)
                    {
                        /*
                           Skip over some bytes to arrive at an aligned
                           position. We don't need to specially mark these
                           wasted front bytes. They will never be accessed
                           anyway because prev_inuse of av->top (and any chunk
                           created from its start) is always true after
                           initialization.
                         */
                        // 跳过一些字节达到对齐的位置。我们不需要特别标记这些浪费的前字节。
                        correction = MALLOC_ALIGNMENT - front_misalign;
                        aligned_brk += correction;
                    }

                    /*
                   If this isn't adjacent to existing space, then we will
                   not be able to merge with old_top space, so must add to
                   2nd request.
                     */
                    // correction=新扩展出来的chunk需要继续扩容的大小 包含brk对齐后头部减少的字节 + 老的top的size
                    correction += old_size;

                    /* Extend the end address to hit a page boundary */
                    //对齐后的新的top chunk的尾部地址
                    end_misalign
                        = (INTERNAL_SIZE_T)(brk + size + correction);
                    //尾部地址需要进行一次对齐，对齐后是否需要增加字节
                    correction
                        += (ALIGN_UP(end_misalign, pagesize)) - end_misalign;

                    // 所以：correction=brk前置的对齐字节 + 老的top的size + 新的brk尾部的对齐字节
                    // aligned_brk：为调整后的新的chunk的起始地址
                    // snd_brk：2次扩容操作的返回值
                    assert(correction >= 0);
                    snd_brk = (char*)(MORECORE(correction));

                    /*
                   If can't allocate correction, try to at least find out
                   current brk.  It might be enough to proceed without
                   failing.

                   Note that if second sbrk did NOT fail, we assume that
                   space is contiguous with first sbrk. This is a safe
                   assumption unless program is multithreaded but doesn't
                   use locks and a foreign sbrk occurred between our first
                   and second calls.
                     */
                    //扩容失败
                    if (snd_brk == (char*)(MORECORE_FAILURE))
                    {
                        correction = 0;
                        snd_brk = (char*)(MORECORE(0));
                    }
                    else
                        madvise_thp(snd_brk, correction);
                }

                /* handle non-contiguous cases */
                // 没有连续标记的Case，使用MMAP分配的情况
                else
                {
                    if (MALLOC_ALIGNMENT == CHUNK_HDR_SZ)
                        /* MORECORE/mmap must correctly align */
                        assert(((unsigned long)chunk2mem(brk)
                            & MALLOC_ALIGN_MASK)
                            == 0);
                    else
                    {
                        front_misalign = (INTERNAL_SIZE_T)chunk2mem(brk)
                            & MALLOC_ALIGN_MASK;
                        if (front_misalign > 0)
                        {
                            /*
                           Skip over some bytes to arrive at an aligned
                           position. We don't need to specially mark these
                           wasted front bytes. They will never be accessed
                           anyway because prev_inuse of av->top (and any
                           chunk created from its start) is always true
                           after initialization.
                             */
                            // 调整对齐，aligned_brk为调整后的起始地址
                            aligned_brk += MALLOC_ALIGNMENT - front_misalign;
                        }
                    }

                    /* Find out current end of memory */
                    // 找出当前尾部的内存
                    if (snd_brk == (char*)(MORECORE_FAILURE))
                    {
                        snd_brk = (char*)(MORECORE(0));
                    }
                }

                /* Adjust top based on results of second sbrk */
                // 设置av->top指向调整后的aligned_brk
                if (snd_brk != (char*)(MORECORE_FAILURE))
                {
                    av->top = (mchunkptr)aligned_brk;
                    set_head(av->top, (snd_brk - aligned_brk + correction)
                        | PREV_INUSE);
                    av->system_mem += correction;

                    /*
                   If not the first time through, we either have a
                   gap due to foreign sbrk or a non-contiguous region.
                   Insert a double fencepost at old_top to prevent
                   consolidation with space we don't own. These fenceposts
                   are artificial chunks that are marked as inuse and are
                   in any case too small to use.  We need two to make
                   sizes and alignments work out.
                     */
                    // 合并加工后的的old_top需要回收free
                    if (old_size != 0)
                    {
                        /*
                           Shrink old_top to insert fenceposts, keeping size a
                           multiple of MALLOC_ALIGNMENT. We know there is at
                           least enough space in old_top to do this.
                           缩小2个字节，并重新设置old_top->mchunk_size，old_size是有足够空间缩小两个字节的
                         */
                        old_size = (old_size - 2 * CHUNK_HDR_SZ)
                            & ~MALLOC_ALIGN_MASK;
                        set_head(old_top, old_size | PREV_INUSE);

                        /*
                           Note that the following assignments completely
                           overwrite old_top when old_size was previously
                           MINSIZE.  This is intentional. We need the
                           fencepost, even if old_top otherwise gets lost.
                         */
                        set_head(chunk_at_offset(old_top, old_size),
                            CHUNK_HDR_SZ | PREV_INUSE);
                        set_head(chunk_at_offset(old_top,
                            old_size + CHUNK_HDR_SZ),
                            CHUNK_HDR_SZ | PREV_INUSE);

                        /* If possible, release the rest. */
                        //释放old_top的chunk
                        if (old_size >= MINSIZE)
                        {
                            _int_free(av, old_top, 1);
                        }
                    }
                }
            }
        }
    } /* if (av !=  &main_arena) */

    if ((unsigned long)av->system_mem > (unsigned long)(av->max_system_mem))
        av->max_system_mem = av->system_mem;
    check_malloc_state(av);

    /* finally, do the allocation */
    // 最后，分配内存
    p = av->top;
    size = chunksize(p);

    /* check that one of the above allocation paths succeeded */
    // 如果Top chunk的大小大于分配的值，则进行切割分配操作，切割出来的分配出去，剩余的remainder变成Top chunk
    if ((unsigned long)(size) >= (unsigned long)(nb + MINSIZE))
    { 
        // 剩下的变成新的top
        remainder_size = size - nb;
        remainder = chunk_at_offset(p, nb);
        av->top = remainder;
        // 设置标志
        set_head(p,
            nb | PREV_INUSE | (av != &main_arena ? NON_MAIN_ARENA : 0));
        set_head(remainder, remainder_size | PREV_INUSE);
        check_malloced_chunk(av, p, nb);
        // 返回
        return chunk2mem(p);
    }

    /* catch all failure paths */
    __set_errno(ENOMEM);
    return 0;
}

/*
   systrim is an inverse of sorts to sysmalloc.  It gives memory back
   to the system (via negative arguments to sbrk) if there is unused
   memory at the `high' end of the malloc pool. It is called
   automatically by free() when top space exceeds the trim
   threshold. It is also called by the public malloc_trim routine.  It
   returns 1 if it actually released any memory, else 0.
 */

static int
systrim(size_t pad, mstate av)
{
    long top_size;	/* Amount of top-most memory */
    long extra;	/* Amount to release */
    long released;	/* Amount actually released */
    char* current_brk; /* address returned by pre-check sbrk call */
    char* new_brk;	/* address returned by post-check sbrk call */
    long top_area;

    top_size = chunksize(av->top);

    top_area = top_size - MINSIZE - 1;
    if (top_area <= pad)
        return 0;

    /* Release in pagesize units and round down to the nearest page.  */
#ifdef MADV_HUGEPAGE
    if (__glibc_unlikely(mp_.thp_pagesize != 0))
        extra = ALIGN_DOWN(top_area - pad, mp_.thp_pagesize);
    else
#endif
        extra = ALIGN_DOWN(top_area - pad, GLRO(dl_pagesize));

    if (extra == 0)
        return 0;

    /*
   Only proceed if end of memory is where we last set it.
   This avoids problems if there were foreign sbrk calls.
     */
    current_brk = (char*)(MORECORE(0));
    if (current_brk == (char*)(av->top) + top_size)
    {
        /*
           Attempt to release memory. We ignore MORECORE return value,
           and instead call again to find out where new end of memory is.
           This avoids problems if first call releases less than we asked,
           of if failure somehow altered brk value. (We could still
           encounter problems if it altered brk in some very bad way,
           but the only thing we can do is adjust anyway, which will cause
           some downstream failure.)
         */

        MORECORE(-extra);
        new_brk = (char*)(MORECORE(0));

        LIBC_PROBE(memory_sbrk_less, 2, new_brk, extra);

        if (new_brk != (char*)MORECORE_FAILURE)
        {
            released = (long)(current_brk - new_brk);

            if (released != 0)
            {
                /* Success. Adjust top. */
                av->system_mem -= released;
                set_head(av->top, (top_size - released) | PREV_INUSE);
                check_malloc_state(av);
                return 1;
            }
        }
    }
    return 0;
}

// 在malloc的时候，如果遇到try map方式实现MMAP方式的分配，都会有此标记。在通过Try mmap的方法中，
// 会在分配chunk的时候进行对齐操作，并且对齐后因为前置会有空余字节，则会生成一个chunk，
// 所以在释放free的时候也需要计算前一个chunk。
static void
munmap_chunk(mchunkptr p)
{
    size_t pagesize = GLRO(dl_pagesize);
    INTERNAL_SIZE_T size = chunksize(p);

    assert(chunk_is_mmapped(p));

    // 在通过Try mmap的方法中，会在分配chunk的时候进行对齐操作，并且对齐后因为前置会有空余字节，则会生成一个chunk，free的时候要计算此chunk

    // 获取chunk的对应的数据内存地址
    uintptr_t mem = (uintptr_t)chunk2mem(p);
    // 指向前一个chunk的地址
    uintptr_t block = (uintptr_t)p - prev_size(p);
    // 前一个空闲的大小
    size_t total_size = prev_size(p) + size;
    /* Unfortunately we have to do the compilers job by hand here.  Normally
   we would test BLOCK and TOTAL-SIZE separately for compliance with the
   page size.  But gcc does not recognize the optimization possibility
   (in the moment at least) so we combine the two values into one before
   the bit test.  */
    if (__glibc_unlikely((block | total_size) & (pagesize - 1)) != 0
        || __glibc_unlikely(!powerof2(mem & (pagesize - 1))))
        malloc_printerr("munmap_chunk(): invalid pointer");

    atomic_fetch_add_relaxed(&mp_.n_mmaps, -1);
    atomic_fetch_add_relaxed(&mp_.mmapped_mem, -total_size);

    /* If munmap failed the process virtual memory address space is in a
   bad shape.  Just leave the block hanging around, the process will
   terminate shortly anyway since not much can be done.  */
   //释放MUNMAP，包含当前MMAP的chunk，以及对齐前置的chunk
    __munmap((char*)block, total_size);
}

#if HAVE_MREMAP

static mchunkptr
mremap_chunk(mchunkptr p, size_t new_size)
{
    size_t pagesize = GLRO(dl_pagesize);
    INTERNAL_SIZE_T offset = prev_size(p);
    INTERNAL_SIZE_T size = chunksize(p);
    char* cp;

    assert(chunk_is_mmapped(p));

    uintptr_t block = (uintptr_t)p - offset;
    uintptr_t mem = (uintptr_t)chunk2mem(p);
    size_t total_size = offset + size;
    if (__glibc_unlikely((block | total_size) & (pagesize - 1)) != 0
        || __glibc_unlikely(!powerof2(mem & (pagesize - 1))))
        malloc_printerr("mremap_chunk(): invalid pointer");

    /* Note the extra SIZE_SZ overhead as in mmap_chunk(). */
    new_size = ALIGN_UP(new_size + offset + SIZE_SZ, pagesize);

    /* No need to remap if the number of pages does not change.  */
    if (total_size == new_size)
        return p;

    cp = (char*)__mremap((char*)block, total_size, new_size,
        MREMAP_MAYMOVE);

    if (cp == MAP_FAILED)
        return 0;

    madvise_thp(cp, new_size);

    p = (mchunkptr)(cp + offset);

    assert(aligned_OK(chunk2mem(p)));

    assert(prev_size(p) == offset);
    set_head(p, (new_size - offset) | IS_MMAPPED);

    INTERNAL_SIZE_T new;
    new = atomic_fetch_add_relaxed(&mp_.mmapped_mem,
        new_size - size - offset)
        + new_size - size - offset;
    atomic_max(&mp_.max_mmapped_mem, new);
    return p;
}
#endif /* HAVE_MREMAP */

/*------------------------ Public wrappers.
 * --------------------------------*/

#if USE_TCACHE

 /* We overlay this structure on the user-data portion of a chunk when
    the chunk is stored in the per-thread cache.  */
    //  tcache的基本结构，通过单项链表连接
typedef struct tcache_entry
{
    // 指向下一个tcache项的指针
    struct tcache_entry* next;
    // 新增防止tcache多次释放的机制
    /* This field exists to detect double frees.  */
    uintptr_t key;
} tcache_entry;

/* There is one of these for each thread, which contains the
   per-thread cache (hence "tcache_perthread_struct").  Keeping
   overall size low is mildly important.  Note that COUNTS and ENTRIES
   are redundant (we could have just counted the linked list each
   time), this is for performance reasons.  */
   // 管理tcache的结构
typedef struct tcache_perthread_struct
{
    // 统计数组中每个下标有多少对应的chunk ，TCHACHE_MAX_BINS的值一般是64
    uint16_t counts[TCACHE_MAX_BINS];
    // 指向不同tcache的指针数组
    tcache_entry* entries[TCACHE_MAX_BINS];
} tcache_perthread_struct;

static __thread bool tcache_shutting_down = false;
static __thread tcache_perthread_struct* tcache = NULL;

/* Process-wide key to try and catch a double-free in the same thread.  */
static uintptr_t tcache_key;

/* The value of tcache_key does not really have to be a cryptographically
   secure random number.  It only needs to be arbitrary enough so that it
   does not collide with values present in applications.  If a collision
   does happen consistently enough, it could cause a degradation in
   performance since the entire list is checked to check if the block indeed
   has been freed the second time.  The odds of this happening are
   exceedingly low though, about 1 in 2^wordsize.  There is probably a
   higher chance of the performance degradation being due to a double free
   where the first free happened in a different thread; that's a case this
   check does not cover.  */
static void
tcache_key_initialize(void)
{
    /* We need to use the _nostatus version here, see BZ 29624.  */
    if (__getrandom_nocancel_nostatus(&tcache_key, sizeof(tcache_key),
        GRND_NONBLOCK)
        != sizeof(tcache_key))
    {
        tcache_key = random_bits();
#  if __WORDSIZE == 64
        tcache_key = (tcache_key << 32) | random_bits();
#  endif
    }
}

/* Caller must ensure that we know tc_idx is valid and there's room
   for more chunks.  */
static __always_inline void
tcache_put(mchunkptr chunk, size_t tc_idx)
{
    tcache_entry* e = (tcache_entry*)chunk2mem(chunk);

    /* Mark this chunk as "in the tcache" so the test in _int_free will
   detect a double free.  */
    e->key = tcache_key;

    e->next = PROTECT_PTR(&e->next, tcache->entries[tc_idx]);
    tcache->entries[tc_idx] = e;
    ++(tcache->counts[tc_idx]);
}

/* Caller must ensure that we know tc_idx is valid and there's
   available chunks to remove.  Removes chunk from the middle of the
   list.  */
static __always_inline void*
tcache_get_n(size_t tc_idx, tcache_entry** ep)
{
    tcache_entry* e;
    if (ep == &(tcache->entries[tc_idx]))
        e = *ep;
    else
        e = REVEAL_PTR(*ep);

    if (__glibc_unlikely(!aligned_OK(e)))
        malloc_printerr("malloc(): unaligned tcache chunk detected");

    if (ep == &(tcache->entries[tc_idx]))
        *ep = REVEAL_PTR(e->next);
    else
        *ep = PROTECT_PTR(ep, REVEAL_PTR(e->next));

    --(tcache->counts[tc_idx]);
    e->key = 0;
    return (void*)e;
}

/* Like the above, but removes from the head of the list.  */
// 获得tc_idx索引对应的头节点，并且将其从列表中脱离出来
static __always_inline void*
tcache_get(size_t tc_idx)
{
    return tcache_get_n(tc_idx, &tcache->entries[tc_idx]);
}

/* Iterates through the tcache linked list.  */
static __always_inline tcache_entry*
tcache_next(tcache_entry* e)
{
    return (tcache_entry*)REVEAL_PTR(e->next);
}

static void
tcache_thread_shutdown(void)
{
    int i;
    tcache_perthread_struct* tcache_tmp = tcache;

    tcache_shutting_down = true;
    // tcahce不存在的情况下直接返回
    if (!tcache)
        return;

    /* Disable the tcache and prevent it from being reinitialized.  */
    // 禁用tcache，防止它被重新初始化
    tcache = NULL;

    /* Free all of the entries and the tcache itself back to the arena
   heap for coalescing.  */
   // 释放所有的tcache ，以便进行合并 */
   // 外层for循环遍历tcache指针数组，数组的每个下标对应不同大小的 tcach
    for (i = 0; i < TCACHE_MAX_BINS; ++i)
    {
        // 内层while循环遍历整个tcache点链表，也就是相同大小的tcache
        while (tcache_tmp->entries[i])
        {
            tcache_entry* e = tcache_tmp->entries[i];
            if (__glibc_unlikely(!aligned_OK(e)))
                malloc_printerr("tcache_thread_shutdown(): "
                    "unaligned tcache chunk detected");
            tcache_tmp->entries[i] = REVEAL_PTR(e->next);
            // 依次释放链表上的tcache
            __libc_free(e);
        }
    }
    // 将管理tcache的结构体也释放掉
    __libc_free(tcache_tmp);
}

static void
tcache_init(void)
{
    mstate ar_ptr;
    void* victim = 0;
    // 计算 tcahce_perthread_struct 结构大小
    const size_t bytes = sizeof(tcache_perthread_struct);

    // 是否被禁用，如果禁用就直接返回
    if (tcache_shutting_down)
        return;

    // 得到分配区
    arena_get(ar_ptr, bytes);
    // 分配内存
    victim = _int_malloc(ar_ptr, bytes);
    // 如果分配区存在，但是分配内存失败
    if (!victim && ar_ptr != NULL)
    {
        // 尝试切换分配区，重新分配
        ar_ptr = arena_get_retry(ar_ptr, bytes);
        victim = _int_malloc(ar_ptr, bytes);
    }

    if (ar_ptr != NULL)
        __libc_lock_unlock(ar_ptr->mutex);

    /* In a low memory situation, we may not be able to allocate memory
   - in which case, we just keep trying later.  However, we
   typically do this very early, so either there is sufficient
   memory, or there isn't enough memory to do non-trivial
   allocations anyway.  */
   // tcache 分配好后，将 tcache 处的内存初始化为 0
    if (victim)
    {
        tcache = (tcache_perthread_struct*)victim;
        memset(tcache, 0, sizeof(tcache_perthread_struct));
    }
}

#  define MAYBE_INIT_TCACHE()                                                 \
    if (__glibc_unlikely (tcache == NULL))                                    \
      tcache_init ();

#else /* !USE_TCACHE */
#  define MAYBE_INIT_TCACHE()

static void
tcache_thread_shutdown(void)
{
    /* Nothing to do if there is no thread cache.  */
}

#endif /* !USE_TCACHE  */

#if IS_IN(libc)
void*
__libc_malloc(size_t bytes)
{
    mstate ar_ptr;
    void* victim;

    _Static_assert (PTRDIFF_MAX <= SIZE_MAX / 2,
        "PTRDIFF_MAX is not more than half of SIZE_MAX");

    if (!__malloc_initialized)
        ptmalloc_init();
#  if USE_TCACHE
    /* int_free also calls request2size, be careful to not pad twice.  */
    /* 判断请求分配字节的大小，在 64 位的情况下，bytes 不能大于
     * 0x7fffffffffffffff；*/
     /* 在 32 位的情况下，bytes 不能超过 0x7fffffff。函数中也会调用
      * request2size 来 */
      /* 计算 bytes 数据需要分配的内存大小，当 bytes 数据的大小比最小 chunk
       * 要还小时，*/
       /* 按最小 chunk 的大小分配；当 bytes 数据的大小比最小 chunk
        * 大时，则分配满足内存 */
        /* 对齐要求的最小大小。将分配的大小赋值给 tbytes 返回。 */
    size_t tbytes = checked_request2size(bytes);
    if (tbytes == 0)
    {
        __set_errno(ENOMEM);
        return NULL;
    }
    // 计算tbytes大小所对应的tcache下标
    size_t tc_idx = csize2tidx(tbytes);
    // 如果tcache还没有被创建，则调用tcache_init()初始化tcache
    MAYBE_INIT_TCACHE();

    DIAG_PUSH_NEEDS_COMMENT;
    /* 这里的 mp_是malloc_par 结构 */
    /* 判断 idx 是否在tcache bins的范围内 */
    /* 判断 tcache 是否存在 */
    /* 判断 idx 对应的 tcache bins 中是否有空闲 tcache chunk */
    if (tc_idx < mp_.tcache_bins && tcache != NULL
        && tcache->counts[tc_idx] > 0)
    {
        victim = tcache_get(tc_idx);
        return tag_new_usable(victim);
    }
    DIAG_POP_NEEDS_COMMENT;
#  endif
    // 单线程情况下
    if (SINGLE_THREAD_P)
    {
        // 调用_int_malloc函数分配内存
        victim = tag_new_usable(_int_malloc(&main_arena, bytes));
        // 当前chunk是从mmap分配的或当前chunk是从主分配区分配的
        assert(!victim || chunk_is_mmapped(mem2chunk(victim))
            || &main_arena == arena_for_chunk(mem2chunk(victim)));
        return victim;
    }
    // 多线程情况下
    // 获取分配区
    arena_get(ar_ptr, bytes);
    // 调用_int_malloc函数分配内存
    victim = _int_malloc(ar_ptr, bytes);
    /* Retry with another arena only if we were able to find a usable arena
   before.  */
   /* 如果成功获取分配区，但是分配内存失败，可能是mmap区域的内存耗尽等多种原因
    */
    /* 不同的原因有不同的解决方法，比如更换分配区等等 */
    /* 所以这里重新进行了获取分配区和分配内存操作，确保内存分配成功 */
    if (!victim && ar_ptr != NULL)
    {
        LIBC_PROBE(memory_malloc_retry, 1, bytes);
        // 更换分配区
        ar_ptr = arena_get_retry(ar_ptr, bytes);
        victim = _int_malloc(ar_ptr, bytes);
    }
    // 释放锁
    if (ar_ptr != NULL)
        __libc_lock_unlock(ar_ptr->mutex);

    victim = tag_new_usable(victim);

    assert(!victim || chunk_is_mmapped(mem2chunk(victim))
        || ar_ptr == arena_for_chunk(mem2chunk(victim)));
    return victim;
}
libc_hidden_def(__libc_malloc)

// 释放内存
void __libc_free(void* mem)
{
    mstate ar_ptr;
    mchunkptr p; /* chunk corresponding to mem */

    if (mem == 0) /* free(0) has no effect */
        return;

    /* Quickly check that the freed pointer matches the tag for the memory.
   This gives a useful double-free detection.  */
    if (__glibc_unlikely(mtag_enabled))
        *(volatile char*)mem;

    int err = errno;
    // 偏移到chunk的内存地址
    p = mem2chunk(mem);
    /*步骤1：如果是MMAP分配的，则调用munmap_chunk进行chunk的释放操作
    * 获取状态值，是否是MMAP分配的 mchunk_size字段中的标记位 IS_MMAPPED*/
    if (chunk_is_mmapped(p)) /* release mmapped memory. */
    {
        /* See if the dynamic brk/mmap threshold needs adjusting.
           Dumped fake mmapped chunks do not affect the threshold.  */
        // 如果mmap_threshold是动态的，并且当前chunk的大小大于mmap_threshold，
        // 并且少于默认的最大的THRESHOLD
        if (!mp_.no_dyn_threshold && chunksize_nomask(p) > mp_.mmap_threshold
            && chunksize_nomask(p) <= DEFAULT_MMAP_THRESHOLD_MAX)
        {
            mp_.mmap_threshold = chunksize(p);
            mp_.trim_threshold = 2 * mp_.mmap_threshold;
            LIBC_PROBE(memory_mallopt_free_dyn_thresholds, 2,
                mp_.mmap_threshold, mp_.trim_threshold);
        }

        munmap_chunk(p);
    }
    else
    {
        MAYBE_INIT_TCACHE();

        /* Mark the chunk as belonging to the library again.  */
        (void)tag_region(chunk2mem(p), memsize(p));

        ar_ptr = arena_for_chunk(p);
        _int_free(ar_ptr, p, 0);
    }

    __set_errno(err);
}
libc_hidden_def(__libc_free)

void* __libc_realloc(void* oldmem, size_t bytes)
{
    mstate ar_ptr;
    INTERNAL_SIZE_T nb; /* padded request size */

    void* newp; /* chunk to return */
    // 如果还没有初始化，则初始化
    if (!__malloc_initialized)
        ptmalloc_init();

#  if REALLOC_ZERO_BYTES_FREES
    // 如果目标内存长度为0，相当于释放原来的内存
    if (bytes == 0 && oldmem != NULL)
    {
        __libc_free(oldmem);
        return 0;
    }
#  endif

    /* realloc of null is supposed to be same as malloc */
    // 如果老的内存指针为空，相当于重新分配一块内存
    if (oldmem == 0)
        return __libc_malloc(bytes);

    /* Perform a quick check to ensure that the pointer's tag matches the
   memory's tag.  */
    if (__glibc_unlikely(mtag_enabled))
        *(volatile char*)oldmem;

    /* chunk corresponding to oldmem */
    //获取老的chunk指针地址
    const mchunkptr oldp = mem2chunk(oldmem);

    /* Return the chunk as is if the request grows within usable bytes,
   typically into the alignment padding.  We want to avoid reusing the
   block for shrinkages because it ends up unnecessarily fragmenting the
   address space. This is also why the heuristic misses alignment padding
   for THP for now.  */
   // 得到内存块的可用大小
    size_t usable = musable(oldmem);
    if (bytes <= usable)
    {
        size_t difference = usable - bytes;
        // 如果尺寸正好可以，直接返回
        if ((unsigned long)difference < 2 * sizeof(INTERNAL_SIZE_T)
            || (chunk_is_mmapped(oldp) && difference <= GLRO(dl_pagesize)))
            return oldmem;
    }

    /* its size */
    // 得到老的快的大小
    const INTERNAL_SIZE_T oldsize = chunksize(oldp);

    // 如果是mmap分配的，不需要分配区
    if (chunk_is_mmapped(oldp))
        ar_ptr = NULL;
    else
    {
        MAYBE_INIT_TCACHE();
        // 得到内存块的分配区
        ar_ptr = arena_for_chunk(oldp);
    }

    /* Little security check which won't hurt performance: the allocator
   never wraps around at the end of the address space.  Therefore
   we can exclude some size values which might appear here by
   accident or by "design" from some intruder.  */
    if ((__builtin_expect((uintptr_t)oldp > (uintptr_t)-oldsize, 0)
        || __builtin_expect(misaligned_chunk(oldp), 0)))
        malloc_printerr("realloc(): invalid pointer");

    // 检查bytes是否在合法区间内，最大小于<2147483647,并调用request2size函数将bytes进行对齐，
    // 最终得到对齐的nb大小内存申请容量
    nb = checked_request2size(bytes);
    if (nb == 0)
    {
        __set_errno(ENOMEM);
        return NULL;
    }

    // 如果是mmap的方式分配的
    if (chunk_is_mmapped(oldp))
    {
        void* newmem;

#  if HAVE_MREMAP
        // 扩大/缩小现有内存映射
        newp = mremap_chunk(oldp, nb);
        if (newp)
        {
            // 成功了，就直接返回
            void* newmem = chunk2mem_tag(newp);
            /* Give the new block a different tag.  This helps to ensure
           that stale handles to the previous mapping are not
           reused.  There's a performance hit for both us and the
           caller for doing this, so we might want to
           reconsider.  */
            return tag_new_usable(newmem);
        }
#  endif
        /* Note the extra SIZE_SZ overhead. */
        // 如果原来的内存已经满足需求，则直接返回
        if (oldsize - SIZE_SZ >= nb)
            return oldmem; /* do nothing */

        /* Must alloc, copy, free. */
        // 调用分配函数
        newmem = __libc_malloc(bytes);
        if (newmem == 0)
            return 0; /* propagate failure */
        // 拷贝数据
        memcpy(newmem, oldmem, oldsize - CHUNK_HDR_SZ);
        // 释放原来的内存
        munmap_chunk(oldp);
        return newmem;
    }
    // 单线程，尝试扩展
    if (SINGLE_THREAD_P)
    {
        newp = _int_realloc(ar_ptr, oldp, oldsize, nb);
        assert(!newp || chunk_is_mmapped(mem2chunk(newp))
            || ar_ptr == arena_for_chunk(mem2chunk(newp)));

        return newp;
    }
    // 多线程模式下，非MMAP分配方式

    // 加锁
    __libc_lock_lock(ar_ptr->mutex);
    // 尝试扩展
    newp = _int_realloc(ar_ptr, oldp, oldsize, nb);

    __libc_lock_unlock(ar_ptr->mutex);
    assert(!newp || chunk_is_mmapped(mem2chunk(newp))
        || ar_ptr == arena_for_chunk(mem2chunk(newp)));

    // 不成功，直接调用__libc_malloc直接分配
    if (newp == NULL)
    {
        /* Try harder to allocate memory in other arenas.  */
        LIBC_PROBE(memory_realloc_retry, 2, bytes, oldmem);
        newp = __libc_malloc(bytes);
        // 分配成功
        if (newp != NULL)
        {
            size_t sz = memsize(oldp);
            // 将原数据拷贝进去
            memcpy(newp, oldmem, sz);
            (void)tag_region(chunk2mem(oldp), sz);
            _int_free(ar_ptr, oldp, 0);
        }
    }

    return newp;
}
libc_hidden_def(__libc_realloc)

void* __libc_memalign(size_t alignment, size_t bytes)
{
    if (!__malloc_initialized)
        ptmalloc_init();

    void* address = RETURN_ADDRESS(0);
    return _mid_memalign(alignment, bytes, address);
}
libc_hidden_def(__libc_memalign)

/* For ISO C17.  */
void* weak_function aligned_alloc(size_t alignment, size_t bytes)
{
    if (!__malloc_initialized)
        ptmalloc_init();

    /* Similar to memalign, but starting with ISO C17 the standard
   requires an error for alignments that are not supported by the
   implementation.  Valid alignments for the current implementation
   are non-negative powers of two.  */
    if (!powerof2(alignment) || alignment == 0)
    {
        __set_errno(EINVAL);
        return 0;
    }

    void* address = RETURN_ADDRESS(0);
    return _mid_memalign(alignment, bytes, address);
}

static void*
_mid_memalign(size_t alignment, size_t bytes, void* address)
{
    mstate ar_ptr;
    void* p;

    /* If we need less alignment than we give anyway, just relay to malloc. */
    if (alignment <= MALLOC_ALIGNMENT)
        return __libc_malloc(bytes);

    /* Otherwise, ensure that it is at least a minimum chunk size */
    if (alignment < MINSIZE)
        alignment = MINSIZE;

    /* If the alignment is greater than SIZE_MAX / 2 + 1 it cannot be a
   power of 2 and will cause overflow in the check below.  */
    if (alignment > SIZE_MAX / 2 + 1)
    {
        __set_errno(EINVAL);
        return 0;
    }

    /* Make sure alignment is power of 2.  */
    if (!powerof2(alignment))
    {
        size_t a = MALLOC_ALIGNMENT * 2;
        while (a < alignment)
            a <<= 1;
        alignment = a;
    }

#  if USE_TCACHE
    {
        size_t tbytes;
        tbytes = checked_request2size(bytes);
        if (tbytes == 0)
        {
            __set_errno(ENOMEM);
            return NULL;
        }
        size_t tc_idx = csize2tidx(tbytes);

        if (tc_idx < mp_.tcache_bins && tcache != NULL
            && tcache->counts[tc_idx] > 0)
        {
            /* The tcache itself isn't encoded, but the chain is.  */
            tcache_entry** tep = &tcache->entries[tc_idx];
            tcache_entry* te = *tep;
            while (te != NULL && !PTR_IS_ALIGNED(te, alignment))
            {
                tep = &(te->next);
                te = tcache_next(te);
            }
            if (te != NULL)
            {
                void* victim = tcache_get_n(tc_idx, tep);
                return tag_new_usable(victim);
            }
        }
    }
#  endif

    if (SINGLE_THREAD_P)
    {
        p = _int_memalign(&main_arena, alignment, bytes);
        assert(!p || chunk_is_mmapped(mem2chunk(p))
            || &main_arena == arena_for_chunk(mem2chunk(p)));
        return tag_new_usable(p);
    }

    arena_get(ar_ptr, bytes + alignment + MINSIZE);

    p = _int_memalign(ar_ptr, alignment, bytes);
    if (!p && ar_ptr != NULL)
    {
        LIBC_PROBE(memory_memalign_retry, 2, bytes, alignment);
        ar_ptr = arena_get_retry(ar_ptr, bytes);
        p = _int_memalign(ar_ptr, alignment, bytes);
    }

    if (ar_ptr != NULL)
        __libc_lock_unlock(ar_ptr->mutex);

    assert(!p || chunk_is_mmapped(mem2chunk(p))
        || ar_ptr == arena_for_chunk(mem2chunk(p)));
    return tag_new_usable(p);
}

void*
__libc_valloc(size_t bytes)
{
    if (!__malloc_initialized)
        ptmalloc_init();

    void* address = RETURN_ADDRESS(0);
    size_t pagesize = GLRO(dl_pagesize);
    return _mid_memalign(pagesize, bytes, address);
}

void*
__libc_pvalloc(size_t bytes)
{
    if (!__malloc_initialized)
        ptmalloc_init();

    void* address = RETURN_ADDRESS(0);
    size_t pagesize = GLRO(dl_pagesize);
    size_t rounded_bytes;
    /* ALIGN_UP with overflow check.  */
    if (__glibc_unlikely(
        __builtin_add_overflow(bytes, pagesize - 1, &rounded_bytes)))
    {
        __set_errno(ENOMEM);
        return 0;
    }
    rounded_bytes = rounded_bytes & -(pagesize - 1);

    return _mid_memalign(pagesize, rounded_bytes, address);
}

void*
__libc_calloc(size_t n, size_t elem_size)
{
    mstate av;
    mchunkptr oldtop;
    INTERNAL_SIZE_T sz, oldtopsize;
    void* mem;
    unsigned long clearsize;
    unsigned long nclears;
    INTERNAL_SIZE_T* d;
    ptrdiff_t bytes;

    if (__glibc_unlikely(__builtin_mul_overflow(n, elem_size, &bytes)))
    {
        __set_errno(ENOMEM);
        return NULL;
    }

    sz = bytes;

    if (!__malloc_initialized)
        ptmalloc_init();

    MAYBE_INIT_TCACHE();

    if (SINGLE_THREAD_P)
        av = &main_arena;
    else
        arena_get(av, sz);

    if (av)
    {
        /* Check if we hand out the top chunk, in which case there may be no
           need to clear. */
#  if MORECORE_CLEARS
        oldtop = top(av);
        oldtopsize = chunksize(top(av));
#    if MORECORE_CLEARS < 2
        /* Only newly allocated memory is guaranteed to be cleared.  */
        if (av == &main_arena
            && oldtopsize
            < mp_.sbrk_base + av->max_system_mem - (char*)oldtop)
            oldtopsize = (mp_.sbrk_base + av->max_system_mem - (char*)oldtop);
#    endif
        if (av != &main_arena)
        {
            heap_info* heap = heap_for_ptr(oldtop);
            if (oldtopsize
                < (char*)heap + heap->mprotect_size - (char*)oldtop)
                oldtopsize
                = (char*)heap + heap->mprotect_size - (char*)oldtop;
        }
#  endif
    }
    else
    {
        /* No usable arenas.  */
        oldtop = 0;
        oldtopsize = 0;
    }
    mem = _int_malloc(av, sz);

    assert(!mem || chunk_is_mmapped(mem2chunk(mem))
        || av == arena_for_chunk(mem2chunk(mem)));

    if (!SINGLE_THREAD_P)
    {
        if (mem == 0 && av != NULL)
        {
            LIBC_PROBE(memory_calloc_retry, 1, sz);
            av = arena_get_retry(av, sz);
            mem = _int_malloc(av, sz);
        }

        if (av != NULL)
            __libc_lock_unlock(av->mutex);
    }

    /* Allocation failed even after a retry.  */
    if (mem == 0)
        return 0;

    mchunkptr p = mem2chunk(mem);

    /* If we are using memory tagging, then we need to set the tags
   regardless of MORECORE_CLEARS, so we zero the whole block while
   doing so.  */
    if (__glibc_unlikely(mtag_enabled))
        return tag_new_zero_region(mem, memsize(p));

    INTERNAL_SIZE_T csz = chunksize(p);

    /* Two optional cases in which clearing not necessary */
    if (chunk_is_mmapped(p))
    {
        if (__builtin_expect(perturb_byte, 0))
            return memset(mem, 0, sz);

        return mem;
    }

#  if MORECORE_CLEARS
    if (perturb_byte == 0 && (p == oldtop && csz > oldtopsize))
    {
        /* clear only the bytes from non-freshly-sbrked memory */
        csz = oldtopsize;
    }
#  endif

    /* Unroll clear of <= 36 bytes (72 if 8byte sizes).  We know that
   contents have an odd number of INTERNAL_SIZE_T-sized words;
   minimally 3.  */
    d = (INTERNAL_SIZE_T*)mem;
    clearsize = csz - SIZE_SZ;
    nclears = clearsize / sizeof(INTERNAL_SIZE_T);
    assert(nclears >= 3);

    if (nclears > 9)
        return memset(d, 0, clearsize);

    else
    {
        *(d + 0) = 0;
        *(d + 1) = 0;
        *(d + 2) = 0;
        if (nclears > 4)
        {
            *(d + 3) = 0;
            *(d + 4) = 0;
            if (nclears > 6)
            {
                *(d + 5) = 0;
                *(d + 6) = 0;
                if (nclears > 8)
                {
                    *(d + 7) = 0;
                    *(d + 8) = 0;
                }
            }
        }
    }

    return mem;
}
#endif /* IS_IN (libc) */

/*
   ------------------------------ malloc ------------------------------
 */
 // malloc的核心分配函数,av：分配区，bytes：分配字节数
static void*
_int_malloc(mstate av, size_t bytes)
{
    INTERNAL_SIZE_T nb; /* normalized request size */
    unsigned int idx;	 /* associated bin index */
    mbinptr bin;	 /* associated bin */

    mchunkptr victim;	   /* inspected/selected chunk */
    INTERNAL_SIZE_T size; /* its size */
    int victim_index;	   /* its bin index */

    mchunkptr remainder;	   /* remainder from a split */
    unsigned long remainder_size; /* its size */

    unsigned int block; /* bit map traverser */
    unsigned int bit;	 /* bit map traverser */
    unsigned int map;	 /* current word of binmap */

    mchunkptr fwd; /* misc temp for linking */
    mchunkptr bck; /* misc temp for linking */

#if USE_TCACHE
    size_t tcache_unsorted_count; /* count of unsorted chunks processed */
#endif

    /*
   Convert request size to internal form by adding SIZE_SZ bytes
   overhead plus possibly more to obtain necessary alignment and/or
   to obtain a size of at least MINSIZE, the smallest allocatable
   size. Also, checked_request2size returns false for request sizes
   that are so large that they wrap around zero when padded and
   aligned.
     */
     // 检查bytes是否在合法区间内，最大小于<2147483647
     // 函数不仅检测每次内存分配的大小是否在合法区间内（<2147483647），还调用request2size函数，将用户申请的内存大小bytes转成nb字段。
    nb = checked_request2size(bytes);
    if (nb == 0)
    {
        __set_errno(ENOMEM);
        return NULL;
    }

    /* There are no usable arenas.  Fall back to sysmalloc to get a chunk from
   mmap.
   如果分配区结构为空，则调用sysmalloc进行再次分配
   */
    if (__glibc_unlikely(av == NULL))
    {
        void* p = sysmalloc(nb, av);
        if (p != NULL)
            alloc_perturb(p, bytes);
        return p;
    }

    /*
   If the size qualifies as a fastbin, first check corresponding bin.
   This code is safe to execute even if av is not yet initialized, so we
   can try it without checking, which saves some time on this fast path.
   如果size符合fastbin的条件，首先检查相应的bin。即使av尚未初始化，
   这段代码也可以安全执行，因此我们可以不检查而尝试它，这在这条快速路径上节省了一些时间。
 */
 // 定义了一个REMOVE_FB宏函数，该函数主要目的： 原子操作，从fast
 // bins链表上，弹出一个chunk
#define REMOVE_FB(fb, victim, pp)                                             \
  do                                                                          \
    {                                                                         \
      victim = pp;                                                            \
      if (victim == NULL)                                                     \
	break;                                                                \
      pp = REVEAL_PTR (victim->fd);                                           \
      if (__glibc_unlikely (pp != NULL && misaligned_chunk (pp)))             \
	malloc_printerr ("malloc(): unaligned fastbin chunk detected");       \
    }                                                                         \
  while ((pp = catomic_compare_and_exchange_val_acq (fb, pp, victim))         \
	 != victim);

     // 在fast bin的大小区域
    if ((unsigned long)(nb) <= (unsigned long)(get_max_fast()))
    {
        // 计算fastbin的索引
        idx = fastbin_index(nb);
        // 得到fastbin指定索引的头节点地址 fastbin(av,
        // idx):((ar_ptr)->fastbinsY[idx])
        mfastbinptr* fb = &fastbin(av, idx);
        mchunkptr pp;
        // 得到fastbin指定索引的头节点
        victim = *fb;

        if (victim != NULL)
        {
            if (__glibc_unlikely(misaligned_chunk(victim)))
                malloc_printerr(
                    "malloc(): unaligned fastbin chunk detected 2");
            // 单线程环境
            if (SINGLE_THREAD_P)
                *fb = REVEAL_PTR(victim->fd);
            else
                // 设置 原子操作，将chunk从fast bins上摘下来，多线程的时候会争抢
                REMOVE_FB(fb, pp, victim);
            //
            if (__glibc_likely(victim != NULL))
            {
                // 计算内存块对应的索引，
                size_t victim_idx = fastbin_index(chunksize(victim));
                if (__builtin_expect(victim_idx != idx, 0))
                    malloc_printerr("malloc(): memory corruption (fast)");
                check_remalloced_chunk(av, victim, nb);
                // 使用缓存宏
#if USE_TCACHE
         /* While we're here, if we see other chunks of the same size,
            stash them in the tcache.
       当我们在这里时，如果我们看到其他相同大小的块，请将它们存储在 tcache
       中
     */
     // 计算缓存索引
                size_t tc_idx = csize2tidx(nb);
                if (tcache != NULL && tc_idx < mp_.tcache_bins)
                {
                    mchunkptr tc_victim;

                    /* While bin not empty and tcache not full, copy chunks.
               如果bin没空，且tcache没满，则复制chunk放入到cache中，默认的tcache_count为7
            */
                    while (tcache->counts[tc_idx] < mp_.tcache_count
                        && (tc_victim = *fb) != NULL)
                    {
                        if (__glibc_unlikely(misaligned_chunk(tc_victim)))
                            malloc_printerr(
                                "malloc(): unaligned fastbin chunk detected 3");
                        if (SINGLE_THREAD_P)
                            *fb = REVEAL_PTR(tc_victim->fd);
                        else
                        {
                            REMOVE_FB(fb, pp, tc_victim);
                            if (__glibc_unlikely(tc_victim == NULL))
                                break;
                        }
                        tcache_put(tc_victim, tc_idx);
                    }
                }
#endif
                // 通过chunk结构，获取data数据部分的指针地址
                void* p = chunk2mem(victim);
                // 初始化内存块
                alloc_perturb(p, bytes);
                return p;
            }
        }
    }

    /*
   If a small request, check regular bin.  Since these "smallbins"
   hold one size each, no searching within bins is necessary.
   (For a large request, we need to wait until unsorted chunks are
   processed to find best fit. But for small ones, fits are exact
   anyway, so we can check now, which is faster.)
   如果是小尺寸的请求，检查常规的bin.因为这些“smallbins”每个只包含一个尺寸，所以不需要在bin中进行搜索。
     */
     /*
      * 如果符合smallbin的大小，则从smallbin的数组上获取一个chunk进行内存分配，如果存在空闲chunk则分配成功；如果空闲链表为空，则分配失败需要往下走
      * 1. 如果是smallbin，则通过bins的数组下标获取到对应的chunk双向链表
      * 2. 然后从链表的尾部通过弹出一个chunk，并修改双向链表
      * 3.
      * 获取得到的victim就是要操作的内存chunk对象，通过chunk2mem和alloc_perturb函数，初始化对象
      */
    if (in_smallbin_range(nb))
    {
        // 获取small bin的索引
        idx = smallbin_index(nb);
        // 得到数据开始的地址
        bin = bin_at(av, idx);
        // 只有bin这个chunk双向链表有2个以上值，才会分配
        // bin是mchunkptr结构的链表（双向链表）；last(bin)
        // 获取bin->bk，获取最后一个chunk；从链表的尾部，取一个victim Case1：
        // A -> B -> C -> D 四个chunk，则取出D，调整双向链表：A -> B -> C
        // Case2： A -> B 两个chunk，则取出B，调整双向链表:A->bk = A / A->fd =
        // A bin里还有下一个chunk才会分配，victim是双向链表中最后一个
        if ((victim = last(bin)) != bin)
        {
            // 获取victim的前一个bin
            bck = victim->bk;
            if (__glibc_unlikely(bck->fd != victim))
                malloc_printerr(
                    "malloc(): smallbin double linked list corrupted");
            // 设置victim为使用状态
            set_inuse_bit_at_offset(victim, nb);
            // 将victim从链表中删除
            bin->bk = bck;
            bck->fd = bin;
            // 如果不是主分配区
            if (av != &main_arena)
                // 设置非主分配区的的标志
                set_non_main_arena(victim);
            check_malloced_chunk(av, victim, nb);
#if USE_TCACHE
            /* While we're here, if we see other chunks of the same size,
           stash them in the tcache.  */
           // 计算缓存索引
            size_t tc_idx = csize2tidx(nb);
            // 如果可以放入缓存
            if (tcache != NULL && tc_idx < mp_.tcache_bins)
            {
                mchunkptr tc_victim;

                /* While bin not empty and tcache not full, copy chunks over.
                 */
                 // 如果bin没空，且tcache没满，则复制chunk放入到cache中，默认的tcache_count为7
                while (tcache->counts[tc_idx] < mp_.tcache_count
                    && (tc_victim = last(bin)) != bin)
                {
                    // 将bin中chunk脱离，然后放入cache中
                    if (tc_victim != 0)
                    {
                        bck = tc_victim->bk;
                        set_inuse_bit_at_offset(tc_victim, nb);
                        if (av != &main_arena)
                            set_non_main_arena(tc_victim);
                        bin->bk = bck;
                        bck->fd = bin;

                        tcache_put(tc_victim, tc_idx);
                    }
                }
            }
#endif
            // 通过chunk结构，获取data数据部分的指针地址
            void* p = chunk2mem(victim);
            // 初始化内存块
            alloc_perturb(p, bytes);
            return p;
        }
    }

    /*
   If this is a large request, consolidate fastbins before continuing.
   While it might look excessive to kill all fastbins before
   even seeing if there is space available, this avoids
   fragmentation problems normally associated with fastbins.
   Also, in practice, programs tend to have runs of either small or
   large requests, but less often mixtures, so consolidation is not
   invoked all that often in most programs. And the programs that
   it is called frequently in otherwise tend to fragment.
     */

    else
    {
        /* 如果是largebin，则会把fast bin中的chunk进行一次整理合并
             然后将合并后的chunk放入unsorted
           bin中，这是通过malloc_consolidate这个函数完成*/
        idx = largebin_index(nb);
        if (atomic_load_relaxed(&av->have_fastchunks))
            malloc_consolidate(av);
    }

    /*
   Process recently freed or remaindered chunks, taking one only if
   it is exact fit, or, if this a small request, the chunk is remainder
   from the most recent non-exact fit.  Place other traversed chunks in
   bins.  Note that this step is the only place in any routine where
   chunks are placed in bins.

   The outer loop here is needed because we might not realize until
   near the end of malloc that we should have consolidated, so must
   do so and retry. This happens at most once, and only when we would
   otherwise need to expand memory to service a "small" request.
 */

#if USE_TCACHE
    INTERNAL_SIZE_T tcache_nb = 0;
    size_t tc_idx = csize2tidx(nb);
    if (tcache != NULL && tc_idx < mp_.tcache_bins)
        tcache_nb = nb;
    int return_cached = 0;

    tcache_unsorted_count = 0;
#endif
    //尝试从unsorted bin中分配，整个的逻辑就是找合适的chunk，如果不合适，就将其从unsorted bin中取出来，放到bin中去，然后返回
    for (;;)
    {
        int iters = 0;

        //从unsorted bin的双向链表上去循环处理，直到循环回到起始节点
        //unsorted bin放置在av->bins[1]的数组上，也是chunk类型的双向链表
        while ((victim = unsorted_chunks(av)->bk) != unsorted_chunks(av))
        {
            // 获取前一个chunk
            bck = victim->bk;
            //确认当前chunk的大小，因为unsorted bin放置的是整理后的bin，大小并不是由small bin和large bin这种数组下标来决定的
            size = chunksize(victim);
            //获取内存块上相邻的下一个chunk的地址，这个不是双向链表上的，是内存相邻的chunk
            mchunkptr next = chunk_at_offset(victim, size);

            if (__glibc_unlikely(size <= CHUNK_HDR_SZ)
                || __glibc_unlikely(size > av->system_mem))
                malloc_printerr("malloc(): invalid size (unsorted)");
            if (__glibc_unlikely(chunksize_nomask(next) < CHUNK_HDR_SZ)
                || __glibc_unlikely(chunksize_nomask(next)
            > av->system_mem))
                malloc_printerr("malloc(): invalid next size (unsorted)");
            if (__glibc_unlikely((prev_size(next) & ~(SIZE_BITS)) != size))
                malloc_printerr(
                    "malloc(): mismatching next->prev_size (unsorted)");
            if (__glibc_unlikely(bck->fd != victim)
                || __glibc_unlikely(victim->fd != unsorted_chunks(av)))
                malloc_printerr(
                    "malloc(): unsorted double linked list corrupted");
            if (__glibc_unlikely(prev_inuse(next)))
                malloc_printerr(
                    "malloc(): invalid next->prev_inuse (unsorted)");

            /*
          If a small request, try to use last remainder if it is the
          only chunk in unsorted bin.  This helps promote locality for
          runs of consecutive small requests. This is the only
          exception to best-fit, and applies only when there is
          no exact fit for a small chunk.
              */
              /*
   * 分配的是small类型的并且小于当前的chunk，则可以进行切割分配
   * 几个条件：
   * 1. 需要符合smallbin
   * 2. unsorted bins上只剩下一个bin
   * 3. 当前的chunk等于last_remainder
   * 4. size需要大于nb，并且分割后的last_remainder，仍然可以使用*/
            if (in_smallbin_range(nb) && bck == unsorted_chunks(av)
                && victim == av->last_remainder
                && (unsigned long)(size) > (unsigned long)(nb + MINSIZE))
            {
                /* split and reattach remainder */
                // 分解并且重新关联remainder
                remainder_size = size - nb;
                // 重新设置remainder的起始地址
                remainder = chunk_at_offset(victim, nb);
                // 重新挂接
                unsorted_chunks(av)->bk = unsorted_chunks(av)->fd
                    = remainder;
                // 上一个剩下的
                av->last_remainder = remainder;
                remainder->bk = remainder->fd = unsorted_chunks(av);
                // 如果不是smallbin，则设置fd_nextsize和bk_nextsize
                if (!in_smallbin_range(remainder_size))
                {
                    remainder->fd_nextsize = NULL;
                    remainder->bk_nextsize = NULL;
                }

                set_head(victim,
                    nb | PREV_INUSE
                    | (av != &main_arena ? NON_MAIN_ARENA : 0));
                set_head(remainder, remainder_size | PREV_INUSE);
                set_foot(remainder, remainder_size);

                check_malloced_chunk(av, victim, nb);
                // 从内存块偏移到指针
                void* p = chunk2mem(victim);
                alloc_perturb(p, bytes);
                return p;
            }

            /* remove from unsorted list */
            // 从unsorted列表中移除
            unsorted_chunks(av)->bk = bck;
            bck->fd = unsorted_chunks(av);

            /* Take now instead of binning if exact fit */
            // 请求的大小，正好和chunk大小匹配，则直接分配成功
            if (size == nb)
            {
                set_inuse_bit_at_offset(victim, size);
                // 不是主分配区
                if (av != &main_arena)
                    set_non_main_arena(victim);
#if USE_TCACHE
                /* Fill cache first, return to user only if cache fills.
                    We may return one of these chunks later.  */
                if (tcache_nb > 0
                    && tcache->counts[tc_idx] < mp_.tcache_count)
                {
                    // 放入cache中
                    tcache_put(victim, tc_idx);
                    return_cached = 1;
                    continue;
                }
                else
                {
#endif
                    check_malloced_chunk(av, victim, nb);
                    // 通过chunk结构，获取data数据部分的指针地址
                    void* p = chunk2mem(victim);
                    // 初始化内存
                    alloc_perturb(p, bytes);
                    return p;
#if USE_TCACHE
                }
#endif
            }

            /* place chunk in bin */
            // 尺寸符合smallbin的大小范围
            if (in_smallbin_range(size))
            {
                // 获取smallbin的索引
                victim_index = smallbin_index(size);
                // 找到插入的位置
                bck = bin_at(av, victim_index);
                fwd = bck->fd;
            }
            else
            {
                // 获取largebin的索引
                victim_index = largebin_index(size);
                // bck表示前面一个，fwd表示后面一个，新的放在中间
                bck = bin_at(av, victim_index);
                fwd = bck->fd;

                /* maintain large bins in sorted order */
                // largebin需要维护大小顺序
                if (fwd != bck)
                {
                    /* Or with inuse bit to speed comparisons */
                    // 添加正在使用的位掩码
                    size |= PREV_INUSE;
                    /* if smaller than smallest, bypass loop below */
                    // 比最小的还小
                    assert(chunk_main_arena(bck->bk));
                    if ((unsigned long)(size) < (unsigned long)chunksize_nomask(bck->bk))
                    {
                        fwd = bck;
                        bck = bck->bk;

                        victim->fd_nextsize = fwd->fd;
                        victim->bk_nextsize = fwd->fd->bk_nextsize;
                        fwd->fd->bk_nextsize
                            = victim->bk_nextsize->fd_nextsize = victim;
                    }
                    else
                    {
                        assert(chunk_main_arena(fwd));
                        // 插入排序的方式找到插入的位置
                        while ((unsigned long)size < chunksize_nomask(fwd))
                        {
                            fwd = fwd->fd_nextsize;
                            assert(chunk_main_arena(fwd));
                        }

                        if ((unsigned long)size
                            == (unsigned long)chunksize_nomask(fwd))
                            // 总是插入第二个位置
                          /* Always insert in the second position.  */
                            fwd = fwd->fd;
                        else
                        {
                            // victim的后面一个指向fwd
                            victim->fd_nextsize = fwd;
                            // victim的前面一个指向fwd的前面一个
                            victim->bk_nextsize = fwd->bk_nextsize;
                            if (__glibc_unlikely(
                                fwd->bk_nextsize->fd_nextsize != fwd))
                                malloc_printerr(
                                    "malloc(): largebin double linked list "
                                    "corrupted (nextsize)");
                            fwd->bk_nextsize = victim;
                            victim->bk_nextsize->fd_nextsize = victim;
                        }
                        bck = fwd->bk;
                        if (bck->fd != fwd)
                            malloc_printerr("malloc(): largebin double linked "
                                "list corrupted (bk)");
                    }
                }
                // 没有其他节点
                else
                    // 指向自己
                    victim->fd_nextsize = victim->bk_nextsize = victim;
            }
            // 挂接在对应的bin上
            mark_bin(av, victim_index);
            victim->bk = bck;
            victim->fd = fwd;
            fwd->bk = victim;
            bck->fd = victim;

#if USE_TCACHE
            /* If we've processed as many chunks as we're allowed while
          filling the cache, return one of the cached ones.  */
            ++tcache_unsorted_count;
            if (return_cached && mp_.tcache_unsorted_limit > 0
                && tcache_unsorted_count > mp_.tcache_unsorted_limit)
            {
                return tcache_get(tc_idx);
            }
#endif

#define MAX_ITERS 10000
            if (++iters >= MAX_ITERS)
                break;
        }

#if USE_TCACHE
        /* If all the small chunks we found ended up cached, return one now.
         */
        if (return_cached)
        {
            return tcache_get(tc_idx);
        }
#endif

        /*
           If a large request, scan through the chunks of current bin in
           sorted order to find smallest that fits.  Use the skip list for
           this.
         */
         // 如果请求很大，请按排序顺序扫描当前bin的块，以找到适合的最小块。为此，请使用跳过列表。
        if (!in_smallbin_range(nb))
        {
            // 获得idx对应bin的头节点
            bin = bin_at(av, idx);

            /* skip scan if empty or largest chunk is too small */
          // 如果idx对应的bin不为空，并且有合适尺寸的chunk
            if ((victim = first(bin)) != bin
                && (unsigned long)chunksize_nomask(victim)
                >= (unsigned long)(nb))
            {
                victim = victim->bk_nextsize;
                // 跳过那些不符合尺寸要求的
                while (((unsigned long)(size = chunksize(victim))
                    < (unsigned long)(nb)))
                    victim = victim->bk_nextsize;

                /* Avoid removing the first entry for a size so that the skip
                    list does not have to be rerouted.  */
                    // 避免删除某个大小的第一个条目，以便不必重新路由跳过列表    
                if (victim != last(bin)
                    && chunksize_nomask(victim)
                    == chunksize_nomask(victim->fd))
                    victim = victim->fd;

                // 获得剩余的尺寸
                remainder_size = size - nb;
                unlink_chunk(av, victim);

                /* Exhaust */
                // 用尽
                if (remainder_size < MINSIZE)
                {
                    set_inuse_bit_at_offset(victim, size);
                    if (av != &main_arena)
                        set_non_main_arena(victim);
                }
                /* Split */
                else
                {
                    // 还有剩余，会将剩余的作为一个单独的节点放入到unsorted bin链表中
                    remainder = chunk_at_offset(victim, nb);
                    /* We cannot assume the unsorted list is empty and
                  therefore have to perform a complete insert here.
                  我们不能假设unsorted bin为空，因此必须在此处执行完整插入
                  */
                    bck = unsorted_chunks(av);
                    fwd = bck->fd;
                    if (__glibc_unlikely(fwd->bk != bck))
                        malloc_printerr("malloc(): corrupted unsorted chunks");
                    remainder->bk = bck;
                    remainder->fd = fwd;
                    bck->fd = remainder;
                    fwd->bk = remainder;
                    // 不是smallbin，给fd_nextsize和bk_nextsize赋值
                    if (!in_smallbin_range(remainder_size))
                    {
                        remainder->fd_nextsize = NULL;
                        remainder->bk_nextsize = NULL;
                    }
                    set_head(victim,
                        nb | PREV_INUSE
                        | (av != &main_arena ? NON_MAIN_ARENA : 0));
                    set_head(remainder, remainder_size | PREV_INUSE);
                    set_foot(remainder, remainder_size);
                }
                // 返回分配的内存
                check_malloced_chunk(av, victim, nb);
                void* p = chunk2mem(victim);
                alloc_perturb(p, bytes);
                return p;
            }
        }

        /*
           Search for a chunk by scanning bins, starting with next largest
           bin. This search is strictly by best-fit; i.e., the smallest
           (with ties going to approximately the least recently used) chunk
           that fits is selected.

           The bitmap avoids needing to check that most blocks are nonempty.
           The particular case of skipping all bins during warm-up phases
           when no chunks have been returned yet is faster than it might look.
         通过扫描bin来搜索块，从下一个最大的bin开始。该搜索严格按照最佳拟合进行；即，选择适合的最小块（与大约最近最少使用的连接）。
         位图避免了需要检查大多数块是否为非空。特殊情况是当没有任何块返回在预热阶段会跳过所有的bins，这比看起来要快。
         */
         // 从bins的数组上，下一个bin开始搜索，因为下一个bin肯定是满足尺寸要求的
        ++idx;
        // 获取一个bin
        bin = bin_at(av, idx);
        // 数组下标idx转成block，block一共四个，idx为0-31则为0；idx为32-63则为1
        block = idx2block(idx);
        //获取map
        map = av->binmap[block];
        // 获取idx对应的位掩码
        bit = idx2bit(idx);

        for (;;)
        {
            /* Skip rest of block if there are no more set bits in this block.
             */
             // 当前的block(32个bin)上，对应的bin，都是空的，则跳过当前block，进入下一个block进行搜索
            if (bit > map || bit == 0)
            {
                do
                {
                    // 获取下一个block,
                    if (++block >= BINMAPSIZE) /* out of bins */
                        //如果四个block 都搜索过了，并且都是空的，则跳转到use_top进行分配
                        goto use_top;
                } while ((map = av->binmap[block]) == 0);
                // 下一个block，从下一个block对应的第一个bin开始搜索
                bin = bin_at(av, (block << BINMAPSHIFT));
                bit = 1;
            }

            /* Advance to bin with set bit. There must be one. */
            // 遍历block中的bin，直到找到一个有元素的bin
            while ((bit & map) == 0)
            {
                // 下一个bin
                bin = next_bin(bin);
                // 下一个bin对应的位掩码
                bit <<= 1;
                assert(bit != 0);
            }

            /* Inspect the bin. It is likely to be non-empty */
            // 获取bin中的最后一个chunk
            victim = last(bin);

            /*  If a false alarm (empty bin), clear the bit. */
            // 如果当前的bin只有一个chunk，则该chunk双向链表则为空，继续循环
            if (victim == bin)
            {
                // 清除位掩码
                av->binmap[block] = map &= ~bit; /* Write through */
                bin = next_bin(bin);
                bit <<= 1;
            }
            else
            {
                // 获取内存块的大小
                size = chunksize(victim);

                /*  We know the first chunk in this bin is big enough to use.
                  */
                assert((unsigned long)(size) >= (unsigned long)(nb));
                // 剩下的大小
                remainder_size = size - nb;

                /* unlink */
                // 从bin中删除该chunk
                unlink_chunk(av, victim);

                /* Exhaust */
                // 如果耗尽了
                if (remainder_size < MINSIZE)
                {
                    // 设置对应的标志位
                    set_inuse_bit_at_offset(victim, size);
                    if (av != &main_arena)
                        set_non_main_arena(victim);
                }
                /* Split */
                // 拆分
                else
                {
                    // 剩下的chunk
                    remainder = chunk_at_offset(victim, nb);

                    /* We cannot assume the unsorted list is empty and
                    therefore have to perform a complete insert here.  */
                    // 得到unsorted bin的head，将剩下的挂到unsorted bin中去
                    bck = unsorted_chunks(av);
                    fwd = bck->fd;
                    if (__glibc_unlikely(fwd->bk != bck))
                        malloc_printerr(
                            "malloc(): corrupted unsorted chunks 2");
                    remainder->bk = bck;
                    remainder->fd = fwd;
                    bck->fd = remainder;
                    fwd->bk = remainder;

                    /* advertise as last remainder */
                    // 将刚才的chunk作为last_remainder
                    if (in_smallbin_range(nb))
                        av->last_remainder = remainder;
                    if (!in_smallbin_range(remainder_size))
                    {
                        remainder->fd_nextsize = NULL;
                        remainder->bk_nextsize = NULL;
                    }
                    set_head(victim,
                        nb | PREV_INUSE
                        | (av != &main_arena ? NON_MAIN_ARENA : 0));
                    set_head(remainder, remainder_size | PREV_INUSE);
                    set_foot(remainder, remainder_size);
                }
                check_malloced_chunk(av, victim, nb);
                // 返回分配的内存
                void* p = chunk2mem(victim);
                // 初始化内存
                alloc_perturb(p, bytes);
                return p;
            }
        }

    use_top:
        /*
           If large enough, split off the chunk bordering the end of memory
           (held in av->top). Note that this is in accord with the best-fit
           search rule.  In effect, av->top is treated as larger (and thus
           less well fitting) than any other available chunk since it can
           be extended to be as large as necessary (up to system
           limitations).

           We require that av->top always exists (i.e., has size >=
           MINSIZE) after initialization, so if it would otherwise be
           exhausted by current request, it is replenished. (The main
           reason for ensuring it exists is that we may need MINSIZE space
           to put in fenceposts in sysmalloc.)
         */
         // 直接使用从未分配的
        victim = av->top;
        // 获取内存块的大小
        size = chunksize(victim);

        if (__glibc_unlikely(size > av->system_mem))
            malloc_printerr("malloc(): corrupted top size");
        // 如果尺寸比需要的大
        if ((unsigned long)(size) >= (unsigned long)(nb + MINSIZE))
        {
            // 计算剩下的尺寸
            remainder_size = size - nb;
            // 找到剩下的chunk的开始地址
            remainder = chunk_at_offset(victim, nb);
            av->top = remainder;
            set_head(victim, nb | PREV_INUSE
                | (av != &main_arena ? NON_MAIN_ARENA : 0));
            set_head(remainder, remainder_size | PREV_INUSE);

            check_malloced_chunk(av, victim, nb);
            // 得到分配的内存块
            void* p = chunk2mem(victim);
            // 初始化内存块
            alloc_perturb(p, bytes);
            return p;
        }

        /* When we are using atomic ops to free fast chunks we can get
           here for all block sizes.  */
           // 当我们使用原子操作来释放fast bin时，我们可以在这里获取所有块大小。
        else if (atomic_load_relaxed(&av->have_fastchunks))
        {
            malloc_consolidate(av);
            /* restore original bin index */
            if (in_smallbin_range(nb))
                idx = smallbin_index(nb);
            else
                idx = largebin_index(nb);
        }

        /*
            Otherwise, relay to handle system-dependent cases
            直接从系统分配
          */
        else
        {
            void* p = sysmalloc(nb, av);
            if (p != NULL)
                alloc_perturb(p, bytes);
            return p;
        }
    }
}

/*
   ------------------------------ free ------------------------------
 */

static void
_int_free(mstate av, mchunkptr p, int have_lock)
{
    INTERNAL_SIZE_T size; /* its size */
    mfastbinptr* fb;	   /* associated fastbin */

    size = chunksize(p);

    /* Little security check which won't hurt performance: the
   allocator never wraps around at the end of the address space.
   Therefore we can exclude some size values which might appear
   here by accident or by "design" from some intruder.  */
    if (__builtin_expect((uintptr_t)p > (uintptr_t)-size, 0)
        || __builtin_expect(misaligned_chunk(p), 0))
        malloc_printerr("free(): invalid pointer");
    /* We know that each chunk is at least MINSIZE bytes in size or a
   multiple of MALLOC_ALIGNMENT.  */
    if (__glibc_unlikely(size < MINSIZE || !aligned_OK(size)))
        malloc_printerr("free(): invalid size");

    check_inuse_chunk(av, p);

#if USE_TCACHE
    {
        size_t tc_idx = csize2tidx(size);
        if (tcache != NULL && tc_idx < mp_.tcache_bins)
        {
            /* Check to see if it's already in the tcache.  */
            tcache_entry* e = (tcache_entry*)chunk2mem(p);

            /* This test succeeds on double free.  However, we don't 100%
               trust it (it also matches random payload data at a 1 in
               2^<size_t> chance), so verify it's not an unlikely
               coincidence before aborting.  */
            if (__glibc_unlikely(e->key == tcache_key))
            {
                tcache_entry* tmp;
                size_t cnt = 0;
                LIBC_PROBE(memory_tcache_double_free, 2, e, tc_idx);
                for (tmp = tcache->entries[tc_idx]; tmp;
                    tmp = REVEAL_PTR(tmp->next), ++cnt)
                {
                    if (cnt >= mp_.tcache_count)
                        malloc_printerr(
                            "free(): too many chunks detected in tcache");
                    if (__glibc_unlikely(!aligned_OK(tmp)))
                        malloc_printerr(
                            "free(): unaligned chunk detected in tcache 2");
                    if (tmp == e)
                        malloc_printerr(
                            "free(): double free detected in tcache 2");
                    /* If we get here, it was a coincidence.  We've wasted a
                       few cycles, but don't abort.  */
                }
            }

            if (tcache->counts[tc_idx] < mp_.tcache_count)
            {
                tcache_put(p, tc_idx);
                return;
            }
        }
    }
#endif

    /*
      If eligible, place chunk on a fastbin so it can be found
      and used quickly in malloc.
    */
    // 如果释放的内存小于get_max_fast（），则释放的chunk放入fastbin
    if ((unsigned long)(size) <= (unsigned long)(get_max_fast())

#if TRIM_FASTBINS
        /*
          If TRIM_FASTBINS set, don't place chunks
          bordering top into fastbins
        */
        && (chunk_at_offset(p, size) != av->top)
#endif
        )
    {

        if (__builtin_expect(chunksize_nomask(chunk_at_offset(p, size))
            <= CHUNK_HDR_SZ,
            0)
            || __builtin_expect(
                chunksize(chunk_at_offset(p, size)) >= av->system_mem, 0))
        {
            bool fail = true;
            /* We might not have a lock at this point and concurrent
           modifications of system_mem might result in a false positive.
           Redo the test after getting the lock.  */
            if (!have_lock)
            {
                __libc_lock_lock(av->mutex);
                fail = (chunksize_nomask(chunk_at_offset(p, size))
                    <= CHUNK_HDR_SZ
                    || chunksize(chunk_at_offset(p, size))
                    >= av->system_mem);
                __libc_lock_unlock(av->mutex);
            }

            if (fail)
                malloc_printerr("free(): invalid next size (fast)");
        }
        // 对当前内存块进行清空操作，设置的大小等于chunk的size减去2个SIZE_SZ的空间
        free_perturb(chunk2mem(p), size - CHUNK_HDR_SZ);
        
        // 设置分配区的标志位表示fastbin有空闲chunk，原子操作
        atomic_store_relaxed(&av->have_fastchunks, true);
        // 获取fastbin 数组下标
        unsigned int idx = fastbin_index(size);
        //找到对应的fb
        fb = &fastbin(av, idx);

        /* Atomically link P to its fastbin: P->FD = *FB; *FB = P;  */
        // 自动链接p到他的fastbin中：P->FD = *FB; *FB = P;
        mchunkptr old = *fb, old2;
        // 如果是单线程处理，则直接操作
        if (SINGLE_THREAD_P)
        {
            /* Check that the top of the bin is not the record we are going to
           add (i.e., double free).  */
            if (__builtin_expect(old == p, 0))
                malloc_printerr("double free or corruption (fasttop)");
            p->fd = PROTECT_PTR(&p->fd, old);
            *fb = p;
        }
        else
            do
            {
                /* Check that the top of the bin is not the record we are going
               to add (i.e., double free).  */
                // 如果是多线程的，则需要进行原子操作
                if (__builtin_expect(old == p, 0))
                    malloc_printerr("double free or corruption (fasttop)");
                old2 = old;
                p->fd = PROTECT_PTR(&p->fd, old);
            } while ((old = catomic_compare_and_exchange_val_rel(fb, p, old2))
                != old2);

            /* Check that size of fastbin chunk at the top is the same as
               size of the chunk that we are adding.  We can dereference OLD
               only if we have the lock, otherwise it might have already been
               allocated again.  */
            if (have_lock && old != NULL
                && __builtin_expect(fastbin_index(chunksize(old)) != idx, 0))
                malloc_printerr("invalid fastbin entry (free)");
    }

    /*
      Consolidate other non-mmapped chunks as they arrive.
    */
    // 如果不是MMAP分配，则释放的时候释放到unsorted bins
    else if (!chunk_is_mmapped(p))
    {

        /* If we're single-threaded, don't lock the arena.  */
        if (SINGLE_THREAD_P)
            have_lock = true;

        if (!have_lock)
            __libc_lock_lock(av->mutex);

        _int_free_merge_chunk(av, p, size);

        if (!have_lock)
            __libc_lock_unlock(av->mutex);
    }
    /*
      If the chunk was allocated via mmap, release via munmap().
    */
    // 如果内存块是通过mmap分配，就通过munmap释放
    else
    {
        munmap_chunk(p);
    }
}

/* Try to merge chunk P of SIZE bytes with its neighbors.  Put the
   resulting chunk on the appropriate bin list.  P must not be on a
   bin list yet, and it can be in use.  
   尝试合并拥有SIZE字节的内存块P和它的邻居。然后将合并的结果块放入合适的bin列表中。
   P必须不在bin列表中，并且它可能正在使用。
   */

static void
_int_free_merge_chunk(mstate av, mchunkptr p, INTERNAL_SIZE_T size)
{
    // 获取下一个chunk
    mchunkptr nextchunk = chunk_at_offset(p, size);

    /* Lightweight tests: check whether the block is already the
   top block.  */
    if (__glibc_unlikely(p == av->top))
        malloc_printerr("double free or corruption (top)");
    /* Or whether the next chunk is beyond the boundaries of the arena.  */
    if (__builtin_expect(
        contiguous(av)
        && (char*)nextchunk
        >= ((char*)av->top + chunksize(av->top)),
        0))
        malloc_printerr("double free or corruption (out)");
    /* Or whether the block is actually not marked used.  */
    if (__glibc_unlikely(!prev_inuse(nextchunk)))
        malloc_printerr("double free or corruption (!prev)");
    
    // 下一个chunk的size
    INTERNAL_SIZE_T nextsize = chunksize(nextchunk);
    if (__builtin_expect(chunksize_nomask(nextchunk) <= CHUNK_HDR_SZ, 0)
        || __builtin_expect(nextsize >= av->system_mem, 0))
        malloc_printerr("free(): invalid next size (normal)");

    // 对当前内存块进行清空操作，设置的大小等于chunk的size 减去2个SIZE_SZ的空间
    free_perturb(chunk2mem(p), size - CHUNK_HDR_SZ);

    /* Consolidate backward.  */
    // 如果前一个内存块不在使用中，合并前一个
    if (!prev_inuse(p))
    {
        INTERNAL_SIZE_T prevsize = prev_size(p);
        size += prevsize;
        p = chunk_at_offset(p, -((long)prevsize));
        if (__glibc_unlikely(chunksize(p) != prevsize))
            malloc_printerr(
                "corrupted size vs. prev_size while consolidating");
        unlink_chunk(av, p);
    }

    /* Write the chunk header, maybe after merging with the following chunk.
     */
    // 尝试和下一个内存块合并
    size = _int_free_create_chunk(av, p, size, nextchunk, nextsize);
    _int_free_maybe_consolidate(av, size);
}

/* Create a chunk at P of SIZE bytes, with SIZE potentially increased
   to cover the immediately following chunk NEXTCHUNK of NEXTSIZE
   bytes (if NEXTCHUNK is unused).  The chunk at P is not actually
   read and does not have to be initialized.  After creation, it is
   placed on the appropriate bin list.  The function returns the size
   of the new chunk.  */
static INTERNAL_SIZE_T
_int_free_create_chunk(mstate av, mchunkptr p, INTERNAL_SIZE_T size,
    mchunkptr nextchunk, INTERNAL_SIZE_T nextsize)
{
    // 下一个内存块不是top
    if (nextchunk != av->top)
    {
        /* get and clear inuse bit */
        // 下一个内存块是否在使用
        bool nextinuse = inuse_bit_at_offset(nextchunk, nextsize);

        /* consolidate forward */
        // 没有使用标记，直接合并
        if (!nextinuse)
        {
            unlink_chunk(av, nextchunk);
            size += nextsize;
        }
        else
            // 清除使用标记
            clear_inuse_bit_at_offset(nextchunk, 0);

        /*
          Place the chunk in unsorted chunk list. Chunks are
          not placed into regular bins until after they have
          been given one chance to be used in malloc.
        */
       // 直接放入unsorted bins链表中
        mchunkptr bck = unsorted_chunks(av);
        mchunkptr fwd = bck->fd;
        if (__glibc_unlikely(fwd->bk != bck))
            malloc_printerr("free(): corrupted unsorted chunks");
        p->fd = fwd;
        p->bk = bck;
        if (!in_smallbin_range(size))
        {
            p->fd_nextsize = NULL;
            p->bk_nextsize = NULL;
        }
        bck->fd = p;
        fwd->bk = p;

        set_head(p, size | PREV_INUSE);
        set_foot(p, size);

        check_free_chunk(av, p);
    }
    // 下一个内存块是top
    else
    {
        /* If the chunk borders the current high end of memory,
           consolidate into top.  */
        // 如果该块与当前内存top接壤，则合并到top内存块中去
        size += nextsize;
        set_head(p, size | PREV_INUSE);
        av->top = p;
        check_chunk(av, p);
    }

    return size;
}

/* If freeing a large space, consolidate possibly-surrounding
   chunks.  Then, if the total unused topmost memory exceeds trim
   threshold, ask malloc_trim to reduce top.  */
// 如果释放一个大的空间，合并可能的相邻chunk。然后如果top内存块中的剩余内存大于trim阈值，则调用malloc_trim进行内存的收缩。
static void
_int_free_maybe_consolidate(mstate av, INTERNAL_SIZE_T size)
{
    /* Unless max_fast is 0, we don't know if there are fastbins
   bordering top, so we cannot tell for sure whether threshold has
   been reached unless fastbins are consolidated.  But we don't want
   to consolidate on each free.  As a compromise, consolidation is
   performed if FASTBIN_CONSOLIDATION_THRESHOLD is reached.  */
   // 如果大小超过FASTBIN_CONSOLIDATION_THRESHOLD
    if (size >= FASTBIN_CONSOLIDATION_THRESHOLD)
    {
        // 合并fastbins
        if (atomic_load_relaxed(&av->have_fastchunks))
            malloc_consolidate(av);

        // 主分配区
        if (av == &main_arena)
        {
#ifndef MORECORE_CANNOT_TRIM
            if (chunksize(av->top) >= mp_.trim_threshold)
                systrim(mp_.top_pad, av);
#endif
        }
        else
        {
            /* Always try heap_trim, even if the top chunk is not large,
           because the corresponding heap might go away.  */
            heap_info* heap = heap_for_ptr(top(av));

            assert(heap->ar_ptr == av);
            heap_trim(heap, mp_.top_pad);
        }
    }
}

/*
  ------------------------- malloc_consolidate -------------------------

  malloc_consolidate is a specialized version of free() that tears
  down chunks held in fastbins.  Free itself cannot be used for this
  purpose since, among other things, it might place chunks back onto
  fastbins.  So, instead, we need to use a minor variant of the same
  code.
  1.遍历fastbins，针对里面的chunk进行一次循环遍历操作，检查每个chunk的前一个和后一个chunk是否是free状态，如果是free状态，则合并
  2.如果合并后的chunk不和top chunk挨着，则将这个chunk放进unsorted bins中
  3.如果合并后的chunk和top chunk挨着，则重新设置top chunk的起始位置
*/

static void
    malloc_consolidate(mstate av)
{
    mfastbinptr* fb;	       /* current fastbin being consolidated */
    mfastbinptr* maxfb;       /* last fastbin (for loop control) */
    mchunkptr p;	       /* current chunk being consolidated */
    mchunkptr nextp;	       /* next chunk to consolidate */
    mchunkptr unsorted_bin;   /* bin header */
    mchunkptr first_unsorted; /* chunk to link to */

    /* These have same use as in free() */
    mchunkptr nextchunk;
    INTERNAL_SIZE_T size;
    INTERNAL_SIZE_T nextsize;
    INTERNAL_SIZE_T prevsize;
    int nextinuse;

    atomic_store_relaxed(&av->have_fastchunks, false);
    // unsorted bin起始节点
    unsorted_bin = unsorted_chunks(av);

    /*
      Remove each chunk from fast bin and consolidate it, placing it
      then in unsorted bin. Among other reasons for doing this,
      placing in unsorted bin avoids needing to calculate actual bins
      until malloc is sure that chunks aren't immediately going to be
      reused anyway.
    */
    // 遍历整个fastbin
    maxfb = &fastbin(av, NFASTBINS - 1);
    fb = &fastbin(av, 0);
    do
    {
        p = atomic_exchange_acquire(fb, NULL);
        if (p != 0)
        {
            // 遍历fastbin某一个索引的列表
            do
            {
                {
                    if (__glibc_unlikely(misaligned_chunk(p)))
                        malloc_printerr("malloc_consolidate(): "
                            "unaligned fastbin chunk detected");

                    unsigned int idx = fastbin_index(chunksize(p));
                    if ((&fastbin(av, idx)) != fb)
                        malloc_printerr(
                            "malloc_consolidate(): invalid chunk size");
                }

                check_inuse_chunk(av, p);
                nextp = REVEAL_PTR(p->fd);

                /* Slightly streamlined version of consolidation code in
                  * free() */
                size = chunksize(p);
                // 下一个块
                nextchunk = chunk_at_offset(p, size);
                nextsize = chunksize(nextchunk);
                // 前一个块不在使用
                if (!prev_inuse(p))
                {
                    prevsize = prev_size(p);
                    // 和前一个块合并
                    size += prevsize;
                    // p前移前一个块的首地址
                    p = chunk_at_offset(p, -((long)prevsize));
                    if (__glibc_unlikely(chunksize(p) != prevsize))
                        malloc_printerr(
                            "corrupted size vs. prev_size in fastbins");
                    unlink_chunk(av, p);
                }
                // 下一块不是未分配的块
                if (nextchunk != av->top)
                {
                    // 下一个块是否在使用
                    nextinuse = inuse_bit_at_offset(nextchunk, nextsize);
                    // 不在使用
                    if (!nextinuse)
                    {
                        // 合并进来
                        size += nextsize;
                        // 删除下一个块的信息
                        unlink_chunk(av, nextchunk);
                    }
                    else
                        clear_inuse_bit_at_offset(nextchunk, 0);
                    // 第一个unsorted节点，将p节点挂接在first_unsorted前面
                    first_unsorted = unsorted_bin->fd;
                    unsorted_bin->fd = p;
                    first_unsorted->bk = p;
                    // 不是smallbin，fd_nextsize和bk_nextsize不可用
                    if (!in_smallbin_range(size))
                    {
                        p->fd_nextsize = NULL;
                        p->bk_nextsize = NULL;
                    }
                    // 设置前一个在使用中
                    set_head(p, size | PREV_INUSE);
                    p->bk = unsorted_bin;
                    p->fd = first_unsorted;
                    set_foot(p, size);
                }
                // 下一个块是av->top
                else
                {
                    // 合并到av->top中去
                    size += nextsize;
                    set_head(p, size | PREV_INUSE);
                    // top从p开始
                    av->top = p;
                }
            } while ((p = nextp) != 0);
        }
    } while (fb++ != maxfb);
}

/*
  ------------------------------ realloc ------------------------------
*/
// 老的chunk内存，大于新分配的，空间足够，则可以进行分割。将老的内存切割成2个chunk，一个返回给用户端，一个放入bins上进行管理。
static void*
_int_realloc(mstate av, mchunkptr oldp, INTERNAL_SIZE_T oldsize,
    INTERNAL_SIZE_T nb)
{
    mchunkptr newp;	      /* chunk to return */
    INTERNAL_SIZE_T newsize; /* its size */
    void* newmem;	      /* corresponding user mem */

    mchunkptr next; /* next contiguous chunk after oldp */

    mchunkptr remainder;	   /* extra space at end of newp */
    unsigned long remainder_size; /* its size */

    /* oldmem size */
    if (__builtin_expect(chunksize_nomask(oldp) <= CHUNK_HDR_SZ, 0)
        || __builtin_expect(oldsize >= av->system_mem, 0)
        || __builtin_expect(oldsize != chunksize(oldp), 0))
        malloc_printerr("realloc(): invalid old size");

    check_inuse_chunk(av, oldp);

    /* All callers already filter out mmap'ed chunks.  */
    assert(!chunk_is_mmapped(oldp));
    // 下一个内存块
    next = chunk_at_offset(oldp, oldsize);
    INTERNAL_SIZE_T nextsize = chunksize(next);
    if (__builtin_expect(chunksize_nomask(next) <= CHUNK_HDR_SZ, 0)
        || __builtin_expect(nextsize >= av->system_mem, 0))
        malloc_printerr("realloc(): invalid next size");
    // 如果是收缩
    if ((unsigned long)(oldsize) >= (unsigned long)(nb))
    {
        /* already big enough; split below */
        newp = oldp;
        newsize = oldsize;
    }
    // 如果是扩大
    else
    {
        /* Try to expand forward into top */
        // 后面就是top，并且有足够的空间
        if (next == av->top
            && (unsigned long)(newsize = oldsize + nextsize)
            >= (unsigned long)(nb + MINSIZE))
        {
            set_head_size(oldp,
                nb | (av != &main_arena ? NON_MAIN_ARENA : 0));
            av->top = chunk_at_offset(oldp, nb);
            set_head(av->top, (newsize - nb) | PREV_INUSE);
            check_inuse_chunk(av, oldp);
            return tag_new_usable(chunk2mem(oldp));
        }

        /* Try to expand forward into next chunk;  split off remainder below
         */
        // 如果后面的内存块没有使用，并且大小加在一起可以满足要求
        else if (next != av->top && !inuse(next)
            && (unsigned long)(newsize = oldsize + nextsize)
            >= (unsigned long)(nb))
        {
            newp = oldp;
            unlink_chunk(av, next);
        }

        /* allocate, copy, free */
        // 分配一块新的内存
        else
        {
            newmem = _int_malloc(av, nb - MALLOC_ALIGN_MASK);
            if (newmem == 0)
                return 0; /* propagate failure */

            newp = mem2chunk(newmem);
            newsize = chunksize(newp);

            /*
           Avoid copy if newp is next chunk after oldp.
             */
            // 如果新分配的内存块是下一个内存块，则合并
            if (newp == next)
            {
                newsize += oldsize;
                newp = oldp;
            }
            else
            {
                void* oldmem = chunk2mem(oldp);
                size_t sz = memsize(oldp);
                (void)tag_region(oldmem, sz);
                newmem = tag_new_usable(newmem);
                // 拷贝数据
                memcpy(newmem, oldmem, sz);
                // 将原来的释放
                _int_free(av, oldp, 1);
                check_inuse_chunk(av, newp);
                return newmem;
            }
        }
    }

    /* If possible, free extra space in old or extended chunk */
    // 如果可能，在原来的或者扩展的内存块中释放多余的空间
    assert((unsigned long)(newsize) >= (unsigned long)(nb));
    // 剩余空间
    remainder_size = newsize - nb;
    // 剩下的很少了
    if (remainder_size < MINSIZE) /* not enough extra to split off */
    {
        set_head_size(newp,
            newsize | (av != &main_arena ? NON_MAIN_ARENA : 0));
        set_inuse_bit_at_offset(newp, newsize);
    }
    // 拆分剩下的内存块
    else /* split remainder */
    {
        remainder = chunk_at_offset(newp, nb);
        /* Clear any user-space tags before writing the header.  */
        remainder = tag_region(remainder, remainder_size);
        set_head_size(newp, nb | (av != &main_arena ? NON_MAIN_ARENA : 0));
        set_head(remainder, remainder_size | PREV_INUSE
            | (av != &main_arena ? NON_MAIN_ARENA : 0));
        /* Mark remainder as inuse so free() won't complain */
        set_inuse_bit_at_offset(remainder, remainder_size);
        // 将剩下的内存块释放
        _int_free(av, remainder, 1);
    }

    check_inuse_chunk(av, newp);
    return tag_new_usable(chunk2mem(newp));
}

/*
   ------------------------------ memalign ------------------------------
 */

 /* BYTES is user requested bytes, not requested chunksize bytes.  */
static void*
_int_memalign(mstate av, size_t alignment, size_t bytes)
{
    INTERNAL_SIZE_T nb;	   /* padded  request size */
    char* m;			   /* memory returned by malloc call */
    mchunkptr p;		   /* corresponding chunk */
    char* brk;			   /* alignment point within p */
    mchunkptr newp;		   /* chunk to return */
    INTERNAL_SIZE_T newsize;	   /* its size */
    INTERNAL_SIZE_T leadsize;	   /* leading space before alignment point */
    mchunkptr remainder;	   /* spare room at end to split off */
    unsigned long remainder_size; /* its size */
    INTERNAL_SIZE_T size;

    nb = checked_request2size(bytes);
    if (nb == 0)
    {
        __set_errno(ENOMEM);
        return NULL;
    }

    /* We can't check tcache here because we hold the arena lock, which
   tcache doesn't expect.  We expect it has been checked
   earlier.  */

   /* Strategy: search the bins looking for an existing block that
  meets our needs.  We scan a range of bins from "exact size" to
  "just under 2x", spanning the small/large barrier if needed.  If
  we don't find anything in those bins, the common malloc code will
  scan starting at 2x.  */

  /* Call malloc with worst case padding to hit alignment. */
    m = (char*)(_int_malloc(av, nb + alignment + MINSIZE));

    if (m == 0)
        return 0; /* propagate failure */

    p = mem2chunk(m);

    if ((((unsigned long)(m)) % alignment) != 0) /* misaligned */
    {
        /* Find an aligned spot inside chunk.  Since we need to give back
           leading space in a chunk of at least MINSIZE, if the first
           calculation places us at a spot with less than MINSIZE leader,
           we can move to the next aligned spot -- we've allocated enough
           total room so that this is always possible.  */
        brk = (char*)mem2chunk(((unsigned long)(m + alignment - 1))
            & -((signed long)alignment));
        if ((unsigned long)(brk - (char*)(p)) < MINSIZE)
            brk += alignment;

        newp = (mchunkptr)brk;
        leadsize = brk - (char*)(p);
        newsize = chunksize(p) - leadsize;

        /* For mmapped chunks, just adjust offset */
        if (chunk_is_mmapped(p))
        {
            set_prev_size(newp, prev_size(p) + leadsize);
            set_head(newp, newsize | IS_MMAPPED);
            return chunk2mem(newp);
        }

        /* Otherwise, give back leader, use the rest */
        set_head(newp, newsize | PREV_INUSE
            | (av != &main_arena ? NON_MAIN_ARENA : 0));
        set_inuse_bit_at_offset(newp, newsize);
        set_head_size(p,
            leadsize | (av != &main_arena ? NON_MAIN_ARENA : 0));
        _int_free_merge_chunk(av, p, leadsize);
        p = newp;

        assert(newsize >= nb
            && (((unsigned long)(chunk2mem(p))) % alignment) == 0);
    }

    /* Also give back spare room at the end */
    if (!chunk_is_mmapped(p))
    {
        size = chunksize(p);
        mchunkptr nextchunk = chunk_at_offset(p, size);
        INTERNAL_SIZE_T nextsize = chunksize(nextchunk);
        if (size > nb)
        {
            remainder_size = size - nb;
            if (remainder_size >= MINSIZE || nextchunk == av->top
                || !inuse_bit_at_offset(nextchunk, nextsize))
            {
                /* We can only give back the tail if it is larger than
                   MINSIZE, or if the following chunk is unused (top
                   chunk or unused in-heap chunk).  Otherwise we would
                   create a chunk that is smaller than MINSIZE.  */
                remainder = chunk_at_offset(p, nb);
                set_head_size(p, nb);
                remainder_size = _int_free_create_chunk(
                    av, remainder, remainder_size, nextchunk, nextsize);
                _int_free_maybe_consolidate(av, remainder_size);
            }
        }
    }

    check_inuse_chunk(av, p);
    return chunk2mem(p);
}

/*
   ------------------------------ malloc_trim ------------------------------
 */

static int
mtrim(mstate av, size_t pad)
{
    /* Ensure all blocks are consolidated.  */
    malloc_consolidate(av);

    const size_t ps = GLRO(dl_pagesize);
    int psindex = bin_index(ps);
    const size_t psm1 = ps - 1;

    int result = 0;
    for (int i = 1; i < NBINS; ++i)
        if (i == 1 || i >= psindex)
        {
            mbinptr bin = bin_at(av, i);

            for (mchunkptr p = last(bin); p != bin; p = p->bk)
            {
                INTERNAL_SIZE_T size = chunksize(p);

                if (size > psm1 + sizeof(struct malloc_chunk))
                {
                    /* See whether the chunk contains at least one unused page.
                     */
                    char* paligned_mem
                        = (char*)(((uintptr_t)p
                            + sizeof(struct malloc_chunk) + psm1)
                            & ~psm1);

                    assert((char*)chunk2mem(p) + 2 * CHUNK_HDR_SZ
                        <= paligned_mem);
                    assert((char*)p + size > paligned_mem);

                    /* This is the size we could potentially free.  */
                    size -= paligned_mem - (char*)p;

                    if (size > psm1)
                    {
#if MALLOC_DEBUG
                        /* When debugging we simulate destroying the memory
                       content.  */
                        memset(paligned_mem, 0x89, size & ~psm1);
#endif
                        __madvise(paligned_mem, size & ~psm1, MADV_DONTNEED);

                        result = 1;
                    }
                }
            }
        }

#ifndef MORECORE_CANNOT_TRIM
    return result | (av == &main_arena ? systrim(pad, av) : 0);

#else
    return result;
#endif
}

int
__malloc_trim(size_t s)
{
    int result = 0;

    if (!__malloc_initialized)
        ptmalloc_init();

    mstate ar_ptr = &main_arena;
    do
    {
        __libc_lock_lock(ar_ptr->mutex);
        result |= mtrim(ar_ptr, s);
        __libc_lock_unlock(ar_ptr->mutex);

        ar_ptr = ar_ptr->next;
    } while (ar_ptr != &main_arena);

    return result;
}

/*
   ------------------------- malloc_usable_size -------------------------
 */

static size_t
musable(void* mem)
{
    mchunkptr p = mem2chunk(mem);

    if (chunk_is_mmapped(p))
        return chunksize(p) - CHUNK_HDR_SZ;
    else if (inuse(p))
        return memsize(p);

    return 0;
}

#if IS_IN(libc)
size_t
__malloc_usable_size(void* m)
{
    if (m == NULL)
        return 0;
    return musable(m);
}
#endif

/*
   ------------------------------ mallinfo ------------------------------
   Accumulate malloc statistics for arena AV into M.
 */
static void
int_mallinfo(mstate av, struct mallinfo2* m)
{
    size_t i;
    mbinptr b;
    mchunkptr p;
    INTERNAL_SIZE_T avail;
    INTERNAL_SIZE_T fastavail;
    int nblocks;
    int nfastblocks;

    check_malloc_state(av);

    /* Account for top */
    avail = chunksize(av->top);
    nblocks = 1; /* top always exists */

    /* traverse fastbins */
    nfastblocks = 0;
    fastavail = 0;

    for (i = 0; i < NFASTBINS; ++i)
    {
        for (p = fastbin(av, i); p != 0; p = REVEAL_PTR(p->fd))
        {
            if (__glibc_unlikely(misaligned_chunk(p)))
                malloc_printerr("int_mallinfo(): "
                    "unaligned fastbin chunk detected");
            ++nfastblocks;
            fastavail += chunksize(p);
        }
    }

    avail += fastavail;

    /* traverse regular bins */
    for (i = 1; i < NBINS; ++i)
    {
        b = bin_at(av, i);
        for (p = last(b); p != b; p = p->bk)
        {
            ++nblocks;
            avail += chunksize(p);
        }
    }

    m->smblks += nfastblocks;
    m->ordblks += nblocks;
    m->fordblks += avail;
    m->uordblks += av->system_mem - avail;
    m->arena += av->system_mem;
    m->fsmblks += fastavail;
    if (av == &main_arena)
    {
        m->hblks = mp_.n_mmaps;
        m->hblkhd = mp_.mmapped_mem;
        m->usmblks = 0;
        m->keepcost = chunksize(av->top);
    }
}

struct mallinfo2
    __libc_mallinfo2(void)
{
    struct mallinfo2 m;
    mstate ar_ptr;

    if (!__malloc_initialized)
        ptmalloc_init();

    memset(&m, 0, sizeof(m));
    ar_ptr = &main_arena;
    do
    {
        __libc_lock_lock(ar_ptr->mutex);
        int_mallinfo(ar_ptr, &m);
        __libc_lock_unlock(ar_ptr->mutex);

        ar_ptr = ar_ptr->next;
    } while (ar_ptr != &main_arena);

    return m;
}
libc_hidden_def(__libc_mallinfo2)

struct mallinfo __libc_mallinfo(void)
{
    struct mallinfo m;
    struct mallinfo2 m2 = __libc_mallinfo2();

    m.arena = m2.arena;
    m.ordblks = m2.ordblks;
    m.smblks = m2.smblks;
    m.hblks = m2.hblks;
    m.hblkhd = m2.hblkhd;
    m.usmblks = m2.usmblks;
    m.fsmblks = m2.fsmblks;
    m.uordblks = m2.uordblks;
    m.fordblks = m2.fordblks;
    m.keepcost = m2.keepcost;

    return m;
}

/*
   ------------------------------ malloc_stats
   ------------------------------
 */

void
__malloc_stats(void)
{
    int i;
    mstate ar_ptr;
    unsigned int in_use_b = mp_.mmapped_mem, system_b = in_use_b;

    if (!__malloc_initialized)
        ptmalloc_init();
    _IO_flockfile(stderr);
    int old_flags2 = stderr->_flags2;
    stderr->_flags2 |= _IO_FLAGS2_NOTCANCEL;
    for (i = 0, ar_ptr = &main_arena;; i++)
    {
        struct mallinfo2 mi;

        memset(&mi, 0, sizeof(mi));
        __libc_lock_lock(ar_ptr->mutex);
        int_mallinfo(ar_ptr, &mi);
        fprintf(stderr, "Arena %d:\n", i);
        fprintf(stderr, "system bytes     = %10u\n",
            (unsigned int)mi.arena);
        fprintf(stderr, "in use bytes     = %10u\n",
            (unsigned int)mi.uordblks);
#if MALLOC_DEBUG > 1
        if (i > 0)
            dump_heap(heap_for_ptr(top(ar_ptr)));
#endif
        system_b += mi.arena;
        in_use_b += mi.uordblks;
        __libc_lock_unlock(ar_ptr->mutex);
        ar_ptr = ar_ptr->next;
        if (ar_ptr == &main_arena)
            break;
    }
    fprintf(stderr, "Total (incl. mmap):\n");
    fprintf(stderr, "system bytes     = %10u\n", system_b);
    fprintf(stderr, "in use bytes     = %10u\n", in_use_b);
    fprintf(stderr, "max mmap regions = %10u\n",
        (unsigned int)mp_.max_n_mmaps);
    fprintf(stderr, "max mmap bytes   = %10lu\n",
        (unsigned long)mp_.max_mmapped_mem);
    stderr->_flags2 = old_flags2;
    _IO_funlockfile(stderr);
}

/*
   ------------------------------ mallopt ------------------------------
 */
static __always_inline int
do_set_trim_threshold(size_t value)
{
    LIBC_PROBE(memory_mallopt_trim_threshold, 3, value, mp_.trim_threshold,
        mp_.no_dyn_threshold);
    mp_.trim_threshold = value;
    mp_.no_dyn_threshold = 1;
    return 1;
}

static __always_inline int
do_set_top_pad(size_t value)
{
    LIBC_PROBE(memory_mallopt_top_pad, 3, value, mp_.top_pad,
        mp_.no_dyn_threshold);
    mp_.top_pad = value;
    mp_.no_dyn_threshold = 1;
    return 1;
}

static __always_inline int
do_set_mmap_threshold(size_t value)
{
    LIBC_PROBE(memory_mallopt_mmap_threshold, 3, value, mp_.mmap_threshold,
        mp_.no_dyn_threshold);
    mp_.mmap_threshold = value;
    mp_.no_dyn_threshold = 1;
    return 1;
}

static __always_inline int
do_set_mmaps_max(int32_t value)
{
    LIBC_PROBE(memory_mallopt_mmap_max, 3, value, mp_.n_mmaps_max,
        mp_.no_dyn_threshold);
    mp_.n_mmaps_max = value;
    mp_.no_dyn_threshold = 1;
    return 1;
}

static __always_inline int
do_set_mallopt_check(int32_t value)
{
    return 1;
}

static __always_inline int
do_set_perturb_byte(int32_t value)
{
    LIBC_PROBE(memory_mallopt_perturb, 2, value, perturb_byte);
    perturb_byte = value;
    return 1;
}

static __always_inline int
do_set_arena_test(size_t value)
{
    LIBC_PROBE(memory_mallopt_arena_test, 2, value, mp_.arena_test);
    mp_.arena_test = value;
    return 1;
}

static __always_inline int
do_set_arena_max(size_t value)
{
    LIBC_PROBE(memory_mallopt_arena_max, 2, value, mp_.arena_max);
    mp_.arena_max = value;
    return 1;
}

#if USE_TCACHE
static __always_inline int
do_set_tcache_max(size_t value)
{
    if (value <= MAX_TCACHE_SIZE)
    {
        LIBC_PROBE(memory_tunable_tcache_max_bytes, 2, value,
            mp_.tcache_max_bytes);
        mp_.tcache_max_bytes = value;
        mp_.tcache_bins = csize2tidx(request2size(value)) + 1;
        return 1;
    }
    return 0;
}

static __always_inline int
do_set_tcache_count(size_t value)
{
    if (value <= MAX_TCACHE_COUNT)
    {
        LIBC_PROBE(memory_tunable_tcache_count, 2, value, mp_.tcache_count);
        mp_.tcache_count = value;
        return 1;
    }
    return 0;
}

static __always_inline int
do_set_tcache_unsorted_limit(size_t value)
{
    LIBC_PROBE(memory_tunable_tcache_unsorted_limit, 2, value,
        mp_.tcache_unsorted_limit);
    mp_.tcache_unsorted_limit = value;
    return 1;
}
#endif

static __always_inline int
do_set_mxfast(size_t value)
{
    if (value <= MAX_FAST_SIZE)
    {
        LIBC_PROBE(memory_mallopt_mxfast, 2, value, get_max_fast());
        set_max_fast(value);
        return 1;
    }
    return 0;
}

static __always_inline int
do_set_hugetlb(size_t value)
{
    if (value == 1)
    {
        enum malloc_thp_mode_t thp_mode = __malloc_thp_mode();
        /*
           Only enable THP madvise usage if system does support it and
           has 'madvise' mode.  Otherwise the madvise() call is wasteful.
         */
        if (thp_mode == malloc_thp_mode_madvise)
            mp_.thp_pagesize = __malloc_default_thp_pagesize();
    }
    else if (value >= 2)
        __malloc_hugepage_config(value == 2 ? 0 : value, &mp_.hp_pagesize,
            &mp_.hp_flags);
    return 0;
}

int
__libc_mallopt(int param_number, int value)
{
    mstate av = &main_arena;
    int res = 1;

    if (!__malloc_initialized)
        ptmalloc_init();
    __libc_lock_lock(av->mutex);

    LIBC_PROBE(memory_mallopt, 2, param_number, value);

    /* We must consolidate main arena before changing max_fast
   (see definition of set_max_fast).  */
    malloc_consolidate(av);

    /* Many of these helper functions take a size_t.  We do not worry
   about overflow here, because negative int values will wrap to
   very large size_t values and the helpers have sufficient range
   checking for such conversions.  Many of these helpers are also
   used by the tunables macros in arena.c.  */

    switch (param_number)
    {
    case M_MXFAST:
        res = do_set_mxfast(value);
        break;

    case M_TRIM_THRESHOLD:
        res = do_set_trim_threshold(value);
        break;

    case M_TOP_PAD:
        res = do_set_top_pad(value);
        break;

    case M_MMAP_THRESHOLD:
        res = do_set_mmap_threshold(value);
        break;

    case M_MMAP_MAX:
        res = do_set_mmaps_max(value);
        break;

    case M_CHECK_ACTION:
        res = do_set_mallopt_check(value);
        break;

    case M_PERTURB:
        res = do_set_perturb_byte(value);
        break;

    case M_ARENA_TEST:
        if (value > 0)
            res = do_set_arena_test(value);
        break;

    case M_ARENA_MAX:
        if (value > 0)
            res = do_set_arena_max(value);
        break;
    }
    __libc_lock_unlock(av->mutex);
    return res;
}
libc_hidden_def(__libc_mallopt)

/*
-------------------- Alternative MORECORE functions
--------------------
*/

/*
General Requirements for MORECORE.

The MORECORE function must have the following properties:

If MORECORE_CONTIGUOUS is false:

* MORECORE must allocate in multiples of pagesize. It will
     only be called with arguments that are multiples of pagesize.

* MORECORE(0) must return an address that is at least
     MALLOC_ALIGNMENT aligned. (Page-aligning always suffices.)

  else (i.e. If MORECORE_CONTIGUOUS is true):

* Consecutive calls to MORECORE with positive arguments
     return increasing addresses, indicating that space has been
     contiguously extended.

* MORECORE need not allocate in multiples of pagesize.
     Calls to MORECORE need not have args of multiples of pagesize.

* MORECORE need not page-align.

  In either case:

* MORECORE may allocate more memory than requested. (Or even less,
     but this will generally result in a malloc failure.)

* MORECORE must not allocate memory when given argument zero, but
     instead return one past the end address of memory from previous
     nonzero call. This malloc does NOT call MORECORE(0)
     until at least one call with positive arguments is made, so
     the initial value returned is not important.

* Even though consecutive calls to MORECORE need not return contiguous
     addresses, it must be OK for malloc'ed chunks to span multiple
     regions in those cases where they do happen to be contiguous.

* MORECORE need not handle negative arguments -- it may instead
     just return MORECORE_FAILURE when given negative arguments.
     Negative arguments are always multiples of pagesize. MORECORE
     must not misinterpret negative args as large positive unsigned
     args. You can suppress all such calls from even occurring by
defining MORECORE_CANNOT_TRIM,

  There is some variation across systems about the type of the
  argument to sbrk/MORECORE. If size_t is unsigned, then it cannot
  actually be size_t, because sbrk supports negative args, so it is
  normally the signed type of the same width as size_t (sometimes
  declared as "intptr_t", and sometimes "ptrdiff_t").  It doesn't much
  matter though. Internally, we use "long" as arguments, which should
  work across all reasonable possibilities.

  Additionally, if MORECORE ever returns failure for a positive
  request, then mmap is used as a noncontiguous system allocator. This
  is a useful backup strategy for systems with holes in address spaces
  -- in this case sbrk cannot contiguously expand the heap, but mmap
  may be able to map noncontiguous space.

  If you'd like mmap to ALWAYS be used, you can define MORECORE to be
  a function that always returns MORECORE_FAILURE.

  If you are using this malloc with something other than sbrk (or its
  emulation) to supply memory regions, you probably want to set
  MORECORE_CONTIGUOUS as false.  As an example, here is a custom
  allocator kindly contributed for pre-OSX macOS.  It uses virtually
  but not necessarily physically contiguous non-paged memory (locked
  in, present and won't get swapped out).  You can use it by
  uncommenting this section, adding some #includes, and setting up the
  appropriate defines above:

*#define MORECORE osMoreCore
*#define MORECORE_CONTIGUOUS 0

  There is also a shutdown routine that should somehow be called for
  cleanup upon program exit.

*#define MAX_POOL_ENTRIES 100
*#define MINIMUM_MORECORE_SIZE  (64 * 1024)
  static int next_os_pool;
  void *our_os_pools[MAX_POOL_ENTRIES];

  void *osMoreCore(int size)
  {
   void *ptr = 0;
   static void *sbrk_top = 0;

   if (size > 0)
   {
     if (size < MINIMUM_MORECORE_SIZE)
    size = MINIMUM_MORECORE_SIZE;
     if (CurrentExecutionLevel() == kTaskLevel)
    ptr = PoolAllocateResident(size + RM_PAGE_SIZE, 0);
     if (ptr == 0)
     {
       return (void *) MORECORE_FAILURE;
     }
     // save ptrs so they can be freed during cleanup
     our_os_pools[next_os_pool] = ptr;
     next_os_pool++;
     ptr = (void *) ((((unsigned long) ptr) + RM_PAGE_MASK) &
~RM_PAGE_MASK); sbrk_top = (char *) ptr + size; return ptr;
   }
   else if (size < 0)
   {
     // we don't currently support shrink behavior
     return (void *) MORECORE_FAILURE;
   }
   else
   {
     return sbrk_top;
   }
  }

  // cleanup any allocated memory pools
  // called as last thing before shutting down driver

  void osCleanupMem(void)
  {
   void **ptr;

   for (ptr = our_os_pools; ptr < &our_os_pools[MAX_POOL_ENTRIES];
ptr++) if (*ptr)
     {
    PoolDeallocate(*ptr);
* ptr = 0;
     }
  }

*/

/* Helper code.  */

extern char** __libc_argv attribute_hidden;

static void
malloc_printerr(const char* str)
{
#if IS_IN(libc)
    __libc_message("%s\n", str);
#else
    __libc_fatal(str);
#endif
    __builtin_unreachable();
}

#if IS_IN(libc)
/* We need a wrapper function for one of the additions of POSIX.  */
int
__posix_memalign(void** memptr, size_t alignment, size_t size)
{
    void* mem;

    if (!__malloc_initialized)
        ptmalloc_init();

    /* Test whether the SIZE argument is valid.  It must be a power of
   two multiple of sizeof (void *).  */
    if (alignment % sizeof(void*) != 0
        || !powerof2(alignment / sizeof(void*)) || alignment == 0)
        return EINVAL;

    void* address = RETURN_ADDRESS(0);
    mem = _mid_memalign(alignment, size, address);

    if (mem != NULL)
    {
        *memptr = mem;
        return 0;
    }

    return ENOMEM;
}
weak_alias(__posix_memalign, posix_memalign)
#endif

int __malloc_info(int options, FILE* fp)
{
    /* For now, at least.  */
    if (options != 0)
        return EINVAL;

    int n = 0;
    size_t total_nblocks = 0;
    size_t total_nfastblocks = 0;
    size_t total_avail = 0;
    size_t total_fastavail = 0;
    size_t total_system = 0;
    size_t total_max_system = 0;
    size_t total_aspace = 0;
    size_t total_aspace_mprotect = 0;

    if (!__malloc_initialized)
        ptmalloc_init();

    fputs("<malloc version=\"1\">\n", fp);

    /* Iterate over all arenas currently in use.  */
    mstate ar_ptr = &main_arena;
    do
    {
        fprintf(fp, "<heap nr=\"%d\">\n<sizes>\n", n++);

        size_t nblocks = 0;
        size_t nfastblocks = 0;
        size_t avail = 0;
        size_t fastavail = 0;
        struct
        {
            size_t from;
            size_t to;
            size_t total;
            size_t count;
        } sizes[NFASTBINS + NBINS - 1];
#define nsizes (sizeof (sizes) / sizeof (sizes[0]))

        __libc_lock_lock(ar_ptr->mutex);

        /* Account for top chunk.  The top-most available chunk is
           treated specially and is never in any bin. See "initial_top"
           comments.  */
        avail = chunksize(ar_ptr->top);
        nblocks = 1; /* Top always exists.  */

        for (size_t i = 0; i < NFASTBINS; ++i)
        {
            mchunkptr p = fastbin(ar_ptr, i);
            if (p != NULL)
            {
                size_t nthissize = 0;
                size_t thissize = chunksize(p);

                while (p != NULL)
                {
                    if (__glibc_unlikely(misaligned_chunk(p)))
                        malloc_printerr("__malloc_info(): "
                            "unaligned fastbin chunk detected");
                    ++nthissize;
                    p = REVEAL_PTR(p->fd);
                }

                fastavail += nthissize * thissize;
                nfastblocks += nthissize;
                sizes[i].from = thissize - (MALLOC_ALIGNMENT - 1);
                sizes[i].to = thissize;
                sizes[i].count = nthissize;
            }
            else
                sizes[i].from = sizes[i].to = sizes[i].count = 0;

            sizes[i].total = sizes[i].count * sizes[i].to;
        }

        mbinptr bin;
        struct malloc_chunk* r;

        for (size_t i = 1; i < NBINS; ++i)
        {
            bin = bin_at(ar_ptr, i);
            r = bin->fd;
            sizes[NFASTBINS - 1 + i].from = ~((size_t)0);
            sizes[NFASTBINS - 1 + i].to = sizes[NFASTBINS - 1 + i].total
                = sizes[NFASTBINS - 1 + i].count = 0;

            if (r != NULL)
                while (r != bin)
                {
                    size_t r_size = chunksize_nomask(r);
                    ++sizes[NFASTBINS - 1 + i].count;
                    sizes[NFASTBINS - 1 + i].total += r_size;
                    sizes[NFASTBINS - 1 + i].from
                        = MIN(sizes[NFASTBINS - 1 + i].from, r_size);
                    sizes[NFASTBINS - 1 + i].to
                        = MAX(sizes[NFASTBINS - 1 + i].to, r_size);

                    r = r->fd;
                }

            if (sizes[NFASTBINS - 1 + i].count == 0)
                sizes[NFASTBINS - 1 + i].from = 0;
            nblocks += sizes[NFASTBINS - 1 + i].count;
            avail += sizes[NFASTBINS - 1 + i].total;
        }

        size_t heap_size = 0;
        size_t heap_mprotect_size = 0;
        size_t heap_count = 0;
        if (ar_ptr != &main_arena)
        {
            /* Iterate over the arena heaps from back to front.  */
            heap_info* heap = heap_for_ptr(top(ar_ptr));
            do
            {
                heap_size += heap->size;
                heap_mprotect_size += heap->mprotect_size;
                heap = heap->prev;
                ++heap_count;
            } while (heap != NULL);
        }

        __libc_lock_unlock(ar_ptr->mutex);

        total_nfastblocks += nfastblocks;
        total_fastavail += fastavail;

        total_nblocks += nblocks;
        total_avail += avail;

        for (size_t i = 0; i < nsizes; ++i)
            if (sizes[i].count != 0 && i != NFASTBINS)
                fprintf(fp, "\
  <size from=\"%zu\" to=\"%zu\" total=\"%zu\" count=\"%zu\"/>\n",
                    sizes[i].from, sizes[i].to, sizes[i].total,
                    sizes[i].count);

        if (sizes[NFASTBINS].count != 0)
            fprintf(fp, "\
  <unsorted from=\"%zu\" to=\"%zu\" total=\"%zu\" count=\"%zu\"/>\n",
                sizes[NFASTBINS].from, sizes[NFASTBINS].to,
                sizes[NFASTBINS].total, sizes[NFASTBINS].count);

        total_system += ar_ptr->system_mem;
        total_max_system += ar_ptr->max_system_mem;

        fprintf(
            fp,
            "</sizes>\n<total type=\"fast\" count=\"%zu\" size=\"%zu\"/>\n"
            "<total type=\"rest\" count=\"%zu\" size=\"%zu\"/>\n"
            "<system type=\"current\" size=\"%zu\"/>\n"
            "<system type=\"max\" size=\"%zu\"/>\n",
            nfastblocks, fastavail, nblocks, avail, ar_ptr->system_mem,
            ar_ptr->max_system_mem);

        if (ar_ptr != &main_arena)
        {
            fprintf(fp,
                "<aspace type=\"total\" size=\"%zu\"/>\n"
                "<aspace type=\"mprotect\" size=\"%zu\"/>\n"
                "<aspace type=\"subheaps\" size=\"%zu\"/>\n",
                heap_size, heap_mprotect_size, heap_count);
            total_aspace += heap_size;
            total_aspace_mprotect += heap_mprotect_size;
        }
        else
        {
            fprintf(fp,
                "<aspace type=\"total\" size=\"%zu\"/>\n"
                "<aspace type=\"mprotect\" size=\"%zu\"/>\n",
                ar_ptr->system_mem, ar_ptr->system_mem);
            total_aspace += ar_ptr->system_mem;
            total_aspace_mprotect += ar_ptr->system_mem;
        }

        fputs("</heap>\n", fp);
        ar_ptr = ar_ptr->next;
    } while (ar_ptr != &main_arena);

    fprintf(fp,
        "<total type=\"fast\" count=\"%zu\" size=\"%zu\"/>\n"
        "<total type=\"rest\" count=\"%zu\" size=\"%zu\"/>\n"
        "<total type=\"mmap\" count=\"%d\" size=\"%zu\"/>\n"
        "<system type=\"current\" size=\"%zu\"/>\n"
        "<system type=\"max\" size=\"%zu\"/>\n"
        "<aspace type=\"total\" size=\"%zu\"/>\n"
        "<aspace type=\"mprotect\" size=\"%zu\"/>\n"
        "</malloc>\n",
        total_nfastblocks, total_fastavail, total_nblocks, total_avail,
        mp_.n_mmaps, mp_.mmapped_mem, total_system, total_max_system,
        total_aspace, total_aspace_mprotect);

    return 0;
}
#if IS_IN(libc)
weak_alias(__malloc_info, malloc_info)

strong_alias(__libc_calloc, __calloc) weak_alias(
    __libc_calloc,
    calloc) strong_alias(__libc_free,
        __free) strong_alias(__libc_free, free)
    strong_alias(__libc_malloc, __malloc) strong_alias(
        __libc_malloc,
        malloc) strong_alias(__libc_memalign,
            __memalign) weak_alias(__libc_memalign,
                memalign)
    strong_alias(__libc_realloc, __realloc) strong_alias(
        __libc_realloc,
        realloc) strong_alias(__libc_valloc,
            __valloc) weak_alias(__libc_valloc,
                valloc)
    strong_alias(__libc_pvalloc, __pvalloc) weak_alias(
        __libc_pvalloc, pvalloc) strong_alias(__libc_mallinfo,
            __mallinfo)
    weak_alias(__libc_mallinfo,
        mallinfo) strong_alias(__libc_mallinfo2,
            __mallinfo2)
    weak_alias(__libc_mallinfo2, mallinfo2)
    strong_alias(__libc_mallopt, __mallopt)
    weak_alias(__libc_mallopt, mallopt)

    weak_alias(__malloc_stats,
        malloc_stats)
    weak_alias(__malloc_usable_size,
        malloc_usable_size)
    weak_alias(__malloc_trim,
        malloc_trim)
#endif

#if SHLIB_COMPAT(libc, GLIBC_2_0, GLIBC_2_26)
    compat_symbol(libc,
        __libc_free,
        cfree,
        GLIBC_2_0);
#endif

/* ------------------------------------------------------------
   History:

   [see ftp://g.oswego.edu/pub/misc/malloc.c for the history of dlmalloc]

 */
 /*
  * Local variables:
  * c-basic-offset: 2
  * End:
  */
