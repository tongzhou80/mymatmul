An analysis why case 1 uses 48 registers in S2 while case 2 uses only 40 registers. 

Basically the A tile B tile loading loops are fully unrolled by the compiler.


  ⎿  === Case 1: bm8 / 32x4 (128 threads) — 2 A-loads + 8 B-loads ===
       /*0630*/               @P0 LDG.E.U16.CONSTANT R45, [R20.64] ;                   /* 0x00000006142d0981 */
       /*06b0*/               @P0 LDG.E.U16.CONSTANT R42, [R22.64] ;                   /* 0x00000006162a0981 */
       /*0730*/               @P0 LDG.E.U16.CONSTANT R25, [R20.64] ;                   /* 0x0000000614190981 */
       /*07a0*/               @P0 LDG.E.U16.CONSTANT R44, [R22.64] ;                   /* 0x00000006162c0981 */
       /*0830*/               @P0 LDG.E.U16.CONSTANT R24, [R20.64] ;                   /* 0x0000000614180981 */
       /*08b0*/               @P0 LDG.E.U16.CONSTANT R43, [R22.64] ;                   /* 0x00000006162b0981 */
       /*0920*/                   STS.U16 [R33], R45 ;                                 /* 0x0000002d21007388 */
       /*0940*/               @P0 LDG.E.U16.CONSTANT R45, [R20.64] ;                   /* 0x00000006142d0981 */
       /*0970*/                   STS.U16 [R29], R42 ;                                 /* 0x0000002a1d007388 */
       /*09d0*/               @P0 LDG.E.U16.CONSTANT R42, [R22.64] ;                   /* 0x00000006162a0981 */
       /*0a00*/                   STS.U16 [R33+0x200], R25 ;                           /* 0x0002001921007388 */
       /*0a60*/               @P0 LDG.E.U16.CONSTANT R25, [R20.64] ;                   /* 0x0000000614190981 */
       /*0a90*/                   STS.U16 [R29+0x200], R44 ;                           /* 0x0002002c1d007388 */
       /*0ae0*/               @P0 LDG.E.U16.CONSTANT R44, [R22.64] ;                   /* 0x00000006162c0981 */
       /*0af0*/                   STS.U16 [R7], R24 ;                                  /* 0x0000001807007388 */
       /*0b00*/                   STS.U16 [R8], R43 ;                                  /* 0x0000002b08007388 */
       /*0b10*/                   STS.U16 [R10], R45 ;                                 /* 0x0000002d0a007388 */
       /*0b20*/                   STS.U16 [R12], R42 ;                                 /* 0x0000002a0c007388 */
       /*0b30*/                   STS.U16 [R14], R25 ;                                 /* 0x000000190e007388 */
       /*0b40*/                   STS.U16 [R16], R44 ;                                 /* 0x0000002c10007388 */
       /*0b50*/                   BAR.SYNC 0x0 ;                                       /* 0x0000000000007b1d */

     === Case 2: bm16 / 32x8 (256 threads) — 2 A-loads + 4 B-loads ===
       /*04d0*/               @P2 LDG.E.U16.CONSTANT R32, [R16.64] ;                   /* 0x0000000610202981 */
       /*04f0*/               @P3 LDG.E.U16.CONSTANT R33, [R14.64] ;                   /* 0x000000060e213981 */
       /*05b0*/               @P0 LDG.E.U16.CONSTANT R35, [R20.64] ;                   /* 0x0000000614230981 */
       /*0630*/               @P1 LDG.E.U16.CONSTANT R37, [R18.64] ;                   /* 0x0000000612251981 */
       /*0650*/               @P2 LDG.E.U16.CONSTANT R36, [R16.64] ;                   /* 0x0000000610242981 */
       /*0660*/               @P3 LDG.E.U16.CONSTANT R34, [R14.64] ;                   /* 0x000000060e223981 */
       /*0690*/                   STS.U16 [R23], R35 ;                                 /* 0x0000002317007388 */
       /*06a0*/                   STS.U16 [R0], R37 ;                                  /* 0x0000002500007388 */
       /*06b0*/                   STS.U16 [R23+0x400], R32 ;                           /* 0x0004002017007388 */
       /*06c0*/                   STS.U16 [R0+0x400], R33 ;                            /* 0x0004002100007388 */
       /*06d0*/                   STS.U16 [R8], R36 ;                                  /* 0x0000002408007388 */
       /*06e0*/                   STS.U16 [R10], R34 ;                                 /* 0x000000220a007388 */
       /*06f0*/                   BAR.SYNC 0x0 ;                                       /* 0x0000000000007b1d */

                                                                                                                                                            01:50 PM claude-sonnet-4-6
● Your intuition is exactly right, and the SASS confirms it:

  Case 2 (4 B-loads): Clean schedule — all 6 LDGs grouped, then all 6 STS grouped, then BAR.SYNC. The warp issuer fires all loads back-to-back with no intervening stores, giving maximum
  opportunity to hide global memory latency.

  Case 1 (8 B-loads): Mixed schedule — 6 LDGs grouped, then alternating STS+LDG pairs for the remaining 4 B-loads, then a final batch of 6 STS:

  LDG LDG LDG LDG LDG LDG          ← 6 loads grouped (good)
  STS LDG STS LDG STS LDG STS LDG  ← interleaved pairs (problematic)
  STS STS STS STS STS STS           ← remaining stores
  BAR.SYNC

  The compiler ran out of register budget to hold all 10 values in flight simultaneously. With only 48 registers it could buffer 6 loads at once, but the 7th and 8th B-load destinations
  (R45, R42, R25, R44) had to reuse registers already consumed, forcing the compiler to drain them via STS before issuing the next LDG into that register. So the interleaving in Case 1 is a
  direct consequence of register pressure — it's not a scheduling choice but a constraint. This is another angle on why more registers isn't always better: 48 registers in Case 1 still
  wasn't enough to group all 10 loads, whereas Case 2's 40 registers were sufficient for its 6 loads.

✻ Baked for 38s

❯ LDG.E.U16.CONSTANT R45, [R20.64] ;                   /* 0x00000006142d0981 */
         /*06b0*/               @P0 LDG.E.U16.CONSTANT R42, [R22.64] ;                   /* 0x00000006162a0981 */
         /*0730*/               @P0 LDG.E.U16.CONSTANT R25, [R20.64] ;                   /* 0x0000000614190981 */
         /*07a0*/               @P0 LDG.E.U16.CONSTANT R44, [R22.64] ;                   /* 0x00000006162c0981 */

         /*0830*/               @P0 LDG.E.U16.CONSTANT R24, [R20.64] ;                   /* 0x0000000614180981 */
         /*08b0*/               @P0 LDG.E.U16.CONSTANT R43, [R22.64] ;  Here why do consecutive loads seem use the same registers? like [R20.64] and [R22.64] then [R20.64] again

                                                                                                                                                            01:56 PM claude-sonnet-4-6
● That's because our filter hid the address-arithmetic instructions between the loads. R20 and R22 are running pointer registers that get incremented after each use — we just can't see the
  IMAD/LEA instructions that bump them. Let me show the full unfiltered sequence:

● Bash(python3 - <<'EOF'
      with open("/tmp/s2.sass") as f:
          lines = f.readlines()

      k1_lines = lines[711:]
      syncs = [i for i, l in enumerate(k1_lines) if 'BAR.SYNC' in l]
      region = k1_lines[:syncs[0]+1]

      # Find start of first LDG
      start = next(i for i, l in enumerate(region) if 'LDG' in l)

      print("=== Case 1: full SASS from first LDG to BAR.SYNC ===")
      for l in region[start:]:
          print(l.rstrip())
      EOF)
  ⎿  === Case 1: full SASS from first LDG to BAR.SYNC ===
             /*0630*/               @P0 LDG.E.U16.CONSTANT R45, [R20.64] ;                   /* 0x00000006142d0981 */
                                                                                             /* 0x0000a2000c1e9500 */
             /*0640*/                   ISETP.GE.AND P0, PT, R23, c[0x0][0x17c], PT ;        /* 0x00005f0017007a0c */
                                                                                             /* 0x000fc80003f06270 */
             /*0650*/                   ISETP.LT.AND P0, PT, R30, c[0x0][0x178], !P0 ;       /* 0x00005e001e007a0c */
                                                                                             /* 0x000fda0004701270 */
             /*0660*/               @P0 IMAD.MOV.U32 R25, RZ, RZ, 0x2 ;                      /* 0x00000002ff190424 */
                                                                                             /* 0x000fe200078e00ff */
             /*0670*/              @!P0 PRMT R42, RZ, 0x7610, R42 ;                          /* 0x00007610ff2a8816 */
                                                                                             /* 0x000fe2000000002a */
             /*0680*/               @P0 IMAD R22, R30, c[0x0][0x17c], R23 ;                  /* 0x00005f001e160a24 */
                                                                                             /* 0x000fc800078e0217 */
             /*0690*/               @P0 IMAD.WIDE R22, R22, R25, c[0x0][0x160] ;             /* 0x0000580016160625 */
                                                                                             /* 0x000fc800078e0219 */
             /*06a0*/                   IMAD.IADD R25, R36, 0x1, R19 ;                       /* 0x0000000124197824 */
                                                                                             /* 0x000fe200078e0213 */
             /*06b0*/               @P0 LDG.E.U16.CONSTANT R42, [R22.64] ;                   /* 0x00000006162a0981 */
                                                                                             /* 0x0002e2000c1e9500 */
             /*06c0*/                   ISETP.GE.AND P0, PT, R28, c[0x0][0x180], PT ;        /* 0x000060001c007a0c */
                                                                                             /* 0x000fc80003f06270 */
             /*06d0*/                   ISETP.LT.AND P0, PT, R25, c[0x0][0x17c], !P0 ;       /* 0x00005f0019007a0c */
                                                                                             /* 0x000fe20004701270 */
             /*06e0*/                   IMAD.IADD R24, R31, 0x1, R19 ;                       /* 0x000000011f187824 */
                                                                                             /* 0x000fd800078e0213 */
             /*06f0*/               @P0 IMAD.MOV.U32 R21, RZ, RZ, 0x2 ;                      /* 0x00000002ff150424 */
                                                                                             /* 0x001fe400078e00ff */
             /*0700*/               @P0 IMAD R20, R25, c[0x0][0x180], R28 ;                  /* 0x0000600019140a24 */
                                                                                             /* 0x000fe200078e021c */
             /*0710*/              @!P0 PRMT R25, RZ, 0x7610, R25 ;                          /* 0x00007610ff198816 */
                                                                                             /* 0x000fc60000000019 */
             /*0720*/               @P0 IMAD.WIDE R20, R20, R21, c[0x0][0x168] ;             /* 0x00005a0014140625 */
                                                                                             /* 0x000fca00078e0215 */
             /*0730*/               @P0 LDG.E.U16.CONSTANT R25, [R20.64] ;                   /* 0x0000000614190981 */
                                                                                             /* 0x000122000c1e9500 */
             /*0740*/                   ISETP.GE.AND P0, PT, R27, c[0x0][0x180], PT ;        /* 0x000060001b007a0c */
                                                                                             /* 0x000fc80003f06270 */
             /*0750*/                   ISETP.LT.AND P0, PT, R24, c[0x0][0x17c], !P0 ;       /* 0x00005f0018007a0c */
                                                                                             /* 0x000fda0004701270 */
             /*0760*/               @P0 IMAD.MOV.U32 R23, RZ, RZ, 0x2 ;                      /* 0x00000002ff170424 */
                                                                                             /* 0x002fe200078e00ff */
             /*0770*/              @!P0 PRMT R44, RZ, 0x7610, R44 ;                          /* 0x00007610ff2c8816 */
                                                                                             /* 0x000fe2000000002c */
             /*0780*/               @P0 IMAD R22, R24, c[0x0][0x180], R27 ;                  /* 0x0000600018160a24 */
                                                                                             /* 0x000fc800078e021b */
             /*0790*/               @P0 IMAD.WIDE R22, R22, R23, c[0x0][0x168] ;             /* 0x00005a0016160625 */
                                                                                             /* 0x000fca00078e0217 */
             /*07a0*/               @P0 LDG.E.U16.CONSTANT R44, [R22.64] ;                   /* 0x00000006162c0981 */
                                                                                             /* 0x000362000c1e9500 */
             /*07b0*/                   IMAD.IADD R43, R26, 0x1, R19 ;                       /* 0x000000011a2b7824 */
                                                                                             /* 0x000fe200078e0213 */
             /*07c0*/                   ISETP.GE.AND P0, PT, R18, c[0x0][0x180], PT ;        /* 0x0000600012007a0c */
                                                                                             /* 0x000fc80003f06270 */
             /*07d0*/                   ISETP.LT.AND P0, PT, R43, c[0x0][0x17c], !P0 ;       /* 0x00005f002b007a0c */
                                                                                             /* 0x000fda0004701270 */
             /*07e0*/               @P0 IMAD R20, R43, c[0x0][0x180], R18 ;                  /* 0x000060002b140a24 */
                                                                                             /* 0x001fe200078e0212 */
             /*07f0*/              @!P0 PRMT R24, RZ, 0x7610, R24 ;                          /* 0x00007610ff188816 */
                                                                                             /* 0x000fe20000000018 */
             /*0800*/               @P0 IMAD.MOV.U32 R21, RZ, RZ, 0x2 ;                      /* 0x00000002ff150424 */
                                                                                             /* 0x000fc800078e00ff */
             /*0810*/               @P0 IMAD.WIDE R20, R20, R21, c[0x0][0x168] ;             /* 0x00005a0014140625 */
                                                                                             /* 0x000fc800078e0215 */
             /*0820*/                   IMAD.IADD R43, R2, 0x1, R19 ;                        /* 0x00000001022b7824 */
                                                                                             /* 0x000fe200078e0213 */
             /*0830*/               @P0 LDG.E.U16.CONSTANT R24, [R20.64] ;                   /* 0x0000000614180981 */
                                                                                             /* 0x000162000c1e9500 */
             /*0840*/                   ISETP.GE.AND P0, PT, R9, c[0x0][0x180], PT ;         /* 0x0000600009007a0c */
                                                                                             /* 0x000fc80003f06270 */
             /*0850*/                   ISETP.LT.AND P0, PT, R43, c[0x0][0x17c], !P0 ;       /* 0x00005f002b007a0c */
                                                                                             /* 0x000fda0004701270 */
             /*0860*/               @P0 IMAD R22, R43, c[0x0][0x180], R9 ;                   /* 0x000060002b160a24 */
                                                                                             /* 0x002fe200078e0209 */
             /*0870*/              @!P0 PRMT R43, RZ, 0x7610, R43 ;                          /* 0x00007610ff2b8816 */
                                                                                             /* 0x000fe2000000002b */
             /*0880*/               @P0 IMAD.MOV.U32 R23, RZ, RZ, 0x2 ;                      /* 0x00000002ff170424 */
                                                                                             /* 0x000fc800078e00ff */
             /*0890*/               @P0 IMAD.WIDE R22, R22, R23, c[0x0][0x168] ;             /* 0x00005a0016160625 */
                                                                                             /* 0x000fc800078e0217 */
             /*08a0*/                   IMAD.IADD R20, R3, 0x1, R19 ;                        /* 0x0000000103147824 */
                                                                                             /* 0x001fe200078e0213 */
             /*08b0*/               @P0 LDG.E.U16.CONSTANT R43, [R22.64] ;                   /* 0x00000006162b0981 */
                                                                                             /* 0x000162000c1e9500 */
             /*08c0*/                   ISETP.GE.AND P0, PT, R11, c[0x0][0x180], PT ;        /* 0x000060000b007a0c */
                                                                                             /* 0x000fc80003f06270 */
             /*08d0*/                   ISETP.LT.AND P0, PT, R20, c[0x0][0x17c], !P0 ;       /* 0x00005f0014007a0c */
                                                                                             /* 0x000fda0004701270 */
             /*08e0*/               @P0 IMAD R20, R20, c[0x0][0x180], R11 ;                  /* 0x0000600014140a24 */
                                                                                             /* 0x000fe400078e020b */
             /*08f0*/               @P0 IMAD.MOV.U32 R21, RZ, RZ, 0x2 ;                      /* 0x00000002ff150424 */
                                                                                             /* 0x000fc800078e00ff */
             /*0900*/               @P0 IMAD.WIDE R20, R20, R21, c[0x0][0x168] ;             /* 0x00005a0014140625 */
                                                                                             /* 0x000fc800078e0215 */
             /*0910*/                   IMAD.IADD R22, R4, 0x1, R19 ;                        /* 0x0000000104167824 */
                                                                                             /* 0x001fe200078e0213 */
             /*0920*/                   STS.U16 [R33], R45 ;                                 /* 0x0000002d21007388 */
                                                                                             /* 0x0041e40000000400 */
             /*0930*/              @!P0 PRMT R45, RZ, 0x7610, R45 ;                          /* 0x00007610ff2d8816 */
                                                                                             /* 0x001fcc000000002d */
             /*0940*/               @P0 LDG.E.U16.CONSTANT R45, [R20.64] ;                   /* 0x00000006142d0981 */
                                                                                             /* 0x0000a2000c1e9500 */
             /*0950*/                   ISETP.GE.AND P0, PT, R13, c[0x0][0x180], PT ;        /* 0x000060000d007a0c */
                                                                                             /* 0x000fc80003f06270 */
             /*0960*/                   ISETP.LT.AND P0, PT, R22, c[0x0][0x17c], !P0 ;       /* 0x00005f0016007a0c */
                                                                                             /* 0x000fe20004701270 */
             /*0970*/                   STS.U16 [R29], R42 ;                                 /* 0x0000002a1d007388 */
                                                                                             /* 0x0083d80000000400 */
             /*0980*/               @P0 IMAD R22, R22, c[0x0][0x180], R13 ;                  /* 0x0000600016160a24 */
                                                                                             /* 0x000fe400078e020d */
             /*0990*/               @P0 IMAD.MOV.U32 R23, RZ, RZ, 0x2 ;                      /* 0x00000002ff170424 */
                                                                                             /* 0x000fe200078e00ff */
             /*09a0*/              @!P0 PRMT R42, RZ, 0x7610, R42 ;                          /* 0x00007610ff2a8816 */
                                                                                             /* 0x002fc6000000002a */
             /*09b0*/               @P0 IMAD.WIDE R22, R22, R23, c[0x0][0x168] ;             /* 0x00005a0016160625 */
                                                                                             /* 0x000fc800078e0217 */
             /*09c0*/                   IMAD.IADD R20, R5, 0x1, R19 ;                        /* 0x0000000105147824 */
                                                                                             /* 0x001fe200078e0213 */
             /*09d0*/               @P0 LDG.E.U16.CONSTANT R42, [R22.64] ;                   /* 0x00000006162a0981 */
                                                                                             /* 0x0000e2000c1e9500 */
             /*09e0*/                   ISETP.GE.AND P0, PT, R15, c[0x0][0x180], PT ;        /* 0x000060000f007a0c */
                                                                                             /* 0x000fc80003f06270 */
             /*09f0*/                   ISETP.LT.AND P0, PT, R20, c[0x0][0x17c], !P0 ;       /* 0x00005f0014007a0c */
                                                                                             /* 0x000fe20004701270 */
             /*0a00*/                   STS.U16 [R33+0x200], R25 ;                           /* 0x0002001921007388 */
                                                                                             /* 0x0103d80000000400 */
             /*0a10*/               @P0 IMAD R20, R20, c[0x0][0x180], R15 ;                  /* 0x0000600014140a24 */
                                                                                             /* 0x000fe400078e020f */
             /*0a20*/               @P0 IMAD.MOV.U32 R21, RZ, RZ, 0x2 ;                      /* 0x00000002ff150424 */
                                                                                             /* 0x000fe200078e00ff */
             /*0a30*/              @!P0 PRMT R25, RZ, 0x7610, R25 ;                          /* 0x00007610ff198816 */
                                                                                             /* 0x002fc60000000019 */
             /*0a40*/               @P0 IMAD.WIDE R20, R20, R21, c[0x0][0x168] ;             /* 0x00005a0014140625 */
                                                                                             /* 0x000fc800078e0215 */
             /*0a50*/                   IMAD.IADD R22, R6, 0x1, R19 ;                        /* 0x0000000106167824 */
                                                                                             /* 0x001fe200078e0213 */
             /*0a60*/               @P0 LDG.E.U16.CONSTANT R25, [R20.64] ;                   /* 0x0000000614190981 */
                                                                                             /* 0x000f22000c1e9500 */
             /*0a70*/                   ISETP.GE.AND P0, PT, R17, c[0x0][0x180], PT ;        /* 0x0000600011007a0c */
                                                                                             /* 0x000fc80003f06270 */
             /*0a80*/                   ISETP.LT.AND P0, PT, R22, c[0x0][0x17c], !P0 ;       /* 0x00005f0016007a0c */
                                                                                             /* 0x000fe20004701270 */
             /*0a90*/                   STS.U16 [R29+0x200], R44 ;                           /* 0x0002002c1d007388 */
                                                                                             /* 0x0201d80000000400 */
             /*0aa0*/               @P0 IMAD R22, R22, c[0x0][0x180], R17 ;                  /* 0x0000600016160a24 */
                                                                                             /* 0x000fe400078e0211 */
             /*0ab0*/               @P0 IMAD.MOV.U32 R23, RZ, RZ, 0x2 ;                      /* 0x00000002ff170424 */
                                                                                             /* 0x000fe200078e00ff */
             /*0ac0*/              @!P0 PRMT R44, RZ, 0x7610, R44 ;                          /* 0x00007610ff2c8816 */
                                                                                             /* 0x001fc6000000002c */
             /*0ad0*/               @P0 IMAD.WIDE R22, R22, R23, c[0x0][0x168] ;             /* 0x00005a0016160625 */
                                                                                             /* 0x000fca00078e0217 */
             /*0ae0*/               @P0 LDG.E.U16.CONSTANT R44, [R22.64] ;                   /* 0x00000006162c0981 */
                                                                                             /* 0x000f68000c1e9500 */
             /*0af0*/                   STS.U16 [R7], R24 ;                                  /* 0x0000001807007388 */
                                                                                             /* 0x000fe80000000400 */
             /*0b00*/                   STS.U16 [R8], R43 ;                                  /* 0x0000002b08007388 */
                                                                                             /* 0x000fe80000000400 */
             /*0b10*/                   STS.U16 [R10], R45 ;                                 /* 0x0000002d0a007388 */
                                                                                             /* 0x004fe80000000400 */
             /*0b20*/                   STS.U16 [R12], R42 ;                                 /* 0x0000002a0c007388 */
                                                                                             /* 0x008fe80000000400 */
             /*0b30*/                   STS.U16 [R14], R25 ;                                 /* 0x000000190e007388 */
                                                                                             /* 0x010fe80000000400 */
             /*0b40*/                   STS.U16 [R16], R44 ;                                 /* 0x0000002c10007388 */
                                                                                             /* 0x020fe80000000400 */
             /*0b50*/                   BAR.SYNC 0x0 ;                                       /* 0x0000000000007b1d */

                                                                                                                                                            01:56 PM claude-sonnet-4-6
● Now the picture is completely clear. Between every pair of LDG instructions is a cluster of ISETP + IMAD instructions — these are:

  - ISETP — bounds checks (the if (glo_row < M && glo_col < K) guards in the source), computing the predicate @P0 that gates the load
  - IMAD / IMAD.WIDE — computing the next linearised element index, the row/column within the tile, and the final 64-bit global address into R20 or R22

  So R20 and R22 are not fixed pointers — they are recomputed fresh for each load iteration. The pattern per unrolled iteration is:

  ISETP ...          ← bounds check → predicate P0
  IMAD  R20 = ...    ← compute flat index → row/col
  IMAD.WIDE R20 = R20 * stride + base_ptr  ← full 64-bit address
  LDG   Rx, [R20.64] ← conditional load using that address

  The register R20 gets overwritten by each iteration's address computation, which is exactly why consecutive loads appear to use the same register name — it's the same register slot being
  recycled to hold a freshly computed address each time. The compiler is being efficient with address registers at the cost of not being able to issue all loads upfront, because each load's
  address depends on the IMAD chain immediately preceding it. That dependency chain is what forces the LDG to wait and prevents the scheduler from grouping all loads together the way Case 2
  manages to do.

✻ Churned for 30s

❯ Ah I see, they were separayed by a cluster of ISETP + IMAD instructions which overwrites the R20 and R22, so in this case, can the warp issuer issue ISETP + IMAD + later loads after
  issuing one load?

                                                                                                                                                            01:58 PM claude-sonnet-4-6
● Yes — and this is a key insight into GPU instruction scheduling. The warp issuer can issue the ISETP + IMAD cluster immediately after the LDG because those instructions don't read the
  LDG's destination register (e.g. R45). They operate on independent index variables. So there's no stall waiting for the LDG result.

  However there's a subtler stall to notice. Look at the dependency chain just before each subsequent LDG:

  /*0690*/  IMAD.WIDE R22, R22, R25, base   ← writes R22  (latency ~4 cycles)
  /*06a0*/  IMAD.IADD R25, ...              ← 1 instruction gap
  /*06b0*/  LDG R42, [R22.64]              ← reads R22  ← likely stalls ~3 cycles

  The IMAD.WIDE has ~4 cycle latency but only 1 instruction separates it from the LDG that consumes R22. So the warp issuer almost certainly stalls a few cycles at each LDG waiting for the
  address to be ready — not the hundreds of cycles of a memory stall, but a short arithmetic dependency stall.

  So the full picture per unrolled iteration is:

  LDG Rx, [R20]     → issued, warp moves on (memory in flight, ~200 cycle latency)
  ISETP             → issued immediately, no dependency on Rx
  IMAD              → issued immediately
  IMAD.WIDE R22=... → issued, ~4 cycle latency
  IMAD.IADD         → 1 gap instruction
  LDG Ry, [R22]    → ~3 cycle stall waiting for R22, then issued
  ...

  The crucial difference vs a CPU is that while this warp stalls those 3 cycles on the address dependency, the SM switches to another ready warp. So at high occupancy those short stalls get
  hidden for free. But at low occupancy (few active warps), even these short IMAD→LDG dependency stalls start showing up as wasted cycles.

  This is exactly why Case 2's schedule (all loads grouped, addresses pre-computed cleanly) is cleaner — though for these small Stage 2 tiles the difference is minor. It becomes more
  significant in Stage 3 where tile sizes and register pressure are much higher.

✻ Worked for 34s

❯ So the GPU does do out of order execution - if the destination register is not used - keep on issuing - but I heard that GPU doesn't really do much OOE before

                                                                                                                                                            02:01 PM claude-sonnet-4-6
● You're right to push back on that — it's a common point of confusion. The GPU is not doing true out-of-order execution. What's happening is more precisely described as in-order issue with
  scoreboarding.

  Here's the distinction:

  CPU OOO: Hardware has a reorder buffer + reservation stations. It dynamically scans a window of many instructions, finds independent ones, and issues them out of program order. The
  hardware is doing the dependency analysis at runtime.

  GPU: Issues instructions strictly in program order for a given warp. But it maintains a scoreboard — a table of which registers have in-flight writes pending. Before issuing each
  instruction it checks:
  - Are any of my source registers marked as pending? → stall this warp
  - No? → issue immediately, mark destination register as pending

  So after LDG R45, [R20]:
  - R45 gets marked pending in the scoreboard
  - The next ISETP reads R23, R30 — not pending → issues immediately, in order
  - The IMAD reads other index vars — not pending → issues in order
  - Eventually LDG R42, [R22] — checks R22 → if IMAD.WIDE just wrote it and it's not ready yet → stall

  When the warp stalls, the SM warp scheduler picks a different ready warp to issue from. That's the real latency hiding mechanism — not OOO, but warp switching. The compiler is responsible
  for ordering instructions to minimize stalls, which is why the SASS you saw has that specific interleaving of ISETP/IMAD between loads — the compiler scheduled them there to fill cycles
  while the previous load's address computation settles, and to keep the scoreboard clear for as long as possible before the next LDG needs the address register.

  So to directly answer: what looked like OOO is just the scoreboard letting the warp continue past the long-latency LDG because nothing downstream immediately needed R45. The hardware
  itself did no reordering.

──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────

