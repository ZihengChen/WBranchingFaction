//
// test from Mayda on expected event yields
// Ititialize acceptances*effy with values available on 
// Dec. 17, 2017 and provided by Nate Odell (superseed those from Ziheng Chen)
//
// Perform calculation of statistical uncertainty: Ziheng's analysis.
//
//    X_data[l] = X_data_top[l] / X_data_bot[l] right now has MC.
//    Will be changed to data after none-WW background is substracted
//
#include <iomanip>
#include <math.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include "TH1D.h"
#include "TH2D.h"
#include <TMath.h>
#include <TRandom.h>

const Double_t BFWe = 0.1080;
const Double_t BFWe_u = 0.0016;
const Double_t BFWm = 0.1080;
const Double_t BFWm_u = 0.0015;
const Double_t BFWt = 0.1080;
const Double_t BFWt_u = 0.0021;
const Double_t BFWh = 1.-3.*BFWe;  //Assume lepton universality in the MC
const Double_t BFWh_u = 0.0027;
const Double_t BFte = 0.1782;   // MC ~0.1776
const Double_t BFte_u = 0.0004;
const Double_t BFtm = 0.1739;   // MC ~0.1731
const Double_t BFtm_u = 0.0004;
const Double_t BFth = 1.-BFte-BFtm;  // MC ~0.6492
const Double_t BFth_u = 0.0006;

const Double_t Rw_em = BFWe/BFWm;
const Double_t Rw_eh = BFWe/BFWh;
const Double_t Rw_hm = BFWh/BFWm;
const Double_t Rt_em = BFte/BFtm;
const Double_t Rt_hm = BFth/BFtm;
const Double_t Rt_mh = BFtm/BFth;
const Double_t Rt_eh = BFte/BFth;
const Double_t Rt_he = BFth/BFte;
const Double_t Rt_me = BFtm/BFte;

const Double_t R_tm = (BFWt * BFtm) / BFWm;
const Double_t R_th = (BFWt * BFth) / BFWm;
const Double_t R_te = (BFWt * BFte) / BFWm;

const Double_t Eff_A[2][2]    = {1., 1., 1., 1.};     // mm  Effy of NN
const Double_t Eff_B[2][2]    = {1., 1., 1., 1.};     // mm  "
const Double_t Eff_C[2][2]    = {1., 1., 1., 1.};     // mm  "

const Double_t Eff_Ah[2][2]    = {1., 1., 1., 1.};     // mm  Effy of NN
const Double_t Eff_Bh[2][2]    = {1., 1., 1., 1.};     // mm  "
const Double_t Eff_Ch[2][2]    = {1., 1., 1., 1.};     // mm  "

const Double_t Eff_Ae[2][2]    = {1., 1., 1., 1.};     // mm  Effy of NN
const Double_t Eff_Be[2][2]    = {1., 1., 1., 1.};     // mm  "
const Double_t Eff_Ce[2][2]    = {1., 1., 1., 1.};     // mm  "

const Double_t Eff_mm[2][2]   = {1., 1., 1., 1.};     // mm  Effy of NN
const Double_t Eff_mtm[2][2]  = {1., 1., 1., 1.};     // mm  "
const Double_t Eff_tmtm[2][2] = {1., 1., 1., 1.};     // mm  "

const Double_t Eff_mh[2][2]   = {1., 1., 1., 1.};     // mm  Effy of NN
const Double_t Eff_mth[2][2]  = {1., 1., 1., 1.};     // mm  "
const Double_t Eff_tmth[2][2] = {1., 1., 1., 1.};     // mm  "

const Double_t Eff_me[2][2]   = {1., 1., 1., 1.};     // mm  Effy of NN
const Double_t Eff_mte_etm[2][2]  = {1., 1., 1., 1.};     // mm  "
const Double_t Eff_tmte[2][2] = {1., 1., 1., 1.};     // mm  "

void New_R_Calc() {

Double_t A_ee [2][4][2];
Double_t A_mm [2][4][2];
Double_t A_hh [2][4][2];
Double_t A_me [2][4][2];
Double_t A_eh [2][4][2];
Double_t A_mh [2][4][2];
Double_t A_mte [2][4][2];
Double_t A_mtm [2][4][2];
Double_t A_mth [2][4][2];
Double_t A_ete [2][4][2];
Double_t A_etm [2][4][2];
Double_t A_eth [2][4][2];
Double_t A_hte [2][4][2];
Double_t A_htm [2][4][2];
Double_t A_hth [2][4][2];
Double_t A_tete [2][4][2];
Double_t A_tmtm [2][4][2];
Double_t A_thth [2][4][2];
Double_t A_tmte [2][4][2];
Double_t A_teth [2][4][2];
Double_t A_tmth [2][4][2];

//
//  i= 0 tt; i = 1  tW
//  j= 0 mm; j= 1 em;  j= 2 mth;  j= 3 m4j 
//  k= 0 1b; k = 1  >=2
//
// mm  tt 1b
   A_ee[0][0][0]  = 2. / 1811409. ; 
   A_mm[0][0][0]  = 325249. / 1811532. ;
   A_hh[0][0][0]  = 268. / 70930278. ; 
   A_me[0][0][0]  = 2201. / 3620281. ;
   A_eh[0][0][0]  = 47. / 22670017. ; 
   A_mh[0][0][0]  = 30213. / 22653517. ;
   A_mte[0][0][0] = 456. / 643368. ;  
   A_mtm[0][0][0] = 73982. / 627277. ;
   A_mth[0][0][0] = 2315. / 2353727. ;  
   A_ete[0][0][0] = 2. / 642646. ; 
   A_etm[0][0][0] = 166. / 626108. ;
   A_eth[0][0][0] = 3. / 2350887. ; 
   A_hte[0][0][0] = 13. / 4016590. ;
   A_htm[0][0][0] = 2072. / 3923504. ; 
   A_hth[0][0][0] = 44. / 14718933. ;
   A_tete[0][0][0]= 1. / 57057. ; 
   A_tmtm[0][0][0]= 3296. / 54227. ; 
   A_thth[0][0][0]= 1. / 763147. ;  
   A_tmte[0][0][0]= 36. / 111125. ; 
   A_teth[0][0][0]= 1. / 418126. ; 
   A_tmth[0][0][0]= 178. / 407605. ; 
// em tt 1b
   A_ee[0][1][0]   = 100. / 1811409. ; 
   A_mm[0][1][0]   = 90. / 1811532. ;
   A_hh[0][1][0]   = 4. / 70930278. ; 
   A_me[0][1][0]   = 337122. / 3620281. ;
   A_eh[0][1][0]   = 1535. / 22670017. ; 
   A_mh[0][1][0]   = 2739. / 22653517. ;
   A_mte[0][1][0]  = 29636. / 643368. ;  
   A_mtm[0][1][0]  = 25. / 627277. ;
   A_mth[0][1][0]  = 677. / 2353727. ;  
   A_ete[0][1][0]  = 34. / 642646. ; 
   A_etm[0][1][0]  = 22293. / 626108. ;
   A_eth[0][1][0]  = 111. / 2350887. ; 
   A_hte[0][1][0]  = 123. / 4016590. ;
   A_htm[0][1][0]  = 162. / 3923504. ; 
   A_hth[0][1][0]  = 3. / 14718933. ;
   A_tete[0][1][0] = 1. / 57057. ; 
   A_tmtm[0][1][0] = 0. / 54227. ; 
   A_thth[0][1][0] = 0. / 763147. ;  
   A_tmte[0][1][0] = 1968. / 111125. ; 
   A_teth[0][1][0] = 7. / 418126. ; 
   A_tmth[0][1][0] = 42. / 407605. ; 
// mth  tt 1b
   A_ee[0][2][0]   = 1. / 1811409. ; 
   A_mm[0][2][0]   = 1063. / 1811532. ;
   A_hh[0][2][0]   = 317. / 70930278. ; 
   A_me[0][2][0]   = 3262. / 3620281. ;
   A_eh[0][2][0]   = 29. / 22670017. ; 
   A_mh[0][2][0]   = 103617. / 22653517. ;
   A_mte[0][2][0]  = 676. / 643368. ;  
   A_mtm[0][2][0]  = 402. / 627277. ;
   A_mth[0][2][0]  = 69184. / 2353727. ;  
   A_ete[0][2][0]  = 0. / 642646. ; 
   A_etm[0][2][0]  = 208. / 626108. ;
   A_eth[0][2][0]  = 22. / 2350887. ; 
   A_hte[0][2][0]  = 13. / 4016590. ;
   A_htm[0][2][0]  = 6587. / 3923504. ; 
   A_hth[0][2][0]  = 337. / 14718933. ;
   A_tete[0][2][0] = 0. / 57057. ; 
   A_tmtm[0][2][0] = 28. / 54227. ; 
   A_thth[0][2][0] = 24. / 763147. ;  
   A_tmte[0][2][0] = 49. / 111125. ; 
   A_teth[0][2][0] = 5. / 418126. ; 
   A_tmth[0][2][0] = 4466. / 407605. ; 
// m4j tt 1b
   A_ee[0][3][0]   = 9. / 1811409. ; 
   A_mm[0][3][0]   = 33469. / 1811532. ;
   A_hh[0][3][0]   = 4663. / 70930278. ; 
   A_me[0][3][0]   = 109068. / 3620281. ;
   A_eh[0][3][0]   = 459. / 22670017. ; 
   A_mh[0][3][0]   = 2486649. / 22653517. ;
   A_mte[0][3][0]  = 19507. / 643368. ;  
   A_mtm[0][3][0]  = 11584. / 627277. ;
   A_mth[0][3][0]  = 101684. / 2353727. ;  
   A_ete[0][3][0]  = 6. / 642646. ; 
   A_etm[0][3][0]  = 7170. / 626108. ;
   A_eth[0][3][0]  = 16. / 2350887. ; 
   A_hte[0][3][0]  = 101. / 4016590. ;
   A_htm[0][3][0]  = 164257. / 3923504. ; 
   A_hth[0][3][0]  = 542. / 14718933. ;
   A_tete[0][3][0] = 0. / 57057. ; 
   A_tmtm[0][3][0] = 631. / 54227. ; 
   A_thth[0][3][0] = 10. / 763147. ;  
   A_tmte[0][3][0] = 1292. / 111125. ; 
   A_teth[0][3][0] = 2. / 418126. ; 
   A_tmth[0][3][0] = 6886. / 407605. ; 
//
// tt >=2b
//
//  mm 
   A_ee[0][0][1]   = 0. / 1811409. ; 
   A_mm[0][0][1]   = 97385. / 1811532. ;
   A_hh[0][0][1]   = 18. / 70930278. ; 
   A_me[0][0][1]   = 250. / 3620281. ;
   A_eh[0][0][1]   = 4. / 22670017. ; 
   A_mh[0][0][1]   = 3617. / 22653517. ;
   A_mte[0][0][1]  = 53. / 643368. ;  
   A_mtm[0][0][1]  = 21636. / 627277. ;
   A_mth[0][0][1]  = 306. / 2353727. ;  
   A_ete[0][0][1]  = 0. / 642646. ; 
   A_etm[0][0][1]  = 21. / 626108. ;
   A_eth[0][0][1]  = 0. / 2350887. ; 
   A_hte[0][0][1]  = 1. / 4016590. ;
   A_htm[0][0][1]  = 245. / 3923504. ; 
   A_hth[0][0][1]  = 3. / 14718933. ;
   A_tete[0][0][1] = 0. / 57057. ; 
   A_tmtm[0][0][1] = 967. / 54227. ; 
   A_thth[0][0][1] = 0. / 763147. ;  
   A_tmte[0][0][1] = 7. / 111125. ; 
   A_teth[0][0][1] = 0. / 418126. ; 
   A_tmth[0][0][1] = 26. / 407605. ; 
// em
   A_ee[0][1][1]   = 11. / 1811409. ; 
   A_mm[0][1][1]   = 19. / 1811532. ;
   A_hh[0][1][1]   = 0. / 70930278. ; 
   A_me[0][1][1]   = 101048. / 3620281. ;
   A_eh[0][1][1]   = 210. / 22670017. ; 
   A_mh[0][1][1]   = 283. / 22653517. ;
   A_mte[0][1][1]  = 8866. / 643368. ;  
   A_mtm[0][1][1]  = 7. / 627277. ;
   A_mth[0][1][1]  = 143. / 2353727. ;  
   A_ete[0][1][1]  = 6. / 642646. ; 
   A_etm[0][1][1]  = 6481. / 626108. ;
   A_eth[0][1][1]  = 15. / 2350887. ; 
   A_hte[0][1][1]  = 15. / 4016590. ;
   A_htm[0][1][1]  = 24. / 3923504. ; 
   A_hth[0][1][1]  = 1. / 14718933. ;
   A_tete[0][1][1] = 0. / 57057. ; 
   A_tmtm[0][1][1] = 0. / 54227. ; 
   A_thth[0][1][1] = 0. / 763147. ;  
   A_tmte[0][1][1] = 613. / 111125. ; 
   A_teth[0][1][1] = 2. / 418126. ; 
   A_tmth[0][1][1] = 15. / 407605. ; 
// mth
   A_ee[0][2][1]   = 0. / 1811409. ; 
   A_mm[0][2][1]   = 219. / 1811532. ;
   A_hh[0][2][1]   = 50. / 70930278. ; 
   A_me[0][2][1]   = 546. / 3620281. ;
   A_eh[0][2][1]   = 3. / 22670017. ; 
   A_mh[0][2][1]   = 23598. / 22653517. ;
   A_mte[0][2][1]  = 144. / 643368. ;  
   A_mtm[0][2][1]  = 74. / 627277. ;
   A_mth[0][2][1]  = 20258. / 2353727. ;  
   A_ete[0][2][1]  = 0. / 642646. ; 
   A_etm[0][2][1]  = 30. / 626108. ;
   A_eth[0][2][1]  = 4. / 2350887. ; 
   A_hte[0][2][1]  = 3. / 4016590. ;
   A_htm[0][2][1]  = 1475. / 3923504. ; 
   A_hth[0][2][1]  = 33. / 14718933. ;
   A_tete[0][2][1] = 0. / 57057. ; 
   A_tmtm[0][2][1] = 5. / 54227. ; 
   A_thth[0][2][1] = 3. / 763147. ;  
   A_tmte[0][2][1] = 9. / 111125. ; 
   A_teth[0][2][1] = 1. / 418126. ; 
   A_tmth[0][2][1] = 1332. / 407605. ; 
// m4j
   A_ee[0][3][1]   = 0. / 1811409. ; 
   A_mm[0][3][1]   = 10936. / 1811532. ;
   A_hh[0][3][1]   = 451. / 70930278. ; 
   A_me[0][3][1]   = 36165. / 3620281. ;
   A_eh[0][3][1]   = 35. / 22670017. ; 
   A_mh[0][3][1]   = 805463. / 22653517. ;
   A_mte[0][3][1]  = 6692. / 643368. ;  
   A_mtm[0][3][1]  = 3949. / 627277. ;
   A_mth[0][3][1]  = 34219. / 2353727. ;  
   A_ete[0][3][1]  = 0. / 642646. ; 
   A_etm[0][3][1]  = 2344. / 626108. ;
   A_eth[0][3][1]  = 1. / 2350887. ; 
   A_hte[0][3][1]  = 9. / 4016590. ;
   A_htm[0][3][1]  = 52759. / 3923504. ; 
   A_hth[0][3][1]  = 66. / 14718933. ;
   A_tete[0][3][1] = 1. / 57057. ; 
   A_tmtm[0][3][1] = 212. / 54227. ; 
   A_thth[0][3][1] = 2. / 763147. ;  
   A_tmte[0][3][1] = 424. / 111125. ; 
   A_teth[0][3][1] = 0. / 418126. ; 
   A_tmth[0][3][1] = 2307. / 407605. ; 
//
//  i = 0 tt; i = 1 tW
//  j = 0 mm; j = 1 em;  j = 2 mth;  j = 3 m4j 
//  k = 0 1b; k = 1 >=2
//
// mm  tW 1b
   A_ee[1][0][0]   = 0. / 23105. ; 
   A_mm[1][0][0]   = 2181. / 23040. ;
   A_hh[1][0][0]   = 2. / 904067. ; 
   A_me[1][0][0]   = 18. / 46342. ;
   A_eh[1][0][0]   = 0. / 290101. ; 
   A_mh[1][0][0]   = 323. / 289467. ;
   A_mte[1][0][0]  = 7. / 8163. ;  
   A_mtm[1][0][0]  = 528. / 8006. ;
   A_mth[1][0][0]  = 24. / 29970. ;  
   A_ete[1][0][0]  = 0. / 8199. ; 
   A_etm[1][0][0]  = 2. / 8029. ;
   A_eth[1][0][0]  = 0. / 30054. ; 
   A_hte[1][0][0]  = 0. / 50864. ;
   A_htm[1][0][0]  = 17. / 50050. ; 
   A_hth[1][0][0]  = 1. / 187515. ;
   A_tete[1][0][0] = 0. / 694. ; 
   A_tmtm[1][0][0] = 28. / 717. ; 
   A_thth[1][0][0] = 0. / 9727. ; 
   A_tmte[1][0][0] = 1. / 1420. ; 
   A_teth[1][0][0] = 0. / 5260. ; 
   A_tmth[1][0][0] = 1. / 5158. ; 
// em  tW 1b
   A_ee[1][1][0]   = 1. / 23105. ; 
   A_mm[1][1][0]   = 1. / 23040. ;
   A_hh[1][1][0]   = 0. / 904067. ; 
   A_me[1][1][0]   = 2360. / 46342. ;
   A_eh[1][1][0]   = 8. / 290101. ; 
   A_mh[1][1][0]   = 19. / 289467. ;
   A_mte[1][1][0]  = 222. / 8163. ;  
   A_mtm[1][1][0]  = 0. / 8006. ;
   A_mth[1][1][0]  = 9. / 29970. ;  
   A_ete[1][1][0]  = 1. / 8199. ; 
   A_etm[1][1][0]  = 165. / 8029. ;
   A_eth[1][1][0]  = 0. / 30054. ; 
   A_hte[1][1][0]  = 2. / 50864. ;
   A_htm[1][1][0]  = 1. / 50050. ; 
   A_hth[1][1][0]  = 0. / 187515. ;
   A_tete[1][1][0] = 0. / 694. ; 
   A_tmtm[1][1][0] = 0. / 717. ; 
   A_thth[1][1][0] = 0. / 9727. ; 
   A_tmte[1][1][0] = 11. / 1420. ; 
   A_teth[1][1][0] = 0. / 5260. ; 
   A_tmth[1][1][0] = 0. / 5158. ; 
// mth tW 1b
   A_ee[1][2][0]   = 0. / 23105. ; 
   A_mm[1][2][0]   = 3. / 23040. ;
   A_hh[1][2][0]   = 3. / 904067. ; 
   A_me[1][2][0]   = 20. / 46342. ;
   A_eh[1][2][0]   = 1. / 290101. ; 
   A_mh[1][2][0]   = 935. / 289467. ;
   A_mte[1][2][0]  = 0. / 8163. ;  
   A_mtm[1][2][0]  = 1. / 8006. ;
   A_mth[1][2][0]  = 513. / 29970. ;  
   A_ete[1][2][0]  = 0. / 8199. ; 
   A_etm[1][2][0]  = 1. / 8029. ;
   A_eth[1][2][0]  = 0. / 30054. ; 
   A_hte[1][2][0]  = 0. / 50864. ;
   A_htm[1][2][0]  = 74. / 50050. ; 
   A_hth[1][2][0]  = 1. / 187515. ;
   A_tete[1][2][0] = 0. / 694. ; 
   A_tmtm[1][2][0] = 0. / 717. ; 
   A_thth[1][2][0] = 0. / 9727. ; 
   A_tmte[1][2][0] = 1. / 1420. ; 
   A_teth[1][2][0] = 0. / 5260. ; 
   A_tmth[1][2][0] = 32. / 5158. ; 
// m4j tW 1b
   A_ee[1][3][0]   = 0. / 23105. ; 
   A_mm[1][3][0]   = 131. / 23040. ;
   A_hh[1][3][0]   = 21. / 904067. ; 
   A_me[1][3][0]   = 475. / 46342. ;
   A_eh[1][3][0]   = 0. / 290101. ; 
   A_mh[1][3][0]   = 15124. / 289467. ;
   A_mte[1][3][0]  = 85. / 8163. ;  
   A_mtm[1][3][0]  = 49. / 8006. ;
   A_mth[1][3][0]  = 467. / 29970. ;  
   A_ete[1][3][0]  = 0. / 8199. ; 
   A_etm[1][3][0]  = 39. / 8029. ;
   A_eth[1][3][0]  = 0. / 30054. ; 
   A_hte[1][3][0]  = 0. / 50864. ;
   A_htm[1][3][0]  = 1150. / 50050. ; 
   A_hth[1][3][0]  = 4. / 187515. ;
   A_tete[1][3][0] = 0. / 694. ; 
   A_tmtm[1][3][0] = 2. / 717. ; 
   A_thth[1][3][0] = 0. / 9727. ; 
   A_tmte[1][3][0] = 9. / 1420. ; 
   A_teth[1][3][0] = 0. / 5260. ; 
   A_tmth[1][3][0] = 25. / 5158. ; 
//
// mm  tW >=2b
//
   A_ee[1][0][1]   = 0. / 23105. ; 
   A_mm[1][0][1]   = 361. / 23040. ;
   A_hh[1][0][1]   = 1. / 904067. ; 
   A_me[1][0][1]   = 4. / 46342. ;
   A_eh[1][0][1]   = 0. / 290101. ; 
   A_mh[1][0][1]   = 13. / 289467. ;
   A_mte[1][0][1]  = 0. / 8163. ;  
   A_mtm[1][0][1]  = 95. / 8006. ;
   A_mth[1][0][1]  = 4. / 29970. ;  
   A_ete[1][0][1]  = 0. / 8199. ; 
   A_etm[1][0][1]  = 0. / 8029. ;
   A_eth[1][0][1]  = 0. / 30054. ; 
   A_hte[1][0][1]  = 0. / 50864. ;
   A_htm[1][0][1]  = 1. / 50050. ; 
   A_hth[1][0][1]  = 0. / 187515. ;
   A_tete[1][0][1] = 0. / 694. ; 
   A_tmtm[1][0][1] = 4. / 717. ; 
   A_thth[1][0][1] = 0. / 9727. ; 
   A_tmte[1][0][1] = 0. / 1420. ; 
   A_teth[1][0][1] = 0. / 5260. ; 
   A_tmth[1][0][1] = 0. / 5158. ; 
// em  tW >=2b
   A_ee[1][1][1]   = 0. / 23105. ; 
   A_mm[1][1][1]   = 0. / 23040. ;
   A_hh[1][1][1]   = 0. / 904067. ; 
   A_me[1][1][1]   = 413. / 46342. ;
   A_eh[1][1][1]   = 2. / 290101. ; 
   A_mh[1][1][1]   = 1. / 289467. ;
   A_mte[1][1][1]  = 49. / 8163. ;  
   A_mtm[1][1][1]  = 0. / 8006. ;
   A_mth[1][1][1]  = 1. / 29970. ;  
   A_ete[1][1][1]  = 0. / 8199. ; 
   A_etm[1][1][1]  = 26. / 8029. ;
   A_eth[1][1][1]  = 0. / 30054. ; 
   A_hte[1][1][1]  = 0. / 50864. ;
   A_htm[1][1][1]  = 0. / 50050. ; 
   A_hth[1][1][1]  = 0. / 187515. ;
   A_tete[1][1][1] = 0. / 694. ; 
   A_tmtm[1][1][1] = 0. / 717. ; 
   A_thth[1][1][1] = 0. / 9727. ; 
   A_tmte[1][1][1] = 4. / 1420. ; 
   A_teth[1][1][1] = 0. / 5260. ; 
   A_tmth[1][1][1] = 0. / 5158. ; 
// mth tW >=2b
   A_ee[1][2][1]   = 0. / 23105. ; 
   A_mm[1][2][1]   = 0. / 23040. ;
   A_hh[1][2][1]   = 0. / 904067. ; 
   A_me[1][2][1]   = 0. / 46342. ;
   A_eh[1][2][1]   = 0. / 290101. ; 
   A_mh[1][2][1]   = 81. / 289467. ;
   A_mte[1][2][1]  = 0. / 8163. ;  
   A_mtm[1][2][1]  = 0. / 8006. ;
   A_mth[1][2][1]  = 106. / 29970. ;  
   A_ete[1][2][1]  = 0. / 8199. ; 
   A_etm[1][2][1]  = 0. / 8029. ;
   A_eth[1][2][1]  = 0. / 30054. ; 
   A_hte[1][2][1]  = 0. / 50864. ;
   A_htm[1][2][1]  = 4. / 50050. ; 
   A_hth[1][2][1]  = 0. / 187515. ;
   A_tete[1][2][1] = 0. / 694. ; 
   A_tmtm[1][2][1] = 0. / 717. ; 
   A_thth[1][2][1] = 0. / 9727. ; 
   A_tmte[1][2][1] = 0. / 1420. ; 
   A_teth[1][2][1] = 0. / 5260. ; 
   A_tmth[1][2][1] = 7. / 5158. ; 
// m4j tW >=2b
   A_ee[1][3][1]   = 0. / 23105. ; 
   A_mm[1][3][1]   = 31. / 23040. ;
   A_hh[1][3][1]   = 1. / 904067. ; 
   A_me[1][3][1]   = 100. / 46342. ;
   A_eh[1][3][1]   = 0. / 290101. ; 
   A_mh[1][3][1]   = 3046. / 289467. ;
   A_mte[1][3][1]  = 14. / 8163. ;  
   A_mtm[1][3][1]  = 10. / 8006. ;
   A_mth[1][3][1]  = 117. / 29970. ;  
   A_ete[1][3][1]  = 0. / 8199. ; 
   A_etm[1][3][1]  = 6. / 8029. ;
   A_eth[1][3][1]  = 0. / 30054. ; 
   A_hte[1][3][1]  = 0. / 50864. ;
   A_htm[1][3][1]  = 207. / 50050. ; 
   A_hth[1][3][1]  = 1. / 187515. ;
   A_tete[1][3][1] = 0. / 694. ; 
   A_tmtm[1][3][1] = 0. / 717. ; 
   A_thth[1][3][1] = 0. / 9727. ; 
   A_tmte[1][3][1] = 1. / 1420. ; 
   A_teth[1][3][1] = 0. / 5260. ; 
   A_tmth[1][3][1] = 6. / 5158. ; 
//
// Yields for each of the 21 channels
//
int i, j, k, l;
Double_t Yield [21][2][4][2];

for (j=0; j<2; j++){   // loop over production type
    for (k=0; k<4; k++){   // loop over channel type
       for (l=0; l<2; l++){   // loop #b category Yield_[i,j,k,l]
//
// rewrite to be a function of R = (W-to-tau to tau-to-mu)/W-to-mu 
//
         Yield[0][j][k][l]  =      BFWm * BFWm * Rw_em * Rw_em * A_ee[j][k][l] ;
         Yield[1][j][k][l]  =      BFWm * BFWm *                 A_mm[j][k][l] ;
         Yield[2][j][k][l]  =      BFWm * BFWm * Rw_hm * Rw_hm * A_hh[j][k][l] ;
         Yield[3][j][k][l]  = 2. * BFWm * BFWm * Rw_em *         A_me[j][k][l] ;
         Yield[4][j][k][l]  = 2. * BFWm * BFWm * Rw_em * Rw_hm * A_eh[j][k][l] ;
         Yield[5][j][k][l]  = 2. * BFWm * BFWm * Rw_hm *         A_mh[j][k][l] ;

         Yield[6][j][k][l]  = 2. * BFWm * BFWm *         Rt_em * R_tm * A_mte[j][k][l] ;
         Yield[7][j][k][l]  = 2. * BFWm * BFWm *                 R_tm * A_mtm[j][k][l] ;
         Yield[8][j][k][l]  = 2. * BFWm * BFWm *         Rt_hm * R_tm * A_mth[j][k][l] ;
         Yield[9][j][k][l]  = 2. * BFWm * BFWm * Rw_em * Rt_em * R_tm * A_ete[j][k][l] ;
         Yield[10][j][k][l] = 2. * BFWm * BFWm * Rw_em *         R_tm * A_etm[j][k][l] ;
         Yield[11][j][k][l] = 2. * BFWm * BFWm * Rw_em * Rt_hm * R_tm * A_eth[j][k][l] ;
         Yield[12][j][k][l] = 2. * BFWm * BFWm * Rw_hm * Rt_em * R_tm * A_hte[j][k][l] ;
         Yield[13][j][k][l] = 2. * BFWm * BFWm * Rw_hm *         R_tm * A_htm[j][k][l] ;
         Yield[14][j][k][l] = 2. * BFWm * BFWm * Rw_hm * Rt_hm * R_tm * A_hth[j][k][l] ;

         Yield[15][j][k][l] = 2. * BFWm * BFWm * Rt_em *         R_tm * R_tm * A_tmte[j][k][l] ;
         Yield[16][j][k][l] = 2. * BFWm * BFWm * Rt_em * Rt_hm * R_tm * R_tm * A_teth[j][k][l] ;
         Yield[17][j][k][l] = 2. * BFWm * BFWm *         Rt_hm * R_tm * R_tm * A_tmth[j][k][l] ;
         Yield[18][j][k][l] =      BFWm * BFWm * Rt_em * Rt_em * R_tm * R_tm * A_tete[j][k][l] ;
         Yield[19][j][k][l] =      BFWm * BFWm *                 R_tm * R_tm * A_tmtm[j][k][l] ;
         Yield[20][j][k][l] =      BFWm * BFWm * Rt_hm * Rt_hm * R_tm * R_tm * A_thth[j][k][l] ;
      }
   }
}

 Double_t crossLumi[2] = {832. * 1000. * 36. ,  35.6 * 1000. * 36.} ;
 Double_t Sum_Yield [2][4][2];

 cout << "Yield with R_mu factorization "  << endl;

 for (j=0; j<2; j++){   // loop over production type
    for (k= 0; k<4; k++){   // loop over channel type
       for (l= 0; l<2; l++){   // loop #b category Yield_[i,j,k,l]  
         Sum_Yield[j][k][l] = 0.0;
         for (i= 0; i<21; i++){   // loop over the 21 channels
            Sum_Yield[j][k][l] = Sum_Yield[j][k][l] + Yield[i][j][k][l];
         }
           if (l==0 && k==0 && j==0) cout << "Yield for ttbar [mm, 1b]  = " << Sum_Yield[j][k][l]*crossLumi[0] << endl;
           if (l==1 && k==0 && j==0) cout << "Yield for ttbar [mm, 2b]  = " << Sum_Yield[j][k][l]*crossLumi[0] << endl;
           if (l==0 && k==0 && j==1) cout << "Yield for tW [mm, 1b]  = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
           if (l==1 && k==0 && j==1) cout << "Yield for tW [mm, 2b]  = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
           if (l==0 && k==1 && j==0) cout << "Yield for ttbar [em, 1b]  = " << Sum_Yield[j][k][l]*crossLumi[0] << endl;
           if (l==1 && k==1 && j==0) cout << "Yield for ttbar [em, 2b]  = " << Sum_Yield[j][k][l]*crossLumi[0] << endl;
           if (l==0 && k==1 && j==1) cout << "Yield for tW [em, 1b]  = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
           if (l==1 && k==1 && j==1) cout << "Yield for tW [em, 2b]  = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
           if (l==0 && k==2 && j==0) cout << "Yield for ttbar [mth, 1b]  = " << Sum_Yield[j][k][l]*crossLumi[0] << endl;
           if (l==1 && k==2 && j==0) cout << "Yield for ttbar [mth, 2b]  = " << Sum_Yield[j][k][l]*crossLumi[0] << endl;
           if (l==0 && k==2 && j==1) cout << "Yield for tW [mth, 1b]  = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
           if (l==1 && k==2 && j==1) cout << "Yield for tW [mth, 2b]  = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
           if (l==0 && k==3 && j==0) cout << "Yield for ttbar [m4j, 1b]  = " << Sum_Yield[j][k][l]*crossLumi[0] << endl;
           if (l==1 && k==3 && j==0) cout << "Yield for ttbar [m4j, 2b]  = " << Sum_Yield[j][k][l]*crossLumi[0] << endl;
           if (l==0 && k==3 && j==1) cout << "Yield for tW [m4j, 1b]  = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
           if (l==1 && k==3 && j==1) cout << "Yield for tW [m4j, 2b]  = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
      }   
   }
 } 

//
//  Test for extracting  R_mu
//

 Double_t X_data [2];     // mm in the 1b and >1b case
 Double_t X_data_top [2];
 Double_t X_data_bot [2];

 Double_t  beta[2][2];     // mm in the 1b and >1b case / process
 Double_t alpha[2][2];     // mm in the 1b and >1b case "
 Double_t gamma[2][2];     // mm in the 1b and >1b case "

 Double_t BKG_A[2][2];     // mm  WW tautau background  "
 Double_t BKG_B[2][2];     // mm  WW tau background  "
 Double_t BKG_C[2][2];     // mm  WW background  "


 Double_t  R_mp,  R_mn ; 
 Double_t  A, B, C; 
 Double_t  A_mm_save[2], A_mtm_save[2], A_tmtm_save[2];      
 Double_t  A_mh_save[2], A_mth_save[2], A_tmth_save[2];      
 Double_t  A_me_save[2], A_mte_save[2], A_etm_save[2], A_tmte_save[2];      

 Double_t XSec[2];

    for (l= 0; l<2; l++){   // loop #b category Yield_[i,j,k,l]  

       X_data_top[l] = 0.0;
       X_data_bot[l] = 0.0;
       X_data[l]  = 0.0;

       for (j=0; j<2; j++){   // loop over production type

         alpha[l][j] = 0.0;    
         gamma[l][j] = 0.0;    
         beta[l][j] = 0.0;     

         BKG_A[l][j] = 0.0;    
         BKG_B[l][j] = 0.0;    
         BKG_C[l][j] = 0.0;     

         XSec[j] = crossLumi[j]  ;

         for (k= 0; k<4; k++){   // loop over channel type

             X_data_bot[l] = X_data_bot[l] + Sum_Yield[j][k][l]*crossLumi[j] ;

             if(k==0){

             X_data_top[l] = X_data_top[l] + Sum_Yield[j][k][l]*crossLumi[j]  ;


             BKG_C[l][j] = BKG_C[l][j] +      Rw_em * Rw_em * A_ee[j][k][l]*XSec[j] ;
             BKG_C[l][j] = BKG_C[l][j] +      Rw_hm * Rw_hm * A_hh[j][k][l]*XSec[j] ;
             BKG_C[l][j] = BKG_C[l][j] + 2. * Rw_em *         A_me[j][k][l]*XSec[j] ;
             BKG_C[l][j] = BKG_C[l][j] + 2. * Rw_em * Rw_hm * A_eh[j][k][l]*XSec[j] ;
             BKG_C[l][j] = BKG_C[l][j] + 2. * Rw_hm *         A_mh[j][k][l]*XSec[j] ;

             BKG_B[l][j] = BKG_B[l][j] + 2. * Rt_em *         A_mte[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. * Rt_hm *         A_mth[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. * Rw_em * Rt_em * A_ete[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. * Rw_em *         A_etm[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. * Rw_em * Rt_hm * A_eth[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. * Rw_hm * Rt_em * A_hte[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. * Rw_hm *         A_htm[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. * Rw_hm * Rt_hm * A_hth[j][k][l]*XSec[j] ;

             BKG_A[l][j] = BKG_A[l][j] + 2. * Rt_em *         A_tmte[j][k][l]*XSec[j] ;
             BKG_A[l][j] = BKG_A[l][j] + 2. * Rt_em * Rt_hm * A_teth[j][k][l]*XSec[j] ;
             BKG_A[l][j] = BKG_A[l][j] + 2. * Rt_hm *         A_tmth[j][k][l]*XSec[j] ;
             BKG_A[l][j] = BKG_A[l][j] +      Rt_em * Rt_em * A_tete[j][k][l]*XSec[j] ;
             BKG_A[l][j] = BKG_A[l][j] +      Rt_hm * Rt_hm * A_thth[j][k][l]*XSec[j] ;
            
             A_mm_save[j] = A_mm[j][k][l]*XSec[j]; 
             A_mtm_save[j] = A_mtm[j][k][l]*XSec[j];  
             A_tmtm_save[j] = A_tmtm[j][k][l]*XSec[j];      
             }
//
// mm gamma terms
// 
            gamma[l][j] = gamma[l][j] +                      A_mm[j][k][l]*XSec[j] ;
            gamma[l][j] = gamma[l][j] +      Rw_em * Rw_em * A_ee[j][k][l]*XSec[j] ;
            gamma[l][j] = gamma[l][j] +      Rw_hm * Rw_hm * A_hh[j][k][l]*XSec[j] ;
            gamma[l][j] = gamma[l][j] + 2. * Rw_em *         A_me[j][k][l]*XSec[j] ;
            gamma[l][j] = gamma[l][j] + 2. * Rw_em * Rw_hm * A_eh[j][k][l]*XSec[j] ;
            gamma[l][j] = gamma[l][j] + 2. * Rw_hm *         A_mh[j][k][l]*XSec[j] ;
//
// mtm terms
//
            beta[l][j] = beta[l][j] + 2. *                 A_mtm[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rt_em *         A_mte[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rt_hm *         A_mth[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rw_em * Rt_em * A_ete[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rw_em *         A_etm[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rw_em * Rt_hm * A_eth[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rw_hm * Rt_em * A_hte[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rw_hm *         A_htm[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rw_hm * Rt_hm * A_hth[j][k][l]*XSec[j] ;

//
// tmtm terms
//
            alpha[l][j] = alpha[l][j] +                      A_tmtm[j][k][l]*XSec[j] ;
            alpha[l][j] = alpha[l][j] + 2. * Rt_em *         A_tmte[j][k][l]*XSec[j] ;
            alpha[l][j] = alpha[l][j] + 2. * Rt_em * Rt_hm * A_teth[j][k][l]*XSec[j] ;
            alpha[l][j] = alpha[l][j] + 2. * Rt_hm *         A_tmth[j][k][l]*XSec[j] ;
            alpha[l][j] = alpha[l][j] +      Rt_em * Rt_em * A_tete[j][k][l]*XSec[j] ;
            alpha[l][j] = alpha[l][j] +      Rt_hm * Rt_hm * A_thth[j][k][l]*XSec[j] ;

           }  // end of k channel
        }  // end of j prodiction type
//
// Calculate X and R for mm-channel
//
       X_data[l] = X_data_top[l] / X_data_bot[l];

       gamma[l][0] = gamma[l][0]/ A_mm_save[0]; 
       gamma[l][1] = gamma[l][1]/ A_mm_save[1]; 
       beta[l][0]  = beta[l][0] /( 2. * A_mtm_save[0] ) ;
       beta[l][1]  = beta[l][1] /( 2. * A_mtm_save[1] ) ;
       alpha[l][0] = alpha[l][0]/( A_tmtm_save[0] ) ;
       alpha[l][1] = alpha[l][1]/( A_tmtm_save[1] ) ;

       C =     A_mm_save[0] * (gamma[l][0] * X_data[l] - Eff_mm[l][0]) - BKG_C[l][0]*Eff_C[l][0];
       C = C + A_mm_save[1] * (gamma[l][1] * X_data[l] - Eff_mm[l][1]) - BKG_C[l][1]*Eff_C[l][1];
       B =     2.* A_mtm_save[0] * (beta[l][0] * X_data[l] - Eff_mtm[l][0]) - BKG_B[l][0]*Eff_B[l][0];      
       B = B + 2.* A_mtm_save[1] * (beta[l][1] * X_data[l] - Eff_mtm[l][1]) - BKG_B[l][1]*Eff_B[l][1];      
       A =     A_tmtm_save[0] * (alpha[l][0] * X_data[l] - Eff_tmtm[l][0]) - BKG_A[l][0]*Eff_A[l][0];   
       A = A + A_tmtm_save[1] * (alpha[l][1] * X_data[l] - Eff_tmtm[l][1]) - BKG_A[l][1]*Eff_A[l][1];   

       R_mp = (-B + sqrt(pow(B,2) - 4.0 * A * C))/(2.0 * A);
       R_mn = (-B - sqrt(pow(B,2) - 4.0 * A * C))/(2.0 * A);

       if(l==0){ 
          cout << " \n  " << endl;
          cout << " ****** mm  1b  Channel ***** " << endl;
          cout << " Yield top  = " << X_data_top[l] << endl;
          cout << " Yield bottom  = " << X_data_bot[l] << endl;
          cout << " X_data = " << X_data[l] << " +/- " << sqrt((X_data[l]*(1.-X_data[l]))/(X_data_bot[l])) << endl;
          //cout << " R_mp = " << R_mp  << " versus " <<  R_tm <<  endl;
          cout << " R_mn = " << R_mn  << " versus " <<  R_tm <<  endl;
       }
       if(l==1){
          cout << " \n  " << endl;
          cout << " ****** mm  2b  Channel ***** " << endl;
          cout << " Yield top  = " << X_data_top[l] << endl;
          cout << " Yield bottom  = " << X_data_bot[l] << endl;
          cout << " X_data = " << X_data[l] << " +/- " << sqrt((X_data[l]*(1.-X_data[l]))/(X_data_bot[l])) << endl;
          // cout << " R_mp = " << R_mp  << " versus " <<  R_tm <<  endl;
          cout << " R_mn = " << R_mn  << " versus " <<  R_tm <<  endl;
       }
          cout << " \n  " << endl;
    }  // end of l

//
// rewrite to be a function of R = (W-to-tau to tau-to-had)/W-to-mu 
//
for (j=0; j<2; j++){   // loop over production type
    for (k=0; k<4; k++){   // loop over channel type
       for (l=0; l<2; l++){   // loop #b category Yield_[i,j,k,l]
         Yield[0][j][k][l]  =      BFWm * BFWm * Rw_em * Rw_em * A_ee[j][k][l] ;
         Yield[1][j][k][l]  =      BFWm * BFWm *                 A_mm[j][k][l] ;
         Yield[2][j][k][l]  =      BFWm * BFWm * Rw_hm * Rw_hm * A_hh[j][k][l] ;
         Yield[3][j][k][l]  = 2. * BFWm * BFWm * Rw_em *         A_me[j][k][l] ;
         Yield[4][j][k][l]  = 2. * BFWm * BFWm * Rw_em * Rw_hm * A_eh[j][k][l] ;
         Yield[5][j][k][l]  = 2. * BFWm * BFWm * Rw_hm *         A_mh[j][k][l] ;

         Yield[6][j][k][l]  = 2. * BFWm * BFWm *         Rt_eh * R_th * A_mte[j][k][l] ;
         Yield[7][j][k][l]  = 2. * BFWm * BFWm *         Rt_mh * R_th * A_mtm[j][k][l] ;
         Yield[8][j][k][l]  = 2. * BFWm * BFWm *                 R_th * A_mth[j][k][l] ;
         Yield[9][j][k][l]  = 2. * BFWm * BFWm * Rw_em * Rt_eh * R_th * A_ete[j][k][l] ;
         Yield[10][j][k][l] = 2. * BFWm * BFWm * Rw_em * Rt_mh * R_th * A_etm[j][k][l] ;
         Yield[11][j][k][l] = 2. * BFWm * BFWm * Rw_em *         R_th * A_eth[j][k][l] ;
         Yield[12][j][k][l] = 2. * BFWm * BFWm * Rw_hm * Rt_eh * R_th * A_hte[j][k][l] ;
         Yield[13][j][k][l] = 2. * BFWm * BFWm * Rw_hm * Rt_mh * R_th * A_htm[j][k][l] ;
         Yield[14][j][k][l] = 2. * BFWm * BFWm * Rw_hm *         R_th * A_hth[j][k][l] ;

         Yield[15][j][k][l] = 2. * BFWm * BFWm * Rt_eh * Rt_mh * R_th * R_th * A_tmte[j][k][l] ;
         Yield[16][j][k][l] = 2. * BFWm * BFWm * Rt_eh *         R_th * R_th * A_teth[j][k][l] ;
         Yield[17][j][k][l] = 2. * BFWm * BFWm * Rt_mh *         R_th * R_th * A_tmth[j][k][l] ;
         Yield[18][j][k][l] =      BFWm * BFWm * Rt_eh * Rt_eh * R_th * R_th * A_tete[j][k][l] ;
         Yield[19][j][k][l] =      BFWm * BFWm * Rt_mh * Rt_mh * R_th * R_th * A_tmtm[j][k][l] ;
         Yield[20][j][k][l] =      BFWm * BFWm *                 R_th * R_th * A_thth[j][k][l] ;
      }
   }
}
// cout << "Yield with R_had factorization "  << endl;
 for (j=0; j<2; j++){   // loop over production type
    for (k= 0; k<4; k++){   // loop over channel type
       for (l= 0; l<2; l++){   // loop #b category Yield_[i,j,k,l]  
         Sum_Yield[j][k][l] = 0.0;
         for (i= 0; i<21; i++){   // loop over the 21 channels
            Sum_Yield[j][k][l] = Sum_Yield[j][k][l] + Yield[i][j][k][l];
         }
//       if (l==0 && k==0 && j==0) cout << "Yield for ttbar [mm, 1b] = " << Sum_Yield[j][k][l]*crossLumi[0]<< endl;
//       if (l==1 && k==0 && j==0) cout << "Yield for ttbar [mm, 2b] = " << Sum_Yield[j][k][l]*crossLumi[0]<< endl;
//       if (l==0 && k==0 && j==1) cout << "Yield for tW [mm, 1b] = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
//       if (l==1 && k==0 && j==1) cout << "Yield for tW [mm, 2b] = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
//       if (l==0 && k==1 && j==0) cout << "Yield for ttbar [em, 1b] = " << Sum_Yield[j][k][l]*crossLumi[0]<< endl;
//       if (l==1 && k==1 && j==0) cout << "Yield for ttbar [em, 2b] = " << Sum_Yield[j][k][l]*crossLumi[0]<< endl;
//       if (l==0 && k==1 && j==1) cout << "Yield for tW [em, 1b] = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
//       if (l==1 && k==1 && j==1) cout << "Yield for tW [em, 2b] = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
//       if (l==0 && k==2 && j==0) cout << "Yield for ttbar [mth, 1b] = " << Sum_Yield[j][k][l]*crossLumi[0]<<endl;
//       if (l==1 && k==2 && j==0) cout << "Yield for ttbar [mth, 2b] = " << Sum_Yield[j][k][l]*crossLumi[0]<<endl;
//       if (l==0 && k==2 && j==1) cout << "Yield for tW [mth, 1b] = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
//       if (l==1 && k==2 && j==1) cout << "Yield for tW [mth, 2b] = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
//       if (l==0 && k==3 && j==0) cout << "Yield for ttbar [m4j, 1b] = " << Sum_Yield[j][k][l]*crossLumi[0]<<endl;
//       if (l==1 && k==3 && j==0) cout << "Yield for ttbar [m4j, 2b] = " << Sum_Yield[j][k][l]*crossLumi[0]<<endl;
//       if (l==0 && k==3 && j==1) cout << "Yield for tW [m4j, 1b]  = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
//       if (l==1 && k==3 && j==1) cout << "Yield for tW [m4j, 2b]  = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
      }   
   }
 } 
//
//  Test for extracting  R_hadron in mu + t-had channel
//
    for (l= 0; l<2; l++){   // loop #b category Yield_[i,j,k,l]  

       X_data_top[l] = 0.0;
       X_data_bot[l] = 0.0;
       X_data[l]     = 0.0;

       for (j=0; j<2; j++){   // loop over production type

         alpha[l][j] = 0.0;    
         gamma[l][j] = 0.0;    
         beta[l][j]  = 0.0;     

         BKG_A[l][j] = 0.0;    
         BKG_B[l][j] = 0.0;    
         BKG_C[l][j] = 0.0;     

         XSec[j] = crossLumi[j]  ;

         for (k= 0; k<4; k++){   // loop over channel type

             X_data_bot[l] = X_data_bot[l] + Sum_Yield[j][k][l]*crossLumi[j] ;

             if(k==2){

             X_data_top[l] = X_data_top[l] + Sum_Yield[j][k][l]*crossLumi[j]  ;

             BKG_C[l][j] = BKG_C[l][j] +                      A_mm[j][k][l]*XSec[j] ;
             BKG_C[l][j] = BKG_C[l][j] +      Rw_em * Rw_em * A_ee[j][k][l]*XSec[j] ;
             BKG_C[l][j] = BKG_C[l][j] +      Rw_hm * Rw_hm * A_hh[j][k][l]*XSec[j] ;
             BKG_C[l][j] = BKG_C[l][j] + 2. * Rw_em *         A_me[j][k][l]*XSec[j] ;
             BKG_C[l][j] = BKG_C[l][j] + 2. * Rw_em * Rw_hm * A_eh[j][k][l]*XSec[j] ;

             BKG_B[l][j] = BKG_B[l][j] + 2. *         Rt_eh * A_mte[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. *         Rt_mh * A_mtm[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. * Rw_em * Rt_eh * A_ete[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. * Rw_em * Rt_mh * A_etm[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. * Rw_em *         A_eth[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. * Rw_hm * Rt_eh * A_hte[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. * Rw_hm * Rt_mh * A_htm[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. * Rw_hm *         A_hth[j][k][l]*XSec[j] ;

             BKG_A[l][j] = BKG_A[l][j] + 2. * Rt_mh * Rt_eh * A_tmte[j][k][l]*XSec[j] ;
             BKG_A[l][j] = BKG_A[l][j] + 2. * Rt_eh *         A_teth[j][k][l]*XSec[j] ;
             BKG_A[l][j] = BKG_A[l][j] +      Rt_mh * Rt_mh * A_tmtm[j][k][l]*XSec[j] ;
             BKG_A[l][j] = BKG_A[l][j] +      Rt_eh * Rt_eh * A_tete[j][k][l]*XSec[j] ;
             BKG_A[l][j] = BKG_A[l][j] +                      A_thth[j][k][l]*XSec[j] ;
            
             A_mh_save[j]   = A_mh[j][k][l]*XSec[j]; 
             A_mth_save[j]  = A_mth[j][k][l]*XSec[j];  
             A_tmth_save[j] = A_tmth[j][k][l]*XSec[j];      
             }
//
// mm gamma terms
// 
            gamma[l][j] = gamma[l][j] +                      A_mm[j][k][l]*XSec[j] ;
            gamma[l][j] = gamma[l][j] +      Rw_em * Rw_em * A_ee[j][k][l]*XSec[j] ;
            gamma[l][j] = gamma[l][j] +      Rw_hm * Rw_hm * A_hh[j][k][l]*XSec[j] ;
            gamma[l][j] = gamma[l][j] + 2. *         Rw_em * A_me[j][k][l]*XSec[j] ;
            gamma[l][j] = gamma[l][j] + 2. * Rw_em * Rw_hm * A_eh[j][k][l]*XSec[j] ;
            gamma[l][j] = gamma[l][j] + 2. *         Rw_hm * A_mh[j][k][l]*XSec[j] ;
//
// mtm terms
//
            beta[l][j] = beta[l][j] + 2. *         Rt_mh * A_mtm[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. *         Rt_eh * A_mte[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. *                 A_mth[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rw_em * Rt_eh * A_ete[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rw_em * Rt_mh * A_etm[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rw_em *         A_eth[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rw_hm * Rt_eh * A_hte[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rw_hm * Rt_mh * A_htm[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rw_hm *         A_hth[j][k][l]*XSec[j] ;
//
// tmtm terms
//
            alpha[l][j] = alpha[l][j] +      Rt_mh * Rt_mh * A_tmtm[j][k][l]*XSec[j] ;
            alpha[l][j] = alpha[l][j] + 2. * Rt_mh * Rt_eh * A_tmte[j][k][l]*XSec[j] ;
            alpha[l][j] = alpha[l][j] + 2. * Rt_eh *         A_teth[j][k][l]*XSec[j] ;
            alpha[l][j] = alpha[l][j] + 2. * Rt_mh *         A_tmth[j][k][l]*XSec[j] ;
            alpha[l][j] = alpha[l][j] +      Rt_eh * Rt_eh * A_tete[j][k][l]*XSec[j] ;
            alpha[l][j] = alpha[l][j] +                      A_thth[j][k][l]*XSec[j] ;

           }  // end of k channel
        }  // end of j prodiction type
//
// Calculate X and R for mm-channel
//
       X_data[l] = X_data_top[l] / X_data_bot[l];

       gamma[l][0] = gamma[l][0]/ (2. * Rw_hm * A_mh_save[0]) ; 
       gamma[l][1] = gamma[l][1]/ (2. * Rw_hm * A_mh_save[1]) ; 
       beta[l][0]  = beta[l][0] / (2. * A_mth_save[0]) ;
       beta[l][1]  = beta[l][1] / (2. * A_mth_save[1]) ;
       alpha[l][0] = alpha[l][0]/ (2. * Rt_mh * A_tmth_save[0]) ;
       alpha[l][1] = alpha[l][1]/ (2. * Rt_mh * A_tmth_save[1]) ;

       C =     2.* Rw_hm * A_mh_save[0] * (gamma[l][0] * X_data[l] - Eff_mh[l][0]) - BKG_C[l][0]*Eff_Ch[l][0];
       C = C + 2.* Rw_hm * A_mh_save[1] * (gamma[l][1] * X_data[l] - Eff_mh[l][1]) - BKG_C[l][1]*Eff_Ch[l][1];
       B =     2.* A_mth_save[0] * (beta[l][0] * X_data[l] - Eff_mth[l][0]) - BKG_B[l][0]*Eff_Bh[l][0];      
       B = B + 2.* A_mth_save[1] * (beta[l][1] * X_data[l] - Eff_mth[l][1]) - BKG_B[l][1]*Eff_Bh[l][1];      
       A =     2.* Rt_mh * A_tmth_save[0] * (alpha[l][0] * X_data[l] - Eff_tmth[l][0]) - BKG_A[l][0]*Eff_Ah[l][0]; 
       A = A + 2.* Rt_mh * A_tmth_save[1] * (alpha[l][1] * X_data[l] - Eff_tmth[l][1]) - BKG_A[l][1]*Eff_Ah[l][1];   

       R_mp = (-B + sqrt(pow(B,2) - 4.0 * A * C))/(2.0 * A);
       R_mn = (-B - sqrt(pow(B,2) - 4.0 * A * C))/(2.0 * A);

       if(l==0){ 
          cout << " \n  " << endl;
          cout << " ****** mh  1b  Channel ***** " << endl;
          cout << " Yield top  = " << X_data_top[l] << endl;
          cout << " Yield bottom  = " << X_data_bot[l] << endl;
          cout << " X_data = " << X_data[l] << " +/- " << sqrt((X_data[l]*(1. - X_data[l]))/(X_data_bot[l])) << endl;
 //         cout << " R_mp = " << R_mp  << " versus " <<  R_th <<  endl;
          cout << " R_mn = " << R_mn  << " versus " <<  R_th <<  endl;
       }
       if(l==1){
          cout << " \n  " << endl;
          cout << " ****** mh  2b  Channel ***** " << endl;
          cout << " Yield top  = " << X_data_top[l] << endl;
          cout << " Yield bottom  = " << X_data_bot[l] << endl;
          cout << " X_data = " << X_data[l] << " +/- " << sqrt((X_data[l]*(1. - X_data[l]))/(X_data_bot[l])) << endl;
//          cout << " R_mp = " << R_mp  << " versus " <<  R_th <<  endl;
          cout << " R_mn = " << R_mn  << " versus " <<  R_th <<  endl;
       }
          cout << " \n  " << endl;
    }  // end of l

//
// rewrite to be a function of R = (W-to-tau to tau-to-e)/W-to-mu 
//
for (j=0; j<2; j++){   // loop over production type
    for (k=0; k<4; k++){   // loop over channel type
       for (l=0; l<2; l++){   // loop #b category Yield_[i,j,k,l]
         Yield[0][j][k][l]  =      BFWm * BFWm * Rw_em * Rw_em * A_ee[j][k][l] ;
         Yield[1][j][k][l]  =      BFWm * BFWm *                 A_mm[j][k][l] ;
         Yield[2][j][k][l]  =      BFWm * BFWm * Rw_hm * Rw_hm * A_hh[j][k][l] ;
         Yield[3][j][k][l]  = 2. * BFWm * BFWm * Rw_em *         A_me[j][k][l] ;
         Yield[4][j][k][l]  = 2. * BFWm * BFWm * Rw_em * Rw_hm * A_eh[j][k][l] ;
         Yield[5][j][k][l]  = 2. * BFWm * BFWm * Rw_hm *         A_mh[j][k][l] ;

         Yield[6][j][k][l]  = 2. * BFWm * BFWm *                 R_te * A_mte[j][k][l] ;
         Yield[7][j][k][l]  = 2. * BFWm * BFWm *         Rt_me * R_te * A_mtm[j][k][l] ;
         Yield[8][j][k][l]  = 2. * BFWm * BFWm *         Rt_he * R_te * A_mth[j][k][l] ;
         Yield[9][j][k][l]  = 2. * BFWm * BFWm * Rw_em *         R_te * A_ete[j][k][l] ;
         Yield[10][j][k][l] = 2. * BFWm * BFWm * Rw_em * Rt_me * R_te * A_etm[j][k][l] ;
         Yield[11][j][k][l] = 2. * BFWm * BFWm * Rw_em * Rt_he * R_te * A_eth[j][k][l] ;
         Yield[12][j][k][l] = 2. * BFWm * BFWm * Rw_hm *         R_te * A_hte[j][k][l] ;
         Yield[13][j][k][l] = 2. * BFWm * BFWm * Rw_hm * Rt_me * R_te * A_htm[j][k][l] ;
         Yield[14][j][k][l] = 2. * BFWm * BFWm * Rw_hm * Rt_he * R_te * A_hth[j][k][l] ;

         Yield[15][j][k][l] = 2. * BFWm * BFWm * Rt_me *         R_te * R_te * A_tmte[j][k][l] ;
         Yield[16][j][k][l] = 2. * BFWm * BFWm *         Rt_he * R_te * R_te * A_teth[j][k][l] ;
         Yield[17][j][k][l] = 2. * BFWm * BFWm * Rt_me * Rt_he * R_te * R_te * A_tmth[j][k][l] ;
         Yield[18][j][k][l] =      BFWm * BFWm *                 R_te * R_te * A_tete[j][k][l] ;
         Yield[19][j][k][l] =      BFWm * BFWm * Rt_me * Rt_me * R_te * R_te * A_tmtm[j][k][l] ;
         Yield[20][j][k][l] =      BFWm * BFWm * Rt_he * Rt_he * R_te * R_te * A_thth[j][k][l] ;
      }
   }
}
//  cout << "Yield with R_em  factorization "  << endl;
 for (j=0; j<2; j++){   // loop over production type
    for (k= 0; k<4; k++){   // loop over channel type
       for (l= 0; l<2; l++){   // loop #b category Yield_[i,j,k,l]  
         Sum_Yield[j][k][l] = 0.0;
         for (i= 0; i<21; i++){   // loop over the 21 channels
            Sum_Yield[j][k][l] = Sum_Yield[j][k][l] + Yield[i][j][k][l];
         }
//       if (l==0 && k==0 && j==0) cout << "Yield for ttbar [mm, 1b] = " << Sum_Yield[j][k][l]*crossLumi[0]<< endl;
//       if (l==1 && k==0 && j==0) cout << "Yield for ttbar [mm, 2b] = " << Sum_Yield[j][k][l]*crossLumi[0]<< endl;
//       if (l==0 && k==0 && j==1) cout << "Yield for tW [mm, 1b] = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
//       if (l==1 && k==0 && j==1) cout << "Yield for tW [mm, 2b] = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
//       if (l==0 && k==1 && j==0) cout << "Yield for ttbar [em, 1b] = " << Sum_Yield[j][k][l]*crossLumi[0]<< endl;
//       if (l==1 && k==1 && j==0) cout << "Yield for ttbar [em, 2b] = " << Sum_Yield[j][k][l]*crossLumi[0]<< endl;
//       if (l==0 && k==1 && j==1) cout << "Yield for tW [em, 1b] = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
//       if (l==1 && k==1 && j==1) cout << "Yield for tW [em, 2b] = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
//       if (l==0 && k==2 && j==0) cout << "Yield for ttbar [mth, 1b] = " << Sum_Yield[j][k][l]*crossLumi[0]<<endl;
//       if (l==1 && k==2 && j==0) cout << "Yield for ttbar [mth, 2b] = " << Sum_Yield[j][k][l]*crossLumi[0]<<endl;
//       if (l==0 && k==2 && j==1) cout << "Yield for tW [mth, 1b] = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
//       if (l==1 && k==2 && j==1) cout << "Yield for tW [mth, 2b] = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
//       if (l==0 && k==3 && j==0) cout << "Yield for ttbar [m4j, 1b] = " << Sum_Yield[j][k][l]*crossLumi[0]<<endl;
//       if (l==1 && k==3 && j==0) cout << "Yield for ttbar [m4j, 2b] = " << Sum_Yield[j][k][l]*crossLumi[0]<<endl;
//       if (l==0 && k==3 && j==1) cout << "Yield for tW [m4j, 1b]  = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
//       if (l==1 && k==3 && j==1) cout << "Yield for tW [m4j, 2b]  = " << Sum_Yield[j][k][l]*crossLumi[1] << endl;
      }   
   }
 } 
//
//  Test for extracting  R_hadron in mu + e channel
//
    for (l= 0; l<2; l++){   // loop #b category Yield_[i,j,k,l]  

       X_data_top[l] = 0.0;
       X_data_bot[l] = 0.0;
       X_data[l]     = 0.0;

       for (j=0; j<2; j++){   // loop over production type

         alpha[l][j] = 0.0;    
         gamma[l][j] = 0.0;    
         beta[l][j]  = 0.0;     

         BKG_A[l][j] = 0.0;    
         BKG_B[l][j] = 0.0;    
         BKG_C[l][j] = 0.0;     

         XSec[j] = crossLumi[j]  ;

         for (k= 0; k<4; k++){   // loop over channel type

             X_data_bot[l] = X_data_bot[l] + Sum_Yield[j][k][l]*crossLumi[j] ;

             if(k==1){

             X_data_top[l] = X_data_top[l] + Sum_Yield[j][k][l]*crossLumi[j]  ;

             BKG_C[l][j] = BKG_C[l][j] +                      A_mm[j][k][l]*XSec[j] ;
             BKG_C[l][j] = BKG_C[l][j] +      Rw_em * Rw_em * A_ee[j][k][l]*XSec[j] ;
             BKG_C[l][j] = BKG_C[l][j] +      Rw_hm * Rw_hm * A_hh[j][k][l]*XSec[j] ;
             BKG_C[l][j] = BKG_C[l][j] + 2. * Rw_em * Rw_hm * A_eh[j][k][l]*XSec[j] ;
             BKG_C[l][j] = BKG_C[l][j] + 2. *         Rw_hm * A_mh[j][k][l]*XSec[j] ;

             BKG_B[l][j] = BKG_B[l][j] + 2. *         Rt_me * A_mtm[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. *         Rt_he * A_mth[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. * Rw_em *         A_ete[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. * Rw_em * Rt_he * A_eth[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. * Rw_hm *         A_hte[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. * Rw_hm * Rt_me * A_htm[j][k][l]*XSec[j] ;
             BKG_B[l][j] = BKG_B[l][j] + 2. * Rw_hm * Rt_he * A_hth[j][k][l]*XSec[j] ;

             BKG_A[l][j] = BKG_A[l][j] + 2. *         Rt_he * A_teth[j][k][l]*XSec[j] ;
             BKG_A[l][j] = BKG_A[l][j] +      Rt_me * Rt_me * A_tmtm[j][k][l]*XSec[j] ;
             BKG_A[l][j] = BKG_A[l][j] +                      A_tete[j][k][l]*XSec[j] ;
             BKG_A[l][j] = BKG_A[l][j] +      Rt_he * Rt_he * A_thth[j][k][l]*XSec[j] ;
             BKG_A[l][j] = BKG_A[l][j] + 2. * Rt_me * Rt_he * A_tmth[j][k][l]*XSec[j] ;

             A_me_save[j]   = A_me[j][k][l]*XSec[j]; 
             A_mte_save[j]  = A_mte[j][k][l]*XSec[j];  
             A_etm_save[j]  = A_etm[j][k][l]*XSec[j];  
             A_tmte_save[j] = A_tmte[j][k][l]*XSec[j];      
             }
//
// mm gamma terms
// 
            gamma[l][j] = gamma[l][j] +                      A_mm[j][k][l]*XSec[j] ;
            gamma[l][j] = gamma[l][j] +      Rw_em * Rw_em * A_ee[j][k][l]*XSec[j] ;
            gamma[l][j] = gamma[l][j] +      Rw_hm * Rw_hm * A_hh[j][k][l]*XSec[j] ;
            gamma[l][j] = gamma[l][j] + 2. *         Rw_em * A_me[j][k][l]*XSec[j] ;
            gamma[l][j] = gamma[l][j] + 2. * Rw_em * Rw_hm * A_eh[j][k][l]*XSec[j] ;
            gamma[l][j] = gamma[l][j] + 2. *         Rw_hm * A_mh[j][k][l]*XSec[j] ;
//
// mtm terms
//
            beta[l][j] = beta[l][j] + 2. *         Rt_me * A_mtm[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. *                 A_mte[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. *         Rt_he * A_mth[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rw_em *         A_ete[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rw_em * Rt_me * A_etm[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rw_em * Rt_he * A_eth[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rw_hm *         A_hte[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rw_hm * Rt_me * A_htm[j][k][l]*XSec[j] ;
            beta[l][j] = beta[l][j] + 2. * Rw_hm * Rt_he * A_hth[j][k][l]*XSec[j] ;
//
// tmtm terms
//
            alpha[l][j] = alpha[l][j] +      Rt_me * Rt_me * A_tmtm[j][k][l]*XSec[j] ;
            alpha[l][j] = alpha[l][j] + 2. * Rt_me *         A_tmte[j][k][l]*XSec[j] ;
            alpha[l][j] = alpha[l][j] + 2. *         Rt_he * A_teth[j][k][l]*XSec[j] ;
            alpha[l][j] = alpha[l][j] + 2. * Rt_me * Rt_he * A_tmth[j][k][l]*XSec[j] ;
            alpha[l][j] = alpha[l][j] +                      A_tete[j][k][l]*XSec[j] ;
            alpha[l][j] = alpha[l][j] +      Rt_he * Rt_he * A_thth[j][k][l]*XSec[j] ;

           }  // end of k channel
        }  // end of j prodiction type
//
// Calculate X and R for em-channel
//
       X_data[l] = X_data_top[l] / X_data_bot[l];

       gamma[l][0] = gamma[l][0]/ (2. * Rw_em * A_me_save[0]) ; 
       gamma[l][1] = gamma[l][1]/ (2. * Rw_em * A_me_save[1]) ; 
       beta[l][0]  = beta[l][0] / (2. * A_mte_save[0] + 2. * Rw_em * Rt_me * A_etm_save[0]) ;
       beta[l][1]  = beta[l][1] / (2. * A_mte_save[1] + 2. * Rw_em * Rt_me * A_etm_save[1]) ;
       alpha[l][0] = alpha[l][0]/ (2. * Rt_me * A_tmte_save[0]) ;
       alpha[l][1] = alpha[l][1]/ (2. * Rt_me * A_tmte_save[1]) ;

       C =     2.* Rw_em * A_me_save[0] * (gamma[l][0] * X_data[l] - Eff_me[l][0]) - BKG_C[l][0]*Eff_Ce[l][0];
       C = C + 2.* Rw_em * A_me_save[1] * (gamma[l][1] * X_data[l] - Eff_me[l][1]) - BKG_C[l][1]*Eff_Ce[l][1];
       B =    (2. * A_mte_save[0] + 2. * Rw_em * Rt_me * A_etm_save[0])*(beta[l][0] * X_data[l] - Eff_mte_etm[l][0]) ;      
       B = B +(2. * A_mte_save[1] + 2. * Rw_em * Rt_me * A_etm_save[1])*(beta[l][1] * X_data[l] - Eff_mte_etm[l][1]) ;      
       B = B - BKG_B[l][0]*Eff_Be[l][0] - BKG_B[l][1]*Eff_Be[l][1];      
       A =     2.* Rt_me * A_tmte_save[0] * (alpha[l][0] * X_data[l] - Eff_tmte[l][0]) - BKG_A[l][0]*Eff_Ae[l][0]; 
       A = A + 2.* Rt_me * A_tmte_save[1] * (alpha[l][1] * X_data[l] - Eff_tmte[l][1]) - BKG_A[l][1]*Eff_Ae[l][1];   

       R_mp = (-B + sqrt(pow(B,2) - 4.0 * A * C))/(2.0 * A);
       R_mn = (-B - sqrt(pow(B,2) - 4.0 * A * C))/(2.0 * A);

       if(l==0){ 
          cout << " \n  " << endl;
          cout << " ****** me  1b  Channel ***** " << endl;
          cout << " Yield top  = " << X_data_top[l] << endl;
          cout << " Yield bottom  = " << X_data_bot[l] << endl;
          cout << " X_data = " << X_data[l] << " +/- " << sqrt((X_data[l]*(1. - X_data[l]))/(X_data_bot[l])) << endl;
 //         cout << " R_mp = " << R_mp  << " versus " <<  R_te <<  endl;
          cout << " R_mn = " << R_mn  << " versus " <<  R_te <<  endl;
       }
       if(l==1){
          cout << " \n  " << endl;
          cout << " ****** me  2b  Channel ***** " << endl;
          cout << " Yield top  = " << X_data_top[l] << endl;
          cout << " Yield bottom  = " << X_data_bot[l] << endl;
          cout << " X_data = " << X_data[l] << " +/- " << sqrt((X_data[l]*(1. - X_data[l]))/(X_data_bot[l])) << endl;
//          cout << " R_mp = " << R_mp  << " versus " <<  R_te <<  endl;
          cout << " R_mn = " << R_mn  << " versus " <<  R_te <<  endl;
       }
          cout << " \n  " << endl;
    }  // end of l

//
//END
} 
