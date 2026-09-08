import pyarrow.parquet as pq, os, sys
# Row counts recorded by sweep_parquet_purity.py against the Modal originals.
EXPECT = {0:105497538,1:83385621,2:110149096,3:123625419,4:92807631,5:132084368,
6:105613047,7:199699002,8:56651521,9:83043988,10:94272826,11:76316279,12:38204991,
13:82891263,14:120403769,15:113533892,16:62108889,17:128070540,18:117759145,
19:61434041,20:40223244,21:78116005,22:85659822,23:108682105,24:34140916,
25:47316831,26:81622481,27:76214102,28:86887680,29:82558893,30:66556584,31:49004124}
d=os.environ["SCRATCH"]+"/cckf/expanded"; bad=[]; tot=0
for i in range(32):
    p=f"{d}/expanded_event{i:09d}.parquet"
    try:
        m=pq.ParquetFile(p).metadata; n=m.num_rows; tot+=n
        if n!=EXPECT[i]: bad.append(f"ev{i}: rows {n:,} != expected {EXPECT[i]:,}")
        elif m.num_columns!=76: bad.append(f"ev{i}: {m.num_columns} cols != 76")
    except Exception as e: bad.append(f"ev{i}: UNREADABLE {type(e).__name__}: {str(e)[:60]}")
print(f"total rows: {tot:,}  (expected 2,721,470,...)")
if bad:
    print(f"FAILURES: {len(bad)}"); [print("  "+b) for b in bad]; sys.exit(1)
print("ALL 32 VERIFIED: row counts and column counts match the Modal originals")
