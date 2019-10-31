import lmdb
env = lmdb.open('/mnt/data/epic_kitchen/sample2/obj/', readonly=True, lock=False)
with env.begin() as txn:
    cursor = txn.cursor()
    count = 0
    for key, value in cursor:
            print((key))

