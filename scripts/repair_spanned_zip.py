#!/usr/bin/env python3
"""Rewrite a concatenated split ZIP64 archive into a valid single-disk archive, in place.

`zip -s` stores each member's offset RELATIVE TO ITS OWN DISK, with the disk recorded
separately. Concatenating the parts therefore yields correct file data at wrong offsets,
and Python's zipfile refuses the archive outright. `zip -FF` fixes this; `zip` is not
installed here and apt is unavailable.

Measured on the Emilia podcast archive (213,164 members): every member's data sits at
either `stored - DISK` or `stored` in the concatenated file, splitting 178/122 over a
300-member sample — i.e. exactly the two disks. So:

    absolute_offset = stored_offset - DISK + disk_number * DISK

where DISK is the split size (== the size of part 1). Offsets above 0xFFFFFFFF live in the
ZIP64 extra field rather than the base record, so both storage locations are handled.

This rewrites offsets and zeroes disk numbers everywhere: central-directory entries, their
ZIP64 extras, the ZIP64 EOCD, its locator, and the 32-bit EOCD. File data never moves.

  python scripts/repair_spanned_zip.py archive.zip --disk-size 10485760000
"""
import argparse, struct, sys, zipfile
from pathlib import Path

EOCD, Z64_EOCD, Z64_LOC, CEN = b"PK\x05\x06", b"PK\x06\x06", b"PK\x06\x07", b"PK\x01\x02"
U32, U16 = 0xFFFFFFFF, 0xFFFF


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("archive", type=Path)
    ap.add_argument("--disk-size", type=int, required=True, help="split size == size of part 1")
    ap.add_argument("--verify", type=int, default=300)
    args = ap.parse_args()
    DISK = args.disk_size
    p = args.archive

    with p.open("r+b") as f:
        size = p.stat().st_size
        tail_len = min(size, 1 << 20)
        f.seek(size - tail_len)
        tail = f.read(tail_len)
        base = size - tail_len

        i, j, k = tail.rfind(EOCD), tail.rfind(Z64_LOC), tail.rfind(Z64_EOCD)
        if min(i, j, k) < 0:
            sys.exit("missing EOCD / ZIP64 EOCD / locator")
        ents_total, cd_size, cd_off = struct.unpack("<QQQ", tail[k + 32:k + 56])
        print(f"entries={ents_total} cd_size={cd_size} cd_off={cd_off} disk_size={DISK}")

        # cd_off is disk-relative as well, so don't trust it. The central directory always
        # ends immediately before the ZIP64 EOCD record, which gives its true position.
        cd_abs = base + k - cd_size
        f.seek(cd_abs)
        if f.read(4) != CEN:
            sys.exit(f"central directory not at {cd_abs} (derived from ZIP64 EOCD - cd_size)")
        print(f"central directory at {cd_abs} (declared {cd_off}, delta {cd_abs - cd_off})")

        f.seek(cd_abs)
        cd = bytearray(f.read(cd_size))
        off, fixed, n, unresolved = 0, 0, 0, 0
        while off + 46 <= len(cd) and bytes(cd[off:off + 4]) == CEN:
            n += 1
            nlen, elen, clen = struct.unpack("<HHH", cd[off + 28:off + 34])
            disk = struct.unpack("<H", cd[off + 34:off + 36])[0]
            lho = struct.unpack("<I", cd[off + 42:off + 46])[0]
            usz, csz = struct.unpack("<II", cd[off + 20:off + 28])
            ex_at = off + 46 + nlen
            ex_end = ex_at + elen

            # Walk the extra fields looking for ZIP64 extended information (0x0001).
            z64_off_at = z64_disk_at = None
            q = ex_at
            while q + 4 <= ex_end:
                hid, hsz = struct.unpack("<HH", cd[q:q + 4])
                if hid == 0x0001:
                    r = q + 4
                    if usz == U32: r += 8
                    if csz == U32: r += 8
                    if lho == U32:
                        z64_off_at = r; r += 8
                    if disk == U16:
                        z64_disk_at = r
                    break
                q += 4 + hsz

            if z64_off_at is not None:
                stored = struct.unpack("<Q", cd[z64_off_at:z64_off_at + 8])[0]
            else:
                stored = lho

            # The recorded disk number proved unreliable on some entries, so resolve it
            # empirically instead: exactly one candidate lands on a local file header whose
            # name matches this entry. Self-validating, and we already have the bytes.
            name = bytes(cd[off + 46:off + 46 + nlen])
            new = None
            for cand in (stored - DISK, stored):
                if not (0 <= cand < size - 30):
                    continue
                f.seek(cand)
                hdr = f.read(30)
                if hdr[:4] != b"PK\x03\x04":
                    continue
                hn = struct.unpack("<H", hdr[26:28])[0]
                if hn != nlen:
                    continue
                if f.read(nlen) != name:
                    continue
                new = cand
                break
            if new is None:
                unresolved += 1
            else:
                if new != stored:
                    if z64_off_at is not None:
                        cd[z64_off_at:z64_off_at + 8] = struct.pack("<Q", new)
                    elif new <= U32:
                        cd[off + 42:off + 46] = struct.pack("<I", new)
                    else:
                        unresolved += 1
                        new = None
                if new is not None:
                    fixed += 1
            cd[off + 34:off + 36] = struct.pack("<H", 0)
            if z64_disk_at is not None:
                cd[z64_disk_at:z64_disk_at + 4] = struct.pack("<I", 0)
            off += 46 + nlen + elen + clen

        f.seek(cd_abs)
        f.write(bytes(cd))
        print(f"central directory: {n} walked, {fixed} resolved, {unresolved} unresolved")

        # EOCD records: single disk, all entries here, CD at its true position.
        f.seek(base + k + 16); f.write(struct.pack("<II", 0, 0))
        f.seek(base + k + 24); f.write(struct.pack("<QQ", ents_total, ents_total))
        f.seek(base + k + 48); f.write(struct.pack("<Q", cd_abs))
        f.seek(base + j + 4);  f.write(struct.pack("<I", 0))
        f.seek(base + j + 16); f.write(struct.pack("<I", 1))
        f.seek(base + i + 4);  f.write(struct.pack("<HH", 0, 0))

    zf = zipfile.ZipFile(p)
    infos = [x for x in zf.infolist() if not x.is_dir()]
    print(f"OK: opens with {len(infos)} files")
    import random
    random.seed(0)
    ok = bad = 0
    for x in random.sample(infos, min(args.verify, len(infos))):
        try:
            zf.read(x.filename); ok += 1
        except Exception:
            bad += 1
    print(f"read verification: {ok} ok / {bad} failed")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
