//! Fast generator of sudoku puzzles with UNIQUE solutions.
//!
//! For each puzzle: build a random solved grid by randomized backtracking, then remove
//! cells in random order, keeping a removal only if the puzzle still has exactly one
//! solution, until the target hole count is reached.
//!
//! Train set: hole count ~ Uniform{min_holes..=max_holes}.  Val sets: exactly `n_val`
//! puzzles at each level in `val_levels`, from a disjoint RNG stream.
//! Output: NumPy .npy files (uint8, shape (N, 81), 0 = hole).
//!
//!   cargo run --release -- --n-train 200000 --min-holes 0 --max-holes 60 \
//!       --val-levels 10,20,30,40,50,55,60 --n-val 1024 --threads 8 --out data/unique
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

type Grid = [u8; 81];

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Rng(seed.wrapping_mul(0x9E3779B97F4A7C15) | 1) }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn below(&mut self, n: u64) -> u64 { (self.next() >> 11) % n }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() { let j = self.below(i as u64 + 1) as usize; v.swap(i, j); }
    }
}

#[inline] fn boxof(i: usize) -> usize { (i / 9 / 3) * 3 + (i % 9) / 3 }

struct State { rows: [u16; 9], cols: [u16; 9], boxes: [u16; 9] }
impl State {
    fn from(g: &Grid) -> Self {
        let mut s = State { rows: [0; 9], cols: [0; 9], boxes: [0; 9] };
        for i in 0..81 { if g[i] != 0 { s.set(i, g[i]); } }
        s
    }
    #[inline] fn set(&mut self, i: usize, d: u8) { let b = 1u16 << d; self.rows[i / 9] |= b; self.cols[i % 9] |= b; self.boxes[boxof(i)] |= b; }
    #[inline] fn clear(&mut self, i: usize, d: u8) { let b = !(1u16 << d); self.rows[i / 9] &= b; self.cols[i % 9] &= b; self.boxes[boxof(i)] &= b; }
    #[inline] fn used(&self, i: usize) -> u16 { self.rows[i / 9] | self.cols[i % 9] | self.boxes[boxof(i)] }
}

fn fill(g: &mut Grid, s: &mut State, i: usize, rng: &mut Rng) -> bool {
    if i == 81 { return true; }
    let used = s.used(i);
    let mut cands: Vec<u8> = (1..=9u8).filter(|d| used & (1 << d) == 0).collect();
    rng.shuffle(&mut cands);
    for d in cands {
        g[i] = d; s.set(i, d);
        if fill(g, s, i + 1, rng) { return true; }
        s.clear(i, d);
    }
    g[i] = 0;
    false
}

fn random_solved(rng: &mut Rng) -> Grid {
    let mut g = [0u8; 81];
    let mut s = State::from(&g);
    assert!(fill(&mut g, &mut s, 0, rng));
    g
}

/// Count completions of `g`, stopping at `limit`.  MRV branching on bitmasks.
fn count_rec(g: &mut Grid, s: &mut State, limit: u32, n: &mut u32) {
    if *n >= limit { return; }
    let mut best = usize::MAX; let mut best_c = 10u32; let mut best_mask = 0u16;
    for i in 0..81 {
        if g[i] != 0 { continue; }
        let mask = !s.used(i) & 0b1111111110;
        let c = mask.count_ones();
        if c < best_c { best_c = c; best = i; best_mask = mask; if c == 0 { return; } }
    }
    if best == usize::MAX { *n += 1; return; }
    let mut m = best_mask;
    while m != 0 {
        let d = m.trailing_zeros() as u8; m &= m - 1;
        g[best] = d; s.set(best, d);
        count_rec(g, s, limit, n);
        s.clear(best, d); g[best] = 0;
        if *n >= limit { return; }
    }
}
fn count_solutions(g: &Grid, limit: u32) -> u32 {
    let mut g2 = *g; let mut s = State::from(g); let mut n = 0;
    count_rec(&mut g2, &mut s, limit, &mut n); n
}

/// Greedy random removal keeping uniqueness. Returns (holed, holes achieved).
fn make_unique(solved: &Grid, target: usize, rng: &mut Rng) -> (Grid, usize) {
    let mut h = *solved;
    let mut order: Vec<usize> = (0..81).collect();
    rng.shuffle(&mut order);
    let mut holes = 0;
    for &i in &order {
        if holes >= target { break; }
        let v = h[i]; h[i] = 0;
        if count_solutions(&h, 2) == 1 { holes += 1; } else { h[i] = v; }
    }
    (h, holes)
}

fn make_exact(target: usize, rng: &mut Rng, max_tries: usize) -> Option<(Grid, Grid)> {
    for _ in 0..max_tries {
        let s = random_solved(rng);
        let (h, k) = make_unique(&s, target, rng);
        if k == target { return Some((h, s)); }
    }
    None
}

fn write_npy(path: &str, rows: &[Grid]) {
    let mut f = BufWriter::new(File::create(path).expect("create"));
    let header = format!("{{'descr': '|u1', 'fortran_order': False, 'shape': ({}, 81), }}", rows.len());
    let mut hdr = header.into_bytes();
    let total = 10 + hdr.len() + 1;
    let pad = (64 - total % 64) % 64;
    hdr.extend(std::iter::repeat(b' ').take(pad)); hdr.push(b'\n');
    f.write_all(b"\x93NUMPY\x01\x00").unwrap();
    f.write_all(&(hdr.len() as u16).to_le_bytes()).unwrap();
    f.write_all(&hdr).unwrap();
    for r in rows { f.write_all(r).unwrap(); }
}

fn arg<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> T {
    args.iter().position(|a| a == key).and_then(|p| args.get(p + 1)).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_train: usize = arg(&args, "--n-train", 200_000);
    let min_h: usize = arg(&args, "--min-holes", 0);
    let max_h: usize = arg(&args, "--max-holes", 60);
    let n_val: usize = arg(&args, "--n-val", 1024);
    let threads: usize = arg(&args, "--threads", 8);
    let seed: u64 = arg(&args, "--seed", 0);
    let out: String = arg(&args, "--out", "data/unique".to_string());
    let val_levels: Vec<usize> = arg(&args, "--val-levels", "10,20,30,40,50,55,60".to_string())
        .split(',').filter_map(|s| s.trim().parse().ok()).collect();

    // ---- train: target ~ U[min_h, max_h]; up to 5 fresh grids per target, keep the best.
    let t0 = Instant::now();
    let results = Arc::new(Mutex::new(Vec::<(Grid, Grid)>::with_capacity(n_train)));
    let short = Arc::new(Mutex::new(0usize));
    let per = (n_train + threads - 1) / threads;
    let handles: Vec<_> = (0..threads).map(|t| {
        let results = Arc::clone(&results); let short = Arc::clone(&short);
        thread::spawn(move || {
            let mut rng = Rng::new(seed * 1000 + t as u64 + 1);
            let mut local = Vec::with_capacity(per); let mut nshort = 0;
            for _ in 0..per {
                let target = min_h + rng.below((max_h - min_h + 1) as u64) as usize;
                let mut best: Option<(Grid, Grid, usize)> = None;
                for _ in 0..5 {
                    let s = random_solved(&mut rng);
                    let (h, k) = make_unique(&s, target, &mut rng);
                    if best.as_ref().map_or(true, |b| k > b.2) { best = Some((h, s, k)); }
                    if k == target { break; }
                }
                let (h, s, k) = best.unwrap();
                if k < target { nshort += 1; }
                local.push((h, s));
            }
            results.lock().unwrap().extend(local); *short.lock().unwrap() += nshort;
        })
    }).collect();
    for h in handles { h.join().unwrap(); }
    let train = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
    let train: Vec<_> = train.into_iter().take(n_train).collect();
    eprintln!("train: {} puzzles in {:.1}s ({} fell short of target)", train.len(), t0.elapsed().as_secs_f64(), *short.lock().unwrap());
    write_npy(&format!("{}_train_holed.npy", out), &train.iter().map(|x| x.0).collect::<Vec<_>>());
    write_npy(&format!("{}_train_solved.npy", out), &train.iter().map(|x| x.1).collect::<Vec<_>>());

    // ---- val: exact hole counts per level, separate RNG stream
    let t1 = Instant::now();
    let mut val: Vec<(Grid, Grid)> = Vec::new();
    for &lvl in &val_levels {
        let got = Arc::new(Mutex::new(Vec::<(Grid, Grid)>::new()));
        let per = (n_val + threads - 1) / threads;
        let hs: Vec<_> = (0..threads).map(|t| {
            let got = Arc::clone(&got);
            thread::spawn(move || {
                let mut rng = Rng::new(seed * 1000 + 500 + lvl as u64 * 64 + t as u64);
                let mut local = Vec::new();
                for _ in 0..per {
                    match make_exact(lvl, &mut rng, 3000) { Some(p) => local.push(p), None => eprintln!("warn: could not reach {} holes", lvl) }
                }
                got.lock().unwrap().extend(local);
            })
        }).collect();
        for h in hs { h.join().unwrap(); }
        let mut g = Arc::try_unwrap(got).unwrap().into_inner().unwrap(); g.truncate(n_val);
        eprintln!("val level {}: {} puzzles", lvl, g.len());
        val.extend(g);
    }
    eprintln!("val: {} puzzles in {:.1}s", val.len(), t1.elapsed().as_secs_f64());
    write_npy(&format!("{}_val_holed.npy", out), &val.iter().map(|x| x.0).collect::<Vec<_>>());
    write_npy(&format!("{}_val_solved.npy", out), &val.iter().map(|x| x.1).collect::<Vec<_>>());
}
