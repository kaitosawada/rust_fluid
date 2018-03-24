#[macro_use(s)]
extern crate ndarray;
extern crate ndarray_parallel;
extern crate rayon;
extern crate rand;
extern crate ndarray_rand;


use ndarray_rand::RandomExt;
use rand::distributions::*;
use ndarray::prelude::*;
use ndarray_parallel::prelude::*;
use rayon::prelude::*;
use std::time::{Duration, Instant};
use ndarray::{
    RemoveAxis,
    Array1,
    Array2,
    Array3,
    arr1,
};
use std::ops::{Add, Sub, Mul, Div};

/*fn bench<F>(times: usize, f: F)
    where F: FnMut() {
    let start = Instant::now();
    for _i in 0..1000 {
        f();
    }
    let end = start.elapsed();
    println!("{}.{:03}", end.as_secs(), end.subsec_nanos() / 1_000_000);
}*/

fn roll_bench(a: Array2<f64>) {
    let start = Instant::now();
    for _i in 0..1000 {
        roll(&a, &[1, 1]);
    }
    let end = start.elapsed();
    println!("{}.{:03}", end.as_secs(), end.subsec_nanos() / 1_000_000);
}

fn main() {
    let size = (1000, 1000);
    let a1 = Array::random(size, Range::new(0.0, 1.0));
    let a2 = Array::random(size, Range::new(0.0, 1.0));
    let a3 = Array::random(size, Range::new(0.0, 1.0));
    let a4 = Array::random(size, Range::new(0.0, 1.0));
    let a5 = Array::random(size, Range::new(0.0, 1.0));
    let a6 = Array::random(size, Range::new(0.0, 1.0));

    let start = Instant::now();
    for _i in 0..1000 {
        let a = a1.view().add(&a2).mul(&a3);
    }
    let end = start.elapsed();
    println!("{}.{:03}", end.as_secs(), end.subsec_nanos() / 1_000_000);


    let start = Instant::now();
    for _i in 0..1000 {
        let mut a = Array::zeros(size);//self.rho_l.add(&self.rho_g);
        ndarray::Zip::from(&mut a)
            .and(&a4)
            .and(&a5)
            .and(&a6)
            .par_apply(|rho, &l, &g, &ar3| {
                *rho = (l + g) * ar3;
            });
    }
    let end = start.elapsed();
    println!("{}.{:03}", end.as_secs(), end.subsec_nanos() / 1_000_000);

    /*let mut a = Array2::<f64>::zeros((64, 64));
    let mut a0 = Array2::<f64>::zeros((64, 128));
    let mut a1 = Array2::<f64>::zeros((64, 256));
    let mut a2 = Array2::<f64>::zeros((64, 512));
    let mut a3 = Array2::<f64>::zeros((64, 1024));
    let mut a4 = Array2::<f64>::zeros((64, 2048));
    let mut a5 = Array2::<f64>::zeros((64, 4096));
    let mut a6 = Array2::<f64>::zeros((64, 8192));
    let mut a7 = Array2::<f64>::zeros((64, 16384));


    roll_bench(a);
    roll_bench(a0);
    roll_bench(a1);
    roll_bench(a2);
    roll_bench(a3);
    roll_bench(a4);
    roll_bench(a5);
    roll_bench(a6);
    roll_bench(a7);*/

// Parallel versions of regular array methods (ParMap trait)
    /*a.par_map_inplace(|x| *x = x.exp());
    a.par_mapv_inplace(f64::exp);

    // You can also use the parallel iterator directly
    a.par_iter_mut().for_each(|x| *x = x.exp());*/
}

fn roll(a: &Array2<f64>, dir: &[isize; 2]) -> Array2<f64> {
    let mut b = Array::zeros(a.dim());
    let x = dir[0];
    let y = dir[1];
    /*b.slice_mut(s![x.., y..]).assign(&a.slice(s![..- x, ..- y]));
    b.slice_mut(s![..x, y..]).assign(&a.slice(s![ - x.., ..- y]));
    b.slice_mut(s![x.., ..y]).assign(&a.slice(s![..- x, - y..]));
    b.slice_mut(s![..x, ..y]).assign(&a.slice(s![ - x.., - y..]));*/
    //b.assign(&a);
    b
}