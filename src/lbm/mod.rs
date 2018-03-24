extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_parallel;
extern crate rayon;
extern crate itertools;
extern crate num_traits;

use itertools::Zip;
use itertools::Itertools;
use std::ops::{Add, Sub, Mul, Div};
use rand::distributions::*;
use rand::Rng;
use ndarray::prelude::*;
use ndarray::{
    RemoveAxis,
    Array1,
    Array2,
    Array3,
    arr1,
};
use ndarray_rand::RandomExt;
use ndarray_parallel::prelude::*;
use ndarray_parallel::*;
use std::ptr::copy_nonoverlapping;
use std::ptr::replace;
use num_traits::Zero;

pub mod lbm_state;
pub mod lbm_state2;
pub mod lbm_state2s;

pub use self::lbm_state::LBMState;
pub use self::lbm_state2::LBMState2;
pub use self::lbm_state2s::LBMState2s;


const REVERSE: [usize; 9] = [0, 2, 1, 4, 3, 8, 7, 6, 5];

#[allow(unused_imports)]
pub trait Fluid {
    fn init(&mut self);
    fn update(&mut self);
    fn draw(&self, pixels: &mut Vec<u8>);
    fn left_click(&mut self, x: f64, y: f64);
    fn right_click(&mut self, x: f64, y: f64);
}

fn roll<A>(a: &ArrayView2<A>, dir: &[isize; 2], elem: A) -> Array2<A>
    where A: Clone {
    let mut b = Array::from_elem(a.dim(), elem);
    let x = dir[0];
    let y = dir[1];
    if x == 0 {
        if y == 0 {
            b.assign(&a);
        } else {
            b.slice_mut(s![.., y..]).assign(&a.slice(s![.., ..-y]));
            b.slice_mut(s![.., ..y]).assign(&a.slice(s![.., -y..]));
        }
    } else {
        if y == 0 {
            b.slice_mut(s![x.., ..]).assign(&a.slice(s![..-x, ..]));
            b.slice_mut(s![..x, ..]).assign(&a.slice(s![-x.., ..]));
        } else {
            b.slice_mut(s![x.., y..]).assign(&a.slice(s![..-x, ..-y]));
            b.slice_mut(s![..x, y..]).assign(&a.slice(s![-x.., ..-y]));
            b.slice_mut(s![x.., ..y]).assign(&a.slice(s![..-x, -y..]));
            b.slice_mut(s![..x, ..y]).assign(&a.slice(s![-x.., -y..]));
        }
    }
    b
}

fn curl(ux: &ArrayView2<f64>, uy: &ArrayView2<f64>) -> Array2<f64> {
    let ux_r = roll(ux, &[1, 0], 0.0);
    let ux_l = roll(ux, &[-1, 0], 0.0);
    let uy_r = roll(uy, &[0, 1], 0.0);
    let uy_l = roll(uy, &[0, -1], 0.0);
    ux_r - ux_l - uy_r + uy_l
}

fn grad(x: &ArrayView2<f64>, dx: f64) -> Array2<f64> {
    let mut r = Array::zeros(x.dim());
    let ux_r = roll(x, &[1, 0], 0.0);
    let uy_r = roll(x, &[0, 1], 0.0);
    ndarray::Zip::from(&mut r)
        .and(x)
        .and(&ux_r)
        .and(&uy_r)
        .par_apply(|r, &x, &xx, &xy| {
            *r = (xx - x + xy - x) / dx;
        });
    r
}

fn nabla(x: &ArrayView2<f64>, dx: f64) -> (Array2<f64>, Array2<f64>) {
    let mut r1 = Array::zeros(x.dim());
    let mut r2 = Array::zeros(x.dim());
    let ux_l = roll(x, &[1, 0], 0.0);
    let uy_l = roll(x, &[0, 1], 0.0);
    let ux_r = roll(x, &[-1, 0], 0.0);
    let uy_r = roll(x, &[0, -1], 0.0);
    ndarray::Zip::from(&mut r1)
        .and(&mut r2)
        .and(&ux_r)
        .and(&uy_r)
        .and(&ux_l)
        .and(&uy_l)
        .par_apply(|r1, r2, &xx, &xy, &xx_, &xy_| {
            *r1 = (xx - xx_) / dx / 2.0;
            *r2 = (xy - xy_) / dx / 2.0;
        });
    (r1, r2)
}

fn lapla(x: &ArrayView2<f64>, dx: f64) -> Array2<f64> {
    let mut r = Array::zeros(x.dim());
    let ux_r = roll(x, &[1, 0], 0.0);
    let ux_l = roll(x, &[-1, 0], 0.0);
    let uy_r = roll(x, &[0, 1], 0.0);
    let uy_l = roll(x, &[0, -1], 0.0);
    ndarray::Zip::from(&mut r)
        .and(x)
        .and(&ux_r)
        .and(&uy_r)
        .and(&ux_l)
        .and(&uy_l)
        .par_apply(|r, &x, &xx, &xy, &xxl, &xyl| {
            *r = (xx + xy + xxl + xyl - 4.0 * x) / dx / dx;
        });
    r
}

fn flow(a: &mut Array3<f64>, block: &Array3<bool>, e: [[isize; 2]; 9]) {
    let mut b = Array::zeros(a.dim());
    for i in 0..9 {
        let c = roll(&a.subview(Axis(0), i), &e[i], 0.0);
        b.subview_mut(Axis(0), i).assign(&c);
    };
    ndarray::Zip::from(b.gencolumns_mut())
        .and(a.gencolumns())
        .and(block.gencolumns())
        .par_apply(|mut x: ArrayViewMut1<f64>, y: ArrayView1<f64>, b: ArrayView1<bool>| {
            for i in 0..9 {
                if b[i] {
                    x[i] = y[REVERSE[i]];
                }
            }
        });
    *a = b;
}